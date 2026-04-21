import argparse
import glob
import math
import os
import re
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import torch

from cartographer_model import initialize_cartographer_model
from cartographer_settings import max_peptide_len as cartographer_max_len
from tensorize import codedseq_to_array, unimod_to_codedseq
from training_loop import resolve_device


UNIMOD_PATTERN = re.compile( r'UNIMOD:\d+' )
NCE_NORMALIZATION_SCALE = 100.0

UNIMOD_TO_MOD = {
    'UNIMOD:1' : 'Acetyl',
    'UNIMOD:3' : 'Biotin',
    'UNIMOD:4' : 'Carbamidomethyl',
    'UNIMOD:7' : 'Deamidation',
    'UNIMOD:21' : 'Phospho',
    'UNIMOD:27' : 'Pyro-Glu',
    'UNIMOD:28' : 'Pyro-Glu',
    'UNIMOD:34' : 'Methyl',
    'UNIMOD:35' : 'Oxidation',
    'UNIMOD:36' : 'Dimethyl',
    'UNIMOD:37' : 'Trimethyl',
    'UNIMOD:43' : 'HexNAc',
    'UNIMOD:58' : 'Propionyl',
    'UNIMOD:64' : 'Succinyl',
    'UNIMOD:121' : 'GlyGly (Ub)',
    'UNIMOD:122' : 'Formyl',
    'UNIMOD:312' : 'Cysteinyl',
    'UNIMOD:354' : 'Nitro',
    'UNIMOD:408' : 'Glycosyl hydroxyproline',
    'UNIMOD:737' : 'TMT6plex',
    'UNIMOD:739' : 'TMT0',
    'UNIMOD:747' : 'Malonyl',
    'UNIMOD:1289' : 'Butyryl',
    'UNIMOD:1363' : 'Crotonyl',
    'UNIMOD:1848' : 'Glutarylation',
    'UNIMOD:1849' : 'Hydroxyisobutyryl',
}


def parse_args( args ):
    parser = argparse.ArgumentParser(
        description='Report per-spectrum spectral contrast angles by PTM group on Cartographer hold-out test shards.'
    )
    parser.add_argument( '--model_file',
                         type=str,
                         required=True,
                         help='Cartographer model checkpoint (.pt state_dict)' )
    parser.add_argument( '--data_root',
                         type=str,
                         default='/Users/searle.brian/Documents/huggingface/data',
                         help=( 'Hugging Face data root (containing prospect-ptms-ms2/) or '
                                'direct prospect-ptms-ms2 dataset root' ) )
    parser.add_argument( '--output_file',
                         type=str,
                         default='models/cartographer_ptm_contrast_angles_by_spectrum.tsv',
                         help='TSV report path (group_name, nce, contrast_angle)' )
    parser.add_argument( '--batch_size',
                         type=int,
                         default=2048,
                         help='Batch size for model inference' )
    parser.add_argument( '--device',
                         type=str,
                         default='auto',
                         help='Inference device {auto,cuda,mps,cpu}' )
    parser.add_argument( '--max_rows',
                         type=int,
                         default=0,
                         help='Optional cap on evaluated test rows (0 = no limit)' )
    parser.add_argument( '--target_nce',
                         type=float,
                         default=None,
                         help=( 'If set, keep only one spectrum per unique modified_sequence: '
                                'the row with NCE (e.g. 30) closest to this value' ) )
    parser.add_argument( '--log_every',
                         type=int,
                         default=250000,
                         help='Log progress every N evaluated spectra' )
    return parser.parse_args( args )


def discover_test_files( dataset_root ):
    pattern = os.path.join( dataset_root, 'data', 'test-*.parquet' )
    return sorted( glob.glob( pattern ) )


def resolve_cartographer_dataset_root( data_root ):
    root = os.path.abspath( data_root )
    candidates = [ root, os.path.join( root, 'prospect-ptms-ms2' ) ]
    for candidate in candidates:
        if len( discover_test_files( candidate ) ) > 0:
            return candidate
    raise FileNotFoundError(
        'Could not locate Cartographer test parquet files under '
        + root
        + ' (checked '
        + ', '.join( candidates )
        + ')'
    )


def classify_ptm_group( modified_sequence ):
    tags = UNIMOD_PATTERN.findall( str( modified_sequence ) )
    noncanonical = set()
    for tag in tags:
        mod_name = UNIMOD_TO_MOD.get( tag, tag )
        if mod_name != 'Carbamidomethyl':
            noncanonical.add( mod_name )

    if len( noncanonical ) == 0:
        return 'Unmodified'
    if len( noncanonical ) == 1:
        return sorted( noncanonical )[0]
    return 'mixed'


def spectral_contrast_angles( pred, true, eps=1e-12 ):
    ion_mask = ( true >= 0.0 ).to( pred.dtype )
    pred_masked = pred * ion_mask
    true_masked = true * ion_mask

    pred_denom = torch.sqrt( torch.sum( pred_masked ** 2, dim=1, keepdim=True ).clamp( min=eps ) )
    true_denom = torch.sqrt( torch.sum( true_masked ** 2, dim=1, keepdim=True ).clamp( min=eps ) )

    pred_norm = pred_masked / pred_denom
    true_norm = true_masked / true_denom

    cosine = torch.sum( pred_norm * true_norm, dim=1 ).clamp( -1.0, 1.0 )
    return 1.0 - ( 2.0 * torch.acos( cosine ) / math.pi )


def nce_to_aligned_normed( nce ):
    return float( nce ) / NCE_NORMALIZATION_SCALE


def aligned_normed_to_nce( aligned_normed ):
    return float( aligned_normed ) * NCE_NORMALIZATION_SCALE


def should_replace_target_nce_match( current_best, candidate_nce, target_nce, eps=1e-12 ):
    candidate_nce = float( candidate_nce )
    candidate_delta = abs( candidate_nce - float( target_nce ) )
    if current_best is None:
        return True

    best_delta = float( current_best[0] )
    best_nce = float( current_best[1] )

    if candidate_delta + eps < best_delta:
        return True
    if abs( candidate_delta - best_delta ) <= eps and candidate_nce + eps < best_nce:
        return True
    return False


def select_target_nce_rows( test_files, target_nce, log_every ):
    best_by_modseq = {}
    scanned = 0
    next_log = max( int( log_every ), 1 )

    for file_idx, path in enumerate( test_files ):
        print( 'Scanning shard for target NCE selection ' + str( file_idx + 1 ) + '/' + str( len( test_files ) ) +
               ': ' + os.path.basename( path ),
               flush=True )
        pf = pq.ParquetFile( path )
        for rg_idx in range( pf.metadata.num_row_groups ):
            table = pf.read_row_group( rg_idx, columns=[ 'modified_sequence', 'collision_energy_aligned_normed' ] )
            mod_seqs = table.column( 'modified_sequence' ).to_pylist()
            nces = table.column( 'collision_energy_aligned_normed' ).to_pylist()

            for row_idx in range( len( mod_seqs ) ):
                mod_seq = str( mod_seqs[ row_idx ] )
                nce_value = float( nces[ row_idx ] )
                current = best_by_modseq.get( mod_seq, None )
                if should_replace_target_nce_match( current, nce_value, target_nce ):
                    delta = abs( nce_value - float( target_nce ) )
                    best_by_modseq[ mod_seq ] = ( delta, nce_value, file_idx, rg_idx, row_idx )

                scanned += 1

            while scanned >= next_log:
                print( 'Scanned=' + str( scanned ) +
                       ' unique_modified_sequences=' + str( len( best_by_modseq ) ),
                       flush=True )
                next_log += max( int( log_every ), 1 )

    selected_by_rowgroup = {}
    for _, _, file_idx, rg_idx, row_idx in best_by_modseq.values():
        key = ( int( file_idx ), int( rg_idx ) )
        if key not in selected_by_rowgroup:
            selected_by_rowgroup[ key ] = set()
        selected_by_rowgroup[ key ].add( int( row_idx ) )

    return selected_by_rowgroup, len( best_by_modseq ), scanned


def main():
    import sys

    args = parse_args( sys.argv[1:] )

    device = resolve_device( args.device )
    dataset_root = resolve_cartographer_dataset_root( args.data_root )
    test_files = discover_test_files( dataset_root )
    if len( test_files ) == 0:
        raise RuntimeError( 'No test parquet files found in ' + dataset_root )

    limit = int( args.max_rows ) if int( args.max_rows ) > 0 else None

    print( 'Model: ' + os.path.abspath( args.model_file ) )
    print( 'Dataset root: ' + dataset_root )
    print( 'Test shards: ' + str( len(test_files) ) )
    print( 'Device: ' + device )
    if limit is not None:
        print( 'Row limit: ' + str( limit ) )
    if args.target_nce is not None:
        print( 'Target NCE mode: nearest spectrum per modified_sequence to NCE=' +
               format( float( args.target_nce ), '.6f' ) +
               ' (aligned_normed=' + format( nce_to_aligned_normed( args.target_nce ), '.6f' ) + ')' )

    selected_rows_by_rowgroup = None
    if args.target_nce is not None:
        target_nce_aligned_normed = nce_to_aligned_normed( args.target_nce )
        selected_rows_by_rowgroup, n_unique, n_scanned = select_target_nce_rows( test_files,
                                                                                  target_nce_aligned_normed,
                                                                                  args.log_every )
        print( 'Target NCE selection complete: scanned=' + str( n_scanned ) +
               ' selected_unique_modified_sequences=' + str( n_unique ),
               flush=True )

    model = initialize_cartographer_model( frag_type='beam', model_file=None )
    state = torch.load( args.model_file, map_location='cpu' )
    model.load_state_dict( state, strict=True )
    model = model.to( device )
    model.eval()

    group_counts = Counter()
    skipped_tokenization = 0
    evaluated = 0
    next_log = max( int( args.log_every ), 1 )

    seq_batch = []
    charge_batch = []
    nce_batch = []
    true_batch = []
    group_batch = []

    output_path = os.path.abspath( args.output_file )
    output_dir = os.path.dirname( output_path )
    if output_dir != '':
        os.makedirs( output_dir, exist_ok=True )
    out_handle = open( output_path, 'w' )
    out_handle.write( 'group_name\tnce\tcontrast_angle\n' )

    def flush():
        nonlocal evaluated
        nonlocal next_log
        if len( seq_batch ) == 0:
            return

        seq_t = torch.as_tensor( np.asarray( seq_batch, dtype='int64' ), dtype=torch.long, device=device )
        charge_t = torch.as_tensor( np.asarray( charge_batch, dtype='float32' ), dtype=torch.float32, device=device )
        nce_t = torch.as_tensor( np.asarray( nce_batch, dtype='float32' ), dtype=torch.float32, device=device )
        true_t = torch.as_tensor( np.asarray( true_batch, dtype='float32' ), dtype=torch.float32, device=device )

        with torch.no_grad():
            pred = model( seq_t, charge_t, nce_t )
            angles = spectral_contrast_angles( pred, true_t )
            angles_np = angles.detach().cpu().numpy()

        for i, angle in enumerate( angles_np ):
            group = group_batch[ i ]
            nce_value = aligned_normed_to_nce( nce_batch[ i ][0] )
            out_handle.write( group + '\t' + format( nce_value, '.6f' ) + '\t' + format( float(angle), '.6f' ) + '\n' )
            group_counts[ group ] += 1

        evaluated += len( group_batch )
        while evaluated >= next_log:
            print( 'Evaluated=' + str( evaluated ) + ' skipped_tokenization=' + str( skipped_tokenization ), flush=True )
            next_log += max( int( args.log_every ), 1 )

        seq_batch.clear()
        charge_batch.clear()
        nce_batch.clear()
        true_batch.clear()
        group_batch.clear()

    try:
        stop = False
        for file_idx, path in enumerate( test_files ):
            print( 'Reading test shard ' + str( file_idx + 1 ) + '/' + str( len(test_files) ) +
                   ': ' + os.path.basename( path ),
                   flush=True )
            pf = pq.ParquetFile( path )
            for rg_idx in range( pf.metadata.num_row_groups ):
                selected_rows = None
                if selected_rows_by_rowgroup is not None:
                    selected_rows = selected_rows_by_rowgroup.get( ( file_idx, rg_idx ), None )
                    if selected_rows is None:
                        continue

                table = pf.read_row_group( rg_idx,
                                           columns=[ 'modified_sequence',
                                                     'precursor_charge_onehot',
                                                     'collision_energy_aligned_normed',
                                                     'intensities_raw' ] )
                mod_seqs = table.column( 'modified_sequence' ).to_pylist()
                charges = table.column( 'precursor_charge_onehot' ).to_pylist()
                nces = table.column( 'collision_energy_aligned_normed' ).to_pylist()
                intensities = table.column( 'intensities_raw' ).to_pylist()

                for i in range( len( mod_seqs ) ):
                    if selected_rows is not None and i not in selected_rows:
                        continue
                    if limit is not None and ( evaluated + len( seq_batch ) ) >= limit:
                        stop = True
                        break

                    mod_seq = mod_seqs[ i ]
                    coded = unimod_to_codedseq( mod_seq, max_len=cartographer_max_len, skip_counts=None )
                    if coded is None:
                        skipped_tokenization += 1
                        continue

                    seq_tokens = codedseq_to_array( coded, max_size=cartographer_max_len + 2 )
                    seq_batch.append( seq_tokens )
                    charge_batch.append( np.asarray( charges[ i ], dtype='float32' ) )
                    nce_batch.append( np.asarray( [ float( nces[ i ] ) ], dtype='float32' ) )
                    true_batch.append( np.asarray( intensities[ i ], dtype='float32' ) )
                    group_batch.append( classify_ptm_group( mod_seq ) )

                    if len( seq_batch ) >= int( args.batch_size ):
                        flush()

                if stop:
                    break
            if stop:
                break

        flush()
    finally:
        out_handle.close()

    if evaluated == 0:
        raise RuntimeError( 'No rows were evaluated. Check model/data compatibility.' )

    print( 'Report written: ' + output_path )
    print( 'Evaluated rows: ' + str( evaluated ) )
    print( 'Skipped tokenization: ' + str( skipped_tokenization ) )
    print( 'Groups observed: ' + str( len( group_counts ) ) )


if __name__ == '__main__':
    main()
