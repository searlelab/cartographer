import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import torch

from sculptor_model import initialize_sculptor_model
from sculptor_settings import max_peptide_len, metadata_filename
from sculptor_tensorize import codedseq_to_array, return_charge_onehot, unimod_to_codedseq


# Match tokens like "UNIMOD:35" inside Cartographer-formatted strings.
UNIMOD_PATTERN = re.compile( r'UNIMOD:\d+' )
RESIDUE_UNIMOD_PATTERN = re.compile( r'([A-Z])\[(UNIMOD:\d+)\]' )
DEFAULT_INPUT_CSV = '/Users/searle.brian/Documents/testing/trainingdata/union_ccs.csv'
CANONICAL_CARBAMIDOMETHYL = 'C[UNIMOD:4]'


class RunningStats( object ):
    def __init__( self ):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update( self, value ):
        x = float( value )
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std( self ):
        if self.n < 2:
            return 0.0
        return math.sqrt( self.m2 / ( self.n - 1 ) )


class ErrorStats( object ):
    def __init__( self ):
        self.n = 0
        self.abs_err_sum = 0.0
        self.sq_err_sum = 0.0

    def update( self, error_value ):
        e = float( error_value )
        self.n += 1
        self.abs_err_sum += abs( e )
        self.sq_err_sum += e * e

    def mae( self ):
        if self.n == 0:
            return 0.0
        return self.abs_err_sum / self.n

    def rmse( self ):
        if self.n == 0:
            return 0.0
        return math.sqrt( self.sq_err_sum / self.n )


def choose_split( modified_sequence, test_fraction ):
    digest = hashlib.md5( modified_sequence.encode( 'utf-8' ) ).hexdigest()
    bucket = int( digest, 16 ) % 10000000
    threshold = int( test_fraction * 10000000 )
    return 'test' if bucket < threshold else 'train'


def parse_args( args ):
    parser = argparse.ArgumentParser(
        description='Summarize Sculptor training-data slice diagnostics'
    )
    parser.add_argument( '--input_csv',
                         type=str,
                         default=DEFAULT_INPUT_CSV,
                         help='Path to union_ccs.csv' )
    parser.add_argument( '--split',
                         type=str,
                         default='train',
                         choices=[ 'train', 'test', 'all' ],
                         help='Which split to summarize' )
    parser.add_argument( '--test_fraction',
                         type=float,
                         default=0.2,
                         help='Split fraction used during dataset prep' )
    parser.add_argument( '--max_rows',
                         type=int,
                         default=None,
                         help='Optional row cap for fast diagnostics' )
    parser.add_argument( '--output_file',
                         type=str,
                         default=None,
                         help='Output markdown path (default next to input csv)' )
    parser.add_argument( '--model_file',
                         type=str,
                         default=None,
                         help='Optional Sculptor checkpoint for per-PTM error rates' )
    parser.add_argument( '--dataset_root',
                         type=str,
                         default=None,
                         help='Dataset root used to load metadata for CCS denormalization' )
    parser.add_argument( '--metadata_file',
                         type=str,
                         default=None,
                         help='Optional metadata JSON override (default: dataset_root/' + metadata_filename + ')' )
    parser.add_argument( '--arch_json',
                         type=str,
                         default=None,
                         help='Optional architecture JSON string or JSON file path for model init overrides' )
    parser.add_argument( '--device',
                         type=str,
                         default='auto',
                         help='Inference device {auto, mps, cuda, cpu}' )
    parser.add_argument( '--pred_batch_size',
                         type=int,
                         default=2048,
                         help='Batch size for per-PTM model inference' )
    parser.add_argument( '--sufficient_min_entries',
                         type=int,
                         default=1000,
                         help='Minimum entries to label PTM support as sufficient' )
    parser.add_argument( '--moderate_min_entries',
                         type=int,
                         default=200,
                         help='Minimum entries to label PTM support as moderate' )
    return parser.parse_args( args )


def length_bin( length ):
    if length <= 10:
        return '07-10'
    if length <= 15:
        return '11-15'
    if length <= 20:
        return '16-20'
    if length <= 25:
        return '21-25'
    if length <= 30:
        return '26-30'
    if length <= 40:
        return '31-40'
    return '41-50'


def should_keep_split( split_name, selected ):
    if selected == 'all':
        return True
    return split_name == selected


def parse_arch_overrides( raw ):
    if raw is None:
        return None

    text = raw
    if os.path.isfile( raw ):
        with open( raw, 'r' ) as f:
            text = f.read()

    payload = json.loads( text )
    if isinstance( payload, dict ):
        if 'arch' in payload and isinstance( payload[ 'arch' ], dict ):
            return payload[ 'arch' ]
        if 'designs' in payload and isinstance( payload[ 'designs' ], list ) and len( payload[ 'designs' ] ) > 0:
            first = payload[ 'designs' ][ 0 ]
            if isinstance( first, dict ) and 'arch' in first and isinstance( first[ 'arch' ], dict ):
                return first[ 'arch' ]
        return payload
    raise ValueError( '--arch_json must decode to a JSON object' )


def select_device( raw_device ):
    device = str( raw_device ).strip().lower()
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr( torch.backends, 'mps' ) and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    if device not in [ 'cpu', 'cuda', 'mps' ]:
        raise ValueError( '--device must be one of {auto, mps, cuda, cpu}' )
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError( 'CUDA requested but not available' )
    if device == 'mps':
        if not hasattr( torch.backends, 'mps' ) or not torch.backends.mps.is_available():
            raise RuntimeError( 'MPS requested but not available' )
    return device


def load_metadata_for_eval( dataset_root=None, metadata_file=None ):
    candidates = []
    if metadata_file is not None:
        candidates.append( metadata_file )
    if dataset_root is not None:
        candidates.append( os.path.join( dataset_root, metadata_filename ) )

    for path in candidates:
        if os.path.isfile( path ):
            with open( path, 'r' ) as f:
                metadata = json.load( f )
            ccs_mean = float( metadata[ 'train_ccs_mean' ] )
            ccs_std = float( metadata[ 'train_ccs_std' ] )
            if ccs_std <= 0.0:
                ccs_std = 1.0
            return ccs_mean, ccs_std, os.path.abspath( path )

    raise FileNotFoundError( 'Could not locate metadata JSON for model evaluation. '
                             'Pass --dataset_root or --metadata_file.' )


def extract_modification_pairs( modified_sequence ):
    parts = modified_sequence.split( '-', 2 )
    if len( parts ) == 3:
        nterm_part, body, _ = parts
    else:
        nterm_part = '[]'
        body = modified_sequence

    pairs = []
    if nterm_part not in [ '[]', '' ]:
        tag = nterm_part.strip( '[]' )
        if tag != '':
            pairs.append( 'NTERM[' + tag + ']' )

    for aa, tag in RESIDUE_UNIMOD_PATTERN.findall( body ):
        pairs.append( aa + '[' + tag + ']' )

    return pairs


def pair_to_tag( pair ):
    left = pair.find( '[' )
    right = pair.find( ']' )
    if left >= 0 and right > left:
        return pair[ left + 1 : right ]
    return pair


def support_band( count, sufficient_min_entries, moderate_min_entries ):
    if count >= sufficient_min_entries:
        return 'sufficient'
    if count >= moderate_min_entries:
        return 'moderate'
    return 'sparse'


def write_markdown( output_file,
                    input_csv,
                    selected_split,
                    test_fraction,
                    total_rows,
                    split_selected_rows,
                    accepted_rows,
                    skip_counts,
                    charge_stats,
                    len_counter,
                    len_bin_counter,
                    ptm_counter,
                    ptm_pair_counter,
                    multi_mod_combo_counter,
                    model_eval_summary,
                    error_by_pair,
                    error_by_tag,
                    sufficient_min_entries,
                    moderate_min_entries, ):
    with open( output_file, 'w' ) as out:
        out.write( '# Sculptor Slice Diagnostics\n\n' )
        out.write( '- generated_at_utc: ' + datetime.now( timezone.utc ).strftime( '%Y-%m-%dT%H:%M:%SZ' ) + '\n' )
        out.write( '- input_csv: `' + os.path.abspath( input_csv ) + '`\n' )
        out.write( '- selected_split: `' + selected_split + '`\n' )
        out.write( '- test_fraction: `' + str(test_fraction) + '`\n' )
        out.write( '- max_peptide_len: `' + str(max_peptide_len) + '`\n\n' )

        out.write( '## Overview\n\n' )
        out.write( '- rows_scanned: `' + str(total_rows) + '`\n' )
        out.write( '- rows_in_selected_split: `' + str(split_selected_rows) + '`\n' )
        out.write( '- accepted_rows: `' + str(accepted_rows) + '`\n' )
        out.write( '- rejected_rows: `' + str(split_selected_rows - accepted_rows) + '`\n\n' )

        out.write( '## Rejection Reasons\n\n' )
        out.write( '| reason | count |\n' )
        out.write( '| --- | ---: |\n' )
        if len( skip_counts ) == 0:
            out.write( '| (none) | 0 |\n' )
        else:
            for reason, count in sorted( skip_counts.items(), key=lambda kv: (-kv[1], kv[0]) ):
                out.write( '| ' + reason + ' | ' + str(count) + ' |\n' )
        out.write( '\n' )

        out.write( '## Charge Slices\n\n' )
        out.write( '| charge | count | mean_ccs | std_ccs |\n' )
        out.write( '| ---: | ---: | ---: | ---: |\n' )
        for z in sorted( charge_stats ):
            stats = charge_stats[ z ]
            out.write( '| ' + str(z) +
                       ' | ' + str(stats.n) +
                       ' | ' + format( stats.mean, '.6f' ) +
                       ' | ' + format( stats.std(), '.6f' ) +
                       ' |\n' )
        out.write( '\n' )

        out.write( '## Length Bins\n\n' )
        out.write( '| bin | count |\n' )
        out.write( '| --- | ---: |\n' )
        ordered_bins = [ '07-10', '11-15', '16-20', '21-25', '26-30', '31-40', '41-50' ]
        for b in ordered_bins:
            out.write( '| ' + b + ' | ' + str(len_bin_counter.get( b, 0 )) + ' |\n' )
        out.write( '\n' )

        out.write( '## Exact Length Counts\n\n' )
        out.write( '| length | count |\n' )
        out.write( '| ---: | ---: |\n' )
        for length in sorted( len_counter ):
            out.write( '| ' + str(length) + ' | ' + str(len_counter[length]) + ' |\n' )
        out.write( '\n' )

        out.write( '## PTM Presence\n\n' )
        out.write( '| PTM tag | count |\n' )
        out.write( '| --- | ---: |\n' )
        for tag, count in sorted( ptm_counter.items(), key=lambda kv: (-kv[1], kv[0]) ):
            out.write( '| ' + tag + ' | ' + str(count) + ' |\n' )
        out.write( '\n' )

        out.write( '## PTM Pair Entries\n\n' )
        out.write( '- counting rule: occurrence-level PTM pairs per peptide; '
                   '`C[UNIMOD:4]` is treated as canonical and excluded from PTM pair counts\n\n' )
        out.write( '| PTM pair | count | support_band |\n' )
        out.write( '| --- | ---: | --- |\n' )
        for pair, count in sorted( ptm_pair_counter.items(), key=lambda kv: (-kv[1], kv[0]) ):
            band = support_band( count, sufficient_min_entries, moderate_min_entries )
            out.write( '| ' + pair + ' | ' + str(count) + ' | ' + band + ' |\n' )
        out.write( '\n' )

        out.write( '## Multi-PTM Enumeration (>2 non-canonical PTM types)\n\n' )
        out.write( '| PTM type combination | peptide_rows |\n' )
        out.write( '| --- | ---: |\n' )
        if len( multi_mod_combo_counter ) == 0:
            out.write( '| (none) | 0 |\n' )
        else:
            for combo, count in sorted( multi_mod_combo_counter.items(), key=lambda kv: (-kv[1], kv[0]) ):
                out.write( '| ' + combo + ' | ' + str(count) + ' |\n' )
        out.write( '\n' )

        if model_eval_summary is not None:
            out.write( '## Per-PTM Error Rates\n\n' )
            out.write( '- model_file: `' + model_eval_summary[ 'model_file' ] + '`\n' )
            out.write( '- metadata_file: `' + model_eval_summary[ 'metadata_file' ] + '`\n' )
            out.write( '- eval_device: `' + model_eval_summary[ 'eval_device' ] + '`\n' )
            out.write( '- pred_batch_size: `' + str(model_eval_summary[ 'pred_batch_size' ]) + '`\n' )
            out.write( '- ccs_mean: `' + format( model_eval_summary[ 'ccs_mean' ], '.6f' ) + '`\n' )
            out.write( '- ccs_std: `' + format( model_eval_summary[ 'ccs_std' ], '.6f' ) + '`\n' )
            out.write( '- eligible_rows_for_error: `' + str(model_eval_summary[ 'eligible_rows_for_error' ]) + '`\n' )
            out.write( '- skipped_rows_gt2_noncanonical_types: `' +
                       str(model_eval_summary[ 'overflow_rows_for_error' ]) + '`\n\n' )

            out.write( '### PTM Pair Error (eligible rows)\n\n' )
            out.write( '| PTM pair | n_entries | MAE CCS | RMSE CCS | support_band |\n' )
            out.write( '| --- | ---: | ---: | ---: | --- |\n' )
            if len( error_by_pair ) == 0:
                out.write( '| (none) | 0 | 0.000000 | 0.000000 | sparse |\n' )
            else:
                pair_rows = []
                for pair, stats in error_by_pair.items():
                    pair_rows.append( (pair, stats.n, stats.mae(), stats.rmse()) )
                pair_rows.sort( key=lambda x: (-x[1], x[2], x[0]) )
                for pair, n_entries, mae, rmse in pair_rows:
                    band = support_band( n_entries, sufficient_min_entries, moderate_min_entries )
                    out.write( '| ' + pair +
                               ' | ' + str(n_entries) +
                               ' | ' + format( mae, '.6f' ) +
                               ' | ' + format( rmse, '.6f' ) +
                               ' | ' + band + ' |\n' )
            out.write( '\n' )

            out.write( '### PTM Type Error (eligible rows)\n\n' )
            out.write( '| PTM type | n_entries | MAE CCS | RMSE CCS | support_band |\n' )
            out.write( '| --- | ---: | ---: | ---: | --- |\n' )
            if len( error_by_tag ) == 0:
                out.write( '| (none) | 0 | 0.000000 | 0.000000 | sparse |\n' )
            else:
                tag_rows = []
                for tag, stats in error_by_tag.items():
                    tag_rows.append( (tag, stats.n, stats.mae(), stats.rmse()) )
                tag_rows.sort( key=lambda x: (-x[1], x[2], x[0]) )
                for tag, n_entries, mae, rmse in tag_rows:
                    band = support_band( n_entries, sufficient_min_entries, moderate_min_entries )
                    out.write( '| ' + tag +
                               ' | ' + str(n_entries) +
                               ' | ' + format( mae, '.6f' ) +
                               ' | ' + format( rmse, '.6f' ) +
                               ' | ' + band + ' |\n' )


def main():
    args = parse_args( os.sys.argv[1:] )
    if not os.path.isfile( args.input_csv ):
        raise FileNotFoundError( 'Input CSV not found: ' + args.input_csv )
    if not ( 0.0 < args.test_fraction < 1.0 ):
        raise ValueError( '--test_fraction must be in (0,1)' )
    if args.pred_batch_size <= 0:
        raise ValueError( '--pred_batch_size must be > 0' )
    if args.moderate_min_entries < 1 or args.sufficient_min_entries < 1:
        raise ValueError( 'support thresholds must be >= 1' )
    if args.moderate_min_entries > args.sufficient_min_entries:
        raise ValueError( '--moderate_min_entries cannot exceed --sufficient_min_entries' )

    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join( os.path.dirname( os.path.abspath( args.input_csv ) ),
                                    'sculptor_slice_diagnostics_' + args.split + '.md' )

    total_rows = 0
    split_selected_rows = 0
    accepted_rows = 0
    skip_counts = Counter()
    charge_stats = defaultdict( RunningStats )
    len_counter = Counter()
    len_bin_counter = Counter()
    ptm_counter = Counter()
    ptm_pair_counter = Counter()
    multi_mod_combo_counter = Counter()

    model = None
    model_eval_summary = None
    error_by_pair = defaultdict( ErrorStats )
    error_by_tag = defaultdict( ErrorStats )

    batch_seq = []
    batch_charge = []
    batch_true_ccs = []
    batch_pairs = []
    batch_overflow = []

    if args.model_file is not None:
        if not os.path.isfile( args.model_file ):
            raise FileNotFoundError( '--model_file not found: ' + args.model_file )

        ccs_mean, ccs_std, metadata_path = load_metadata_for_eval( args.dataset_root, args.metadata_file )
        arch_overrides = parse_arch_overrides( args.arch_json )
        eval_device = select_device( args.device )

        model = initialize_sculptor_model( model_file=args.model_file,
                                           arch_overrides=arch_overrides,
                                           map_location='cpu' )
        model = model.to( eval_device )
        model.eval()

        model_eval_summary = { 'model_file' : os.path.abspath( args.model_file ),
                               'metadata_file' : metadata_path,
                               'eval_device' : eval_device,
                               'pred_batch_size' : int( args.pred_batch_size ),
                               'ccs_mean' : float( ccs_mean ),
                               'ccs_std' : float( ccs_std ),
                               'eligible_rows_for_error' : 0,
                               'overflow_rows_for_error' : 0, }

    def flush_prediction_batch():
        if model is None or len( batch_seq ) == 0:
            return

        seq_tensor = torch.as_tensor( np.asarray( batch_seq, dtype='int64' ),
                                      dtype=torch.long,
                                      device=model_eval_summary[ 'eval_device' ] )
        charge_tensor = torch.as_tensor( np.asarray( batch_charge, dtype='float32' ),
                                         dtype=torch.float32,
                                         device=model_eval_summary[ 'eval_device' ] )
        true_tensor = torch.as_tensor( np.asarray( batch_true_ccs, dtype='float32' ),
                                       dtype=torch.float32,
                                       device=model_eval_summary[ 'eval_device' ] )

        with torch.no_grad():
            pred_norm = model( seq_tensor, charge_tensor ).squeeze( -1 )
            pred_ccs = pred_norm * model_eval_summary[ 'ccs_std' ] + model_eval_summary[ 'ccs_mean' ]
            errors = ( pred_ccs - true_tensor ).detach().cpu().numpy()

        for i, row_error in enumerate( errors ):
            if batch_overflow[ i ]:
                continue

            pairs = batch_pairs[ i ]
            if len( pairs ) == 0:
                error_by_pair[ 'UNMODIFIED' ].update( row_error )
                error_by_tag[ 'UNMODIFIED' ].update( row_error )
            else:
                for pair in pairs:
                    error_by_pair[ pair ].update( row_error )
                    error_by_tag[ pair_to_tag( pair ) ].update( row_error )

        batch_seq.clear()
        batch_charge.clear()
        batch_true_ccs.clear()
        batch_pairs.clear()
        batch_overflow.clear()

    with open( args.input_csv, 'r', newline='' ) as handle:
        reader = csv.DictReader( handle )
        required = { 'modified_sequence', 'charge', 'ccs' }
        if not required.issubset( set( reader.fieldnames or [] ) ):
            raise ValueError( 'Input CSV missing one of required columns: ' + str( sorted( required ) ) )

        for row in reader:
            if args.max_rows is not None and total_rows >= args.max_rows:
                break

            total_rows += 1
            modified_sequence = row[ 'modified_sequence' ].strip()
            split_name = choose_split( modified_sequence, args.test_fraction )
            if not should_keep_split( split_name, args.split ):
                continue
            split_selected_rows += 1

            try:
                ccs = float( row[ 'ccs' ] )
                charge = int( row[ 'charge' ] )
            except Exception:
                skip_counts[ 'bad_numeric' ] += 1
                continue

            coded = unimod_to_codedseq( modified_sequence,
                                        max_len=max_peptide_len,
                                        skip_counts=skip_counts )
            if coded is None:
                continue

            accepted_rows += 1
            peptide_len = len( coded ) - 2

            charge_stats[ charge ].update( ccs )
            len_counter[ peptide_len ] += 1
            len_bin_counter[ length_bin( peptide_len ) ] += 1

            tags = set( UNIMOD_PATTERN.findall( modified_sequence ) )
            if len( tags ) == 0:
                ptm_counter[ 'UNMODIFIED' ] += 1
            else:
                for tag in tags:
                    ptm_counter[ tag ] += 1

            mod_pairs = extract_modification_pairs( modified_sequence )
            noncanonical_pairs = [ p for p in mod_pairs if p != CANONICAL_CARBAMIDOMETHYL ]
            if len( noncanonical_pairs ) == 0:
                ptm_pair_counter[ 'UNMODIFIED' ] += 1
            else:
                for pair in noncanonical_pairs:
                    ptm_pair_counter[ pair ] += 1

            noncanonical_types = sorted( set( [ pair_to_tag( pair ) for pair in noncanonical_pairs ] ) )
            is_overflow = len( noncanonical_types ) > 2
            if is_overflow:
                if len( noncanonical_types ) == 0:
                    combo_name = '(none)'
                else:
                    combo_name = ' + '.join( noncanonical_types )
                multi_mod_combo_counter[ combo_name ] += 1

            if model_eval_summary is not None:
                if is_overflow:
                    model_eval_summary[ 'overflow_rows_for_error' ] += 1
                else:
                    model_eval_summary[ 'eligible_rows_for_error' ] += 1

                seq_tokens = codedseq_to_array( coded, max_size=max_peptide_len + 2 )
                charge_onehot = return_charge_onehot( charge )

                batch_seq.append( seq_tokens )
                batch_charge.append( charge_onehot )
                batch_true_ccs.append( ccs )
                batch_pairs.append( noncanonical_pairs )
                batch_overflow.append( is_overflow )

                if len( batch_seq ) >= args.pred_batch_size:
                    flush_prediction_batch()

    flush_prediction_batch()

    write_markdown( output_file,
                    args.input_csv,
                    args.split,
                    args.test_fraction,
                    total_rows,
                    split_selected_rows,
                    accepted_rows,
                    skip_counts,
                    charge_stats,
                    len_counter,
                    len_bin_counter,
                    ptm_counter,
                    ptm_pair_counter,
                    multi_mod_combo_counter,
                    model_eval_summary,
                    error_by_pair,
                    error_by_tag,
                    args.sufficient_min_entries,
                    args.moderate_min_entries, )

    print( 'Wrote slice diagnostics: ' + output_file )
    print( 'Rows scanned: ' + str(total_rows) +
           ' | selected split rows: ' + str(split_selected_rows) +
           ' | accepted rows: ' + str(accepted_rows) )
    if model_eval_summary is not None:
        print( 'Model PTM error rows (eligible): ' + str(model_eval_summary[ 'eligible_rows_for_error' ]) +
               ' | skipped_gt2_types: ' + str(model_eval_summary[ 'overflow_rows_for_error' ]) )


if __name__ == '__main__':
    main()
