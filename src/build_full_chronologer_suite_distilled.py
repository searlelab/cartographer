import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import cartographer_settings
from cartographer_model import cartographer_hcd
from electrician_model import initialize_electrician_model
from electrician_settings import max_peptide_len as common_max_len
from ptm_loss_report import (
    initialize_chronologer_from_checkpoint,
    legacy_chronologer_codedseq_to_array,
    legacy_chronologer_modseq_to_codedseq,
    load_chronologer_state_dict,
)
from sculptor_model import initialize_sculptor_model
from sculptor_settings import max_peptide_len as sculptor_max_len
from sculptor_tensorize import codedseq_to_array as sculptor_codedseq_to_array
from sculptor_tensorize import return_charge_onehot as sculptor_return_charge_onehot
from sculptor_tensorize import unimod_to_codedseq as sculptor_unimod_to_codedseq
from tensorize import codedseq_to_array as common_codedseq_to_array
from tensorize import return_charge_array
from tensorize import unimod_to_codedseq as common_unimod_to_codedseq
from training_loop import resolve_device


DEFAULT_INPUT_MARKDOWN = '/Users/searle.brian/Documents/projects/cartographer/all_peptides.md'
DEFAULT_OUTPUT_ROOT = '/Users/searle.brian/Documents/huggingface/data/full_chronologer_suite_distilled'
DEFAULT_CARTOGRAPHER_MODEL = '/Users/searle.brian/Documents/projects/cartographer/models/Cartographer_multistart_best_run01.pt'
DEFAULT_CHRONOLOGER_MODEL = '/Users/searle.brian/Documents/projects/cartographer/models/Chronologer_20220601193755.pt'
DEFAULT_ELECTRICIAN_MODEL = '/Users/searle.brian/Documents/projects/cartographer/models/Electrician_20260225110528.pt'
DEFAULT_SCULPTOR_MODEL = '/Users/searle.brian/Documents/projects/cartographer/models/Sculptor_20260311095327.pt'
DEFAULT_SCULPTOR_METADATA = '/Users/searle.brian/Documents/huggingface/data/IM2Deep_CCS/sculptor_dataset_metadata.json'

OUTPUT_METADATA_NAME = 'distilled_dataset_metadata.json'
OUTPUT_UNSUPPORTED_NAME = 'unsupported_peptides.tsv'

ROW_SCHEMA = pa.schema(
    [
        pa.field( 'modified_sequence', pa.string(), nullable=False ),
        pa.field( 'precursor_charge_onehot', pa.list_( pa.int32() ), nullable=False ),
        pa.field( 'charge_state_dist', pa.list_( pa.float32() ), nullable=False ),
        pa.field( 'collision_energy_aligned_normed', pa.float64(), nullable=False ),
        pa.field( 'indexed_retention_time', pa.float64(), nullable=True ),
        pa.field( 'ccs', pa.float64(), nullable=True ),
        pa.field( 'intensities_raw', pa.list_( pa.float32() ), nullable=True ),
    ]
)

CHRONOLOGER_NTERM_UNIMOD_TO_MASS = {
    'UNIMOD:1' : '+42.01',
    'UNIMOD:737' : '+229.1',
    'UNIMOD:739' : '+224.1',
}

CHRONOLOGER_RESIDUE_UNIMOD_TO_MASS = {
    ('C', 'UNIMOD:4') : '+57.02',
    ('M', 'UNIMOD:35') : '+15.99',
    ('W', 'UNIMOD:35') : '+15.99',
    ('S', 'UNIMOD:21') : '+79.96',
    ('T', 'UNIMOD:21') : '+79.96',
    ('Y', 'UNIMOD:21') : '+79.96',
    ('K', 'UNIMOD:1') : '+42.01',
    ('K', 'UNIMOD:64') : '+100.0',
    ('K', 'UNIMOD:121') : '+114.0',
    ('K', 'UNIMOD:34') : '+14.01',
    ('K', 'UNIMOD:36') : '+28.03',
    ('K', 'UNIMOD:37') : '+42.04',
    ('R', 'UNIMOD:34') : '+14.01',
    ('R', 'UNIMOD:36') : '+28.03',
    ('K', 'UNIMOD:739') : '+224.1',
    ('K', 'UNIMOD:737') : '+229.1',
    ('N', 'UNIMOD:7') : '+0.98',
    ('Q', 'UNIMOD:7') : '+0.98',
    ('R', 'UNIMOD:7') : '+0.98',
    ('N', 'UNIMOD:43') : '+203.0',
    ('S', 'UNIMOD:43') : '+203.0',
    ('T', 'UNIMOD:43') : '+203.0',
}

MARKDOWN_BULLET_RE = re.compile( r'^\-\s+(.*\S)\s*$' )
FULL_MODSEQ_RE = re.compile( r'^\[[^\]]*\]\-.*\-\[[^\]]*\]$' )


class SplitParquetWriter( object ):
    def __init__( self, split_name, output_data_dir, rows_per_shard ):
        self.split_name = split_name
        self.output_data_dir = output_data_dir
        self.rows_per_shard = int( rows_per_shard )
        self.shard_index = 0
        self.rows_written = 0
        self.columns = self._empty_columns()

    def _empty_columns( self ):
        return {
            'modified_sequence' : [],
            'precursor_charge_onehot' : [],
            'charge_state_dist' : [],
            'collision_energy_aligned_normed' : [],
            'indexed_retention_time' : [],
            'ccs' : [],
            'intensities_raw' : [],
        }

    def append( self, row ):
        for key in self.columns:
            self.columns[ key ].append( row[ key ] )
        if len( self.columns[ 'modified_sequence' ] ) >= self.rows_per_shard:
            self.flush()

    def flush( self ):
        n_rows = len( self.columns[ 'modified_sequence' ] )
        if n_rows == 0:
            return
        table = pa.Table.from_pydict( self.columns, schema=ROW_SCHEMA )
        out_name = self.split_name + '-' + format( self.shard_index, '05d' ) + '.parquet'
        out_path = os.path.join( self.output_data_dir, out_name )
        pq.write_table( table, out_path, compression='zstd' )
        self.rows_written += n_rows
        self.shard_index += 1
        self.columns = self._empty_columns()

    def close( self ):
        self.flush()


class UnsupportedWriter( object ):
    def __init__( self, path ):
        self.path = path
        self.handle = open( path, 'w', newline='' )
        self.writer = csv.writer( self.handle, delimiter='\t' )
        self.writer.writerow( [ 'modified_sequence', 'split', 'stage', 'reason', 'written_rows' ] )

    def write( self, modified_sequence, split_name, stage, reason, written_rows ):
        self.writer.writerow( [ modified_sequence, split_name, stage, reason, int( written_rows ) ] )

    def close( self ):
        self.handle.close()


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Build a denormalized distilled Chronologer/Electrician/Sculptor/Cartographer parquet dataset.' )
    parser.add_argument( '--input_markdown', type=str, default=DEFAULT_INPUT_MARKDOWN,
                         help='Markdown peptide list to convert into distilled parquet shards.' )
    parser.add_argument( '--output_root', type=str, default=DEFAULT_OUTPUT_ROOT,
                         help='Output root for data/, metadata JSON, and unsupported TSV.' )
    parser.add_argument( '--cartographer_model', type=str, default=DEFAULT_CARTOGRAPHER_MODEL,
                         help='Cartographer checkpoint path.' )
    parser.add_argument( '--chronologer_model', type=str, default=DEFAULT_CHRONOLOGER_MODEL,
                         help='Chronologer checkpoint path.' )
    parser.add_argument( '--electrician_model', type=str, default=DEFAULT_ELECTRICIAN_MODEL,
                         help='Electrician checkpoint path.' )
    parser.add_argument( '--sculptor_model', type=str, default=DEFAULT_SCULPTOR_MODEL,
                         help='Sculptor checkpoint path.' )
    parser.add_argument( '--sculptor_metadata', type=str, default=DEFAULT_SCULPTOR_METADATA,
                         help='Sculptor metadata JSON containing train_ccs_mean/std.' )
    parser.add_argument( '--rows_per_shard', type=int, default=200000,
                         help='Rows per written parquet shard.' )
    parser.add_argument( '--test_fraction', type=float, default=0.1,
                         help='Peptide-level fraction routed to test via stable hash.' )
    parser.add_argument( '--seed', type=int, default=1337,
                         help='Seed used for deterministic split and NCE offsets.' )
    parser.add_argument( '--device', type=str, default='auto',
                         help='Inference device {auto,cuda,mps,cpu}.' )
    parser.add_argument( '--batch_size_peptides', type=int, default=4096,
                         help='Number of peptide records processed per inference chunk.' )
    parser.add_argument( '--batch_size_fragmentation', type=int, default=8192,
                         help='Number of expanded (peptide, charge, nce) rows inferred per Cartographer batch.' )
    parser.add_argument( '--max_peptides', type=int, default=None,
                         help='Optional cap on the number of markdown peptides to process.' )
    parser.add_argument( '--overwrite', action='store_true',
                         help='Overwrite any existing output shards and reports.' )
    return parser.parse_args( args )


def ensure_output_dirs( output_root, overwrite ):
    output_data_dir = os.path.join( output_root, 'data' )
    os.makedirs( output_data_dir, exist_ok=True )

    metadata_path = os.path.join( output_root, OUTPUT_METADATA_NAME )
    unsupported_path = os.path.join( output_root, OUTPUT_UNSUPPORTED_NAME )
    existing = sorted( glob.glob( os.path.join( output_data_dir, 'train-*.parquet' ) ) )
    existing += sorted( glob.glob( os.path.join( output_data_dir, 'test-*.parquet' ) ) )
    existing += [ path for path in [ metadata_path, unsupported_path ] if os.path.exists( path ) ]

    if len( existing ) > 0 and not overwrite:
        raise RuntimeError( 'Output already exists. Use --overwrite to replace: ' + output_root )

    if overwrite:
        for path in glob.glob( os.path.join( output_data_dir, 'train-*.parquet' ) ):
            os.remove( path )
        for path in glob.glob( os.path.join( output_data_dir, 'test-*.parquet' ) ):
            os.remove( path )
        for path in [ metadata_path, unsupported_path ]:
            if os.path.exists( path ):
                os.remove( path )

    return output_data_dir, metadata_path, unsupported_path


def normalize_markdown_sequence( raw_text ):
    text = str( raw_text ).strip()
    if FULL_MODSEQ_RE.match( text ):
        return text
    return '[]-' + text + '-[]'


def parse_terminal_tags( part ):
    if part in [ '', '[]' ]:
        return []
    tags = []
    i = 0
    while i < len( part ):
        if part[ i ] != '[':
            return None
        end = part.find( ']', i )
        if end < 0:
            return None
        inner = part[ i + 1 : end ]
        if inner != '':
            tags.append( inner )
        i = end + 1
    return tags


def parse_site_aware_modified_sequence( modified_sequence ):
    parts = modified_sequence.split( '-', 2 )
    if len( parts ) != 3:
        return { 'ok' : False, 'reason' : 'unexpected_format' }

    nterm_part, body, cterm_part = parts
    nterm_tags = parse_terminal_tags( nterm_part )
    cterm_tags = parse_terminal_tags( cterm_part )
    if nterm_tags is None or cterm_tags is None:
        return { 'ok' : False, 'reason' : 'invalid_terminal_format' }

    residues = []
    i = 0
    while i < len( body ):
        aa = body[ i ]
        if not ( 'A' <= aa <= 'Z' ):
            return { 'ok' : False, 'reason' : 'invalid_residue:' + aa }
        i += 1
        tags = []
        while i < len( body ) and body[ i ] == '[':
            end = body.find( ']', i )
            if end < 0:
                return { 'ok' : False, 'reason' : 'unterminated_mod' }
            tag = body[ i + 1 : end ]
            if tag != '':
                tags.append( tag )
            i = end + 1
        residues.append( { 'aa' : aa, 'tags' : tags } )

    return {
        'ok' : True,
        'nterm_tags' : nterm_tags,
        'cterm_tags' : cterm_tags,
        'residues' : residues,
        'stacked_same_site' : any( len( entry[ 'tags' ] ) > 1 for entry in residues ),
    }


def stable_digest( seed, namespace, modified_sequence ):
    text = str( seed ) + '|' + str( namespace ) + '|' + modified_sequence
    return hashlib.md5( text.encode( 'utf-8' ) ).hexdigest()


def choose_split( modified_sequence, test_fraction, seed ):
    digest = stable_digest( seed, 'split', modified_sequence )
    bucket = int( digest, 16 ) % 10000000
    threshold = int( float( test_fraction ) * 10000000 )
    if bucket < threshold:
        return 'test'
    return 'train'


def deterministic_nce_values( modified_sequence, seed ):
    digest = stable_digest( seed, 'nce', modified_sequence )
    numer = int( digest[:16], 16 )
    denom = float( 0xFFFFFFFFFFFFFFFF )
    u = numer / denom if denom > 0 else 0.5
    x = ( 2.0 * u ) - 1.0
    return [ ( base + x ) / 100.0 for base in [ 25.0, 27.0, 29.0, 31.0, 33.0, 35.0 ] ]


def onehot_charge_ints( precursor_charge ):
    return [ int( precursor_charge == z ) for z in range( 1, 7 ) ]


def project_fragment_intensities( vector_174 ):
    flat = np.asarray( vector_174, dtype='float32' )
    if flat.shape[0] != 174:
        raise ValueError( 'Expected 174 Cartographer channels, got ' + str( flat.shape[0] ) )
    matrix = flat.reshape( 29, 6 )
    projected = matrix[ :, [ 0, 1, 3, 4 ] ].reshape( 116 )
    return projected.astype( 'float32' ).tolist()


def top_two_charges( distribution ):
    pairs = [ ( idx + 1, float( distribution[ idx ] ) ) for idx in range( len( distribution ) ) ]
    pairs.sort( key=lambda item: ( -item[1], item[0] ) )
    return [ pairs[0][0], pairs[1][0] ]


def load_sculptor_norm_stats( metadata_path ):
    with open( metadata_path, 'r' ) as handle:
        metadata = json.load( handle )
    ccs_mean = float( metadata.get( 'train_ccs_mean', 0.0 ) )
    ccs_std = float( metadata.get( 'train_ccs_std', 1.0 ) )
    if ccs_std <= 0.0:
        ccs_std = 1.0
    return ccs_mean, ccs_std


def unwrap_state_dict( state ):
    if isinstance( state, dict ):
        nested = state.get( 'state_dict', None )
        if isinstance( nested, dict ):
            return nested
    return state


def infer_resnet_n_blocks_and_kernel( state ):
    block_ids = set()
    kernel = None
    for key, value in state.items():
        if key.startswith( 'resnet_blocks.' ):
            parts = key.split( '.' )
            if len( parts ) > 1 and parts[1].isdigit():
                block_ids.add( int( parts[1] ) )
        if key.endswith( 'process_blocks.1.0.0.weight' ) and hasattr( value, 'shape' ) and len( value.shape ) == 3:
            kernel = int( value.shape[-1] )
    n_blocks = ( max( block_ids ) + 1 ) if len( block_ids ) > 0 else 0
    return n_blocks, kernel


def infer_electrician_arch_from_state_dict( state ):
    seq_embed = state.get( 'seq_embed.weight', None )
    output_weight = state.get( 'output.weight', None )
    if seq_embed is None or output_weight is None:
        raise ValueError( 'Electrician checkpoint missing required weights' )
    embed_dim = int( seq_embed.shape[1] )
    out_inputs = int( output_weight.shape[1] )
    vec_length = int( out_inputs // embed_dim )
    n_blocks, kernel = infer_resnet_n_blocks_and_kernel( state )
    return {
        'embed_dim' : embed_dim,
        'n_blocks' : n_blocks,
        'kernel' : kernel,
    }


def infer_cartographer_arch_from_state_dict( state ):
    seq_embed = state.get( 'seq_embed.weight', None )
    charge_embed_weight = state.get( 'charge_embed.embed_dim1.weight', None )
    nce_embed_weight = state.get( 'nce_embed.embed_dim1.0.weight', None )
    hcd_conv = state.get( 'hcd_conv.weight', None )
    if seq_embed is None or charge_embed_weight is None or nce_embed_weight is None or hcd_conv is None:
        raise ValueError( 'Cartographer checkpoint missing required weights' )
    embed_dim = int( seq_embed.shape[1] )
    n_states = int( seq_embed.shape[0] )
    vec_length = int( charge_embed_weight.shape[0] )
    n_charges = int( charge_embed_weight.shape[1] )
    nce_dim = int( nce_embed_weight.shape[0] )
    n_ion_channels = int( hcd_conv.shape[0] )
    n_blocks, kernel = infer_resnet_n_blocks_and_kernel( state )
    return {
        'vec_length' : vec_length,
        'n_states' : n_states,
        'n_charges' : n_charges,
        'embed_dim' : embed_dim,
        'nce_dim' : nce_dim,
        'n_blocks' : n_blocks,
        'kernel' : kernel,
        'n_ion_channels' : n_ion_channels,
    }


def summarize_skip_reason( skip_counts, default_reason ):
    if skip_counts is None or len( skip_counts ) == 0:
        return default_reason
    items = sorted( skip_counts.items(), key=lambda kv: ( -kv[1], kv[0] ) )
    return items[0][0]


def translate_unimod_to_legacy_chronologer( parsed ):
    if not parsed.get( 'ok', False ):
        return None, parsed.get( 'reason', 'parse_failed' )

    if len( parsed[ 'cterm_tags' ] ) > 0:
        return None, 'unsupported_cterm_mod'

    nterm_prefix = ''
    pyro_tag = None
    if len( parsed[ 'nterm_tags' ] ) > 1:
        return None, 'stacked_nterm_mods'
    if len( parsed[ 'nterm_tags' ] ) == 1:
        tag = parsed[ 'nterm_tags' ][0]
        if tag in [ 'UNIMOD:27', 'UNIMOD:28' ]:
            pyro_tag = tag
        elif tag in CHRONOLOGER_NTERM_UNIMOD_TO_MASS:
            nterm_prefix = '[' + CHRONOLOGER_NTERM_UNIMOD_TO_MASS[ tag ] + ']'
        else:
            return None, 'unsupported_nterm:' + tag

    body_parts = []
    for idx, entry in enumerate( parsed[ 'residues' ] ):
        aa = entry[ 'aa' ]
        tags = list( entry[ 'tags' ] )
        if len( tags ) > 1:
            return None, 'stacked_same_site_residue_mod'
        if len( tags ) == 0:
            if pyro_tag is not None and idx == 0:
                if aa == 'Q' and pyro_tag == 'UNIMOD:28':
                    body_parts.append( 'Q[-17.02]' )
                    pyro_tag = None
                    continue
                if aa == 'E' and pyro_tag == 'UNIMOD:27':
                    body_parts.append( 'E[-18.01]' )
                    pyro_tag = None
                    continue
                return None, 'pyro_without_matching_first_residue'
            body_parts.append( aa )
            continue

        tag = tags[0]
        if pyro_tag is not None and idx == 0:
            return None, 'stacked_nterm_and_first_residue_mod'
        if idx == 0 and tag == 'UNIMOD:28':
            if aa != 'Q':
                return None, 'pyro_q_on_non_q'
            body_parts.append( 'Q[-17.02]' )
            continue
        if idx == 0 and tag == 'UNIMOD:27':
            if aa != 'E':
                return None, 'pyro_e_on_non_e'
            body_parts.append( 'E[-18.01]' )
            continue

        mass = CHRONOLOGER_RESIDUE_UNIMOD_TO_MASS.get( ( aa, tag ), None )
        if mass is None:
            return None, aa + '[' + tag + ']'
        body_parts.append( aa + '[' + mass + ']' )

    if pyro_tag is not None:
        return None, 'dangling_pyro_nterm'

    legacy_modseq = nterm_prefix + ''.join( body_parts )
    return legacy_modseq, None


def run_electrician( model, device, records ):
    if len( records ) == 0:
        return
    seq_batch = np.asarray( [ record[ 'common_seq_array' ] for record in records ], dtype='int64' )
    seq_tensor = torch.as_tensor( seq_batch, dtype=torch.long, device=device )
    with torch.no_grad():
        pred = model( seq_tensor ).detach().cpu().numpy()
    for record, dist in zip( records, pred ):
        record[ 'charge_distribution' ] = dist.astype( 'float32' ).tolist()
        record[ 'charges' ] = top_two_charges( dist )


def run_chronologer( model, device, records ):
    eligible = [ record for record in records if record.get( 'chrono_seq_array', None ) is not None ]
    if len( eligible ) == 0:
        return
    seq_batch = np.asarray( [ record[ 'chrono_seq_array' ] for record in eligible ], dtype='int64' )
    seq_tensor = torch.as_tensor( seq_batch, dtype=torch.long, device=device )
    with torch.no_grad():
        pred = model( seq_tensor ).squeeze( -1 ).detach().cpu().numpy()
    for record, value in zip( eligible, pred ):
        record[ 'indexed_retention_time' ] = float( value )


def run_sculptor( model, device, records, ccs_mean, ccs_std ):
    eligible_records = [ record for record in records if record.get( 'sculptor_seq_array', None ) is not None ]
    if len( eligible_records ) == 0:
        return

    seq_rows = []
    charge_rows = []
    pairs = []
    for record in eligible_records:
        for charge in record[ 'charges' ]:
            seq_rows.append( record[ 'sculptor_seq_array' ] )
            charge_rows.append( sculptor_return_charge_onehot( charge ) )
            pairs.append( ( record, charge ) )

    seq_tensor = torch.as_tensor( np.asarray( seq_rows, dtype='int64' ), dtype=torch.long, device=device )
    charge_tensor = torch.as_tensor( np.asarray( charge_rows, dtype='float32' ), dtype=torch.float32, device=device )
    with torch.no_grad():
        pred_norm = model( seq_tensor, charge_tensor ).squeeze( -1 ).detach().cpu().numpy()
    for ( record, charge ), value in zip( pairs, pred_norm ):
        record[ 'ccs_by_charge' ][ charge ] = float( value * ccs_std + ccs_mean )


def run_cartographer( model, device, records, batch_size_fragmentation ):
    expanded = []
    for record in records:
        for charge in record[ 'charges' ]:
            charge_ohe = return_charge_array( charge, 1 )[0]
            for nce_idx, nce_norm in enumerate( record[ 'nce_values' ] ):
                expanded.append( ( record, charge, nce_idx, charge_ohe, nce_norm ) )

    if len( expanded ) == 0:
        return

    for start in range( 0, len( expanded ), int( batch_size_fragmentation ) ):
        batch = expanded[ start : start + int( batch_size_fragmentation ) ]
        seq_rows = np.asarray( [ entry[0][ 'common_seq_array' ] for entry in batch ], dtype='int64' )
        charge_rows = np.asarray( [ entry[3] for entry in batch ], dtype='float32' )
        nce_rows = np.asarray( [ [ entry[4] ] for entry in batch ], dtype='float32' )
        seq_tensor = torch.as_tensor( seq_rows, dtype=torch.long, device=device )
        charge_tensor = torch.as_tensor( charge_rows, dtype=torch.float32, device=device )
        nce_tensor = torch.as_tensor( nce_rows, dtype=torch.float32, device=device )
        with torch.no_grad():
            pred = model( seq_tensor, charge_tensor, nce_tensor ).detach().cpu().numpy()
        for entry, vector in zip( batch, pred ):
            record = entry[0]
            charge = entry[1]
            nce_idx = entry[2]
            record[ 'fragmentation' ][ ( charge, nce_idx ) ] = project_fragment_intensities( vector )


def process_chunk( args,
                   chunk_records,
                   models,
                   device,
                   ccs_mean,
                   ccs_std,
                   writers,
                   unsupported_writer,
                   stats ):
    kept_records = []

    for record in chunk_records:
        stats[ 'total_peptides_seen' ] += 1
        split_name = record[ 'split' ]
        modified_sequence = record[ 'modified_sequence' ]
        parsed = record[ 'parsed' ]

        if not parsed.get( 'ok', False ):
            stats[ 'peptides_skipped_total' ] += 1
            stats[ 'peptides_skipped_parse' ] += 1
            reason = parsed.get( 'reason', 'parse_failed' )
            stats[ 'skip_reasons' ][ reason ] += 1
            unsupported_writer.write( modified_sequence, split_name, 'parse', reason, 0 )
            continue

        if parsed.get( 'stacked_same_site', False ):
            stats[ 'peptides_skipped_total' ] += 1
            stats[ 'peptides_skipped_stacked_same_site' ] += 1
            stats[ 'skip_reasons' ][ 'stacked_same_site_residue_mod' ] += 1
            unsupported_writer.write( modified_sequence, split_name, 'parse', 'stacked_same_site_residue_mod', 0 )
            continue

        peptide_len = len( parsed[ 'residues' ] )
        if peptide_len > common_max_len:
            reason = 'peptide_length_gt_' + str( common_max_len )
            stats[ 'peptides_skipped_total' ] += 1
            stats[ 'peptides_skipped_too_long' ] += 1
            stats[ 'skip_reasons' ][ reason ] += 1
            unsupported_writer.write( modified_sequence, split_name, 'length', reason, 0 )
            continue

        electric_skip = {}
        common_coded = common_unimod_to_codedseq( modified_sequence, max_len=common_max_len, skip_counts=electric_skip )
        if common_coded is None:
            reason = summarize_skip_reason( electric_skip, 'electrician_tokenization_failed' )
            stats[ 'peptides_skipped_total' ] += 1
            stats[ 'peptides_skipped_electrician' ] += 1
            stats[ 'skip_reasons' ][ reason ] += 1
            unsupported_writer.write( modified_sequence, split_name, 'electrician', reason, 0 )
            continue

        record[ 'common_seq_array' ] = common_codedseq_to_array( common_coded, max_size=common_max_len + 2 )
        record[ 'nce_values' ] = deterministic_nce_values( modified_sequence, args.seed )
        record[ 'fragmentation' ] = {}
        record[ 'ccs_by_charge' ] = {}
        record[ 'indexed_retention_time' ] = None
        record[ 'charge_distribution' ] = None

        legacy_modseq, chrono_reason = translate_unimod_to_legacy_chronologer( parsed )
        if legacy_modseq is not None:
            chrono_coded = legacy_chronologer_modseq_to_codedseq( legacy_modseq )
            if chrono_coded is not None:
                record[ 'chrono_seq_array' ] = legacy_chronologer_codedseq_to_array( chrono_coded, 52 )
            else:
                record[ 'chrono_seq_array' ] = None
                record[ 'chrono_reason' ] = 'legacy_tokenization_failed'
        else:
            record[ 'chrono_seq_array' ] = None
            record[ 'chrono_reason' ] = chrono_reason

        sculptor_skip = {}
        sculptor_coded = sculptor_unimod_to_codedseq( modified_sequence, max_len=sculptor_max_len, skip_counts=sculptor_skip )
        if sculptor_coded is not None:
            record[ 'sculptor_seq_array' ] = sculptor_codedseq_to_array( sculptor_coded, max_size=sculptor_max_len + 2 )
        else:
            record[ 'sculptor_seq_array' ] = None
            record[ 'sculptor_reason' ] = summarize_skip_reason( sculptor_skip, 'sculptor_tokenization_failed' )

        kept_records.append( record )

    if len( kept_records ) == 0:
        return

    run_electrician( models[ 'electrician' ], device, kept_records )
    run_chronologer( models[ 'chronologer' ], device, kept_records )
    run_sculptor( models[ 'sculptor' ], device, kept_records, ccs_mean, ccs_std )
    run_cartographer( models[ 'cartographer' ], device, kept_records, args.batch_size_fragmentation )

    for record in kept_records:
        stats[ 'peptides_written_by_split' ][ record[ 'split' ] ] += 1
        if record.get( 'chrono_seq_array', None ) is None:
            unsupported_writer.write( record[ 'modified_sequence' ], record[ 'split' ], 'chronologer', record.get( 'chrono_reason', 'unsupported' ), 12 )
        if record.get( 'sculptor_seq_array', None ) is None:
            unsupported_writer.write( record[ 'modified_sequence' ], record[ 'split' ], 'sculptor', record.get( 'sculptor_reason', 'unsupported' ), 12 )

        writers[ record[ 'split' ] ]
        for charge in record[ 'charges' ]:
            charge_ohe = onehot_charge_ints( charge )
            ccs_value = record[ 'ccs_by_charge' ].get( charge, None )
            for nce_idx, nce_norm in enumerate( record[ 'nce_values' ] ):
                intensities = record[ 'fragmentation' ].get( ( charge, nce_idx ), None )
                row = {
                    'modified_sequence' : record[ 'modified_sequence' ],
                    'precursor_charge_onehot' : charge_ohe,
                    'charge_state_dist' : record[ 'charge_distribution' ],
                    'collision_energy_aligned_normed' : float( nce_norm ),
                    'indexed_retention_time' : record[ 'indexed_retention_time' ],
                    'ccs' : ccs_value,
                    'intensities_raw' : intensities,
                }
                writers[ record[ 'split' ] ].append( row )
                stats[ 'rows_written_by_split' ][ record[ 'split' ] ] += 1
                if row[ 'indexed_retention_time' ] is None:
                    stats[ 'null_counts' ][ 'indexed_retention_time' ] += 1
                if row[ 'ccs' ] is None:
                    stats[ 'null_counts' ][ 'ccs' ] += 1
                if row[ 'intensities_raw' ] is None:
                    stats[ 'null_counts' ][ 'intensities_raw' ] += 1


def iter_markdown_sequences( input_markdown ):
    with open( input_markdown, 'r' ) as handle:
        for line in handle:
            match = MARKDOWN_BULLET_RE.match( line )
            if match is None:
                continue
            yield normalize_markdown_sequence( match.group( 1 ) )


def write_metadata( metadata_path, args, output_data_dir, writers, stats, ccs_mean, ccs_std ):
    metadata = {
        'input_markdown' : os.path.abspath( args.input_markdown ),
        'output_root' : os.path.abspath( args.output_root ),
        'output_data_dir' : os.path.abspath( output_data_dir ),
        'seed' : int( args.seed ),
        'test_fraction' : float( args.test_fraction ),
        'rows_per_shard' : int( args.rows_per_shard ),
        'batch_size_peptides' : int( args.batch_size_peptides ),
        'batch_size_fragmentation' : int( args.batch_size_fragmentation ),
        'max_peptides' : args.max_peptides,
        'device' : str( args.device ),
        'cartographer_model' : os.path.abspath( args.cartographer_model ),
        'chronologer_model' : os.path.abspath( args.chronologer_model ),
        'electrician_model' : os.path.abspath( args.electrician_model ),
        'sculptor_model' : os.path.abspath( args.sculptor_model ),
        'sculptor_metadata' : os.path.abspath( args.sculptor_metadata ),
        'sculptor_ccs_mean' : float( ccs_mean ),
        'sculptor_ccs_std' : float( ccs_std ),
        'total_peptides_seen' : int( stats[ 'total_peptides_seen' ] ),
        'peptides_skipped_total' : int( stats[ 'peptides_skipped_total' ] ),
        'peptides_written_total' : int( stats[ 'peptides_written_by_split' ][ 'train' ] + stats[ 'peptides_written_by_split' ][ 'test' ] ),
        'peptides_skipped_parse' : int( stats[ 'peptides_skipped_parse' ] ),
        'peptides_skipped_stacked_same_site' : int( stats[ 'peptides_skipped_stacked_same_site' ] ),
        'peptides_skipped_too_long' : int( stats[ 'peptides_skipped_too_long' ] ),
        'peptides_skipped_electrician' : int( stats[ 'peptides_skipped_electrician' ] ),
        'peptides_written_by_split' : {
            'train' : int( stats[ 'peptides_written_by_split' ][ 'train' ] ),
            'test' : int( stats[ 'peptides_written_by_split' ][ 'test' ] ),
        },
        'rows_written_by_split' : {
            'train' : int( stats[ 'rows_written_by_split' ][ 'train' ] ),
            'test' : int( stats[ 'rows_written_by_split' ][ 'test' ] ),
        },
        'null_counts' : { key : int( value ) for key, value in sorted( stats[ 'null_counts' ].items() ) },
        'skip_reasons' : { key : int( value ) for key, value in sorted( stats[ 'skip_reasons' ].items(), key=lambda kv: (-kv[1], kv[0]) ) },
        'train_shards_written' : int( writers[ 'train' ].shard_index ),
        'test_shards_written' : int( writers[ 'test' ].shard_index ),
    }
    with open( metadata_path, 'w' ) as handle:
        json.dump( metadata, handle, indent=2 )


def load_models( args, device ):
    chronologer_state = load_chronologer_state_dict( args.chronologer_model, map_location='cpu' )
    chronologer_model, _ = initialize_chronologer_from_checkpoint( chronologer_state )
    chronologer_model = chronologer_model.to( device )
    chronologer_model.eval()

    electrician_state = unwrap_state_dict( torch.load( args.electrician_model, map_location='cpu' ) )
    electrician_arch = infer_electrician_arch_from_state_dict( electrician_state )
    electrician_model = initialize_electrician_model( model_file=None,
                                                      arch_overrides=electrician_arch,
                                                      map_location='cpu' )
    electrician_model.load_state_dict( electrician_state, strict=True )
    electrician_model = electrician_model.to( device )
    electrician_model.eval()

    sculptor_model = initialize_sculptor_model( model_file=args.sculptor_model, map_location='cpu' ).to( device )
    sculptor_model.eval()

    cartographer_state = unwrap_state_dict( torch.load( args.cartographer_model, map_location='cpu' ) )
    cartographer_arch = infer_cartographer_arch_from_state_dict( cartographer_state )
    cartographer_model = cartographer_hcd( cartographer_arch[ 'vec_length' ],
                                           cartographer_arch[ 'n_states' ],
                                           cartographer_arch[ 'n_charges' ],
                                           cartographer_arch[ 'embed_dim' ],
                                           cartographer_arch[ 'nce_dim' ],
                                           cartographer_arch[ 'n_blocks' ],
                                           cartographer_arch[ 'kernel' ],
                                           cartographer_settings.training_parameters[ 'dropout_rate' ],
                                           cartographer_settings.hyperparameters[ 'activation_function' ],
                                           cartographer_arch[ 'n_ion_channels' ] )
    cartographer_model.load_state_dict( cartographer_state, strict=True )
    cartographer_model = cartographer_model.to( device )
    cartographer_model.eval()

    return {
        'chronologer' : chronologer_model,
        'electrician' : electrician_model,
        'sculptor' : sculptor_model,
        'cartographer' : cartographer_model,
    }


def main():
    args = parse_args( os.sys.argv[1:] )

    if args.test_fraction <= 0.0 or args.test_fraction >= 1.0:
        raise ValueError( '--test_fraction must be in (0, 1)' )
    if args.rows_per_shard <= 0:
        raise ValueError( '--rows_per_shard must be > 0' )
    if args.batch_size_peptides <= 0:
        raise ValueError( '--batch_size_peptides must be > 0' )
    if args.batch_size_fragmentation <= 0:
        raise ValueError( '--batch_size_fragmentation must be > 0' )
    if not os.path.isfile( args.input_markdown ):
        raise FileNotFoundError( 'Input markdown not found: ' + args.input_markdown )
    if not os.path.isfile( args.sculptor_metadata ):
        raise FileNotFoundError( 'Sculptor metadata not found: ' + args.sculptor_metadata )

    output_data_dir, metadata_path, unsupported_path = ensure_output_dirs( args.output_root, args.overwrite )
    device = resolve_device( args.device )
    ccs_mean, ccs_std = load_sculptor_norm_stats( args.sculptor_metadata )
    models = load_models( args, device )

    writers = {
        'train' : SplitParquetWriter( 'train', output_data_dir, args.rows_per_shard ),
        'test' : SplitParquetWriter( 'test', output_data_dir, args.rows_per_shard ),
    }
    unsupported_writer = UnsupportedWriter( unsupported_path )
    stats = {
        'total_peptides_seen' : 0,
        'peptides_skipped_total' : 0,
        'peptides_skipped_parse' : 0,
        'peptides_skipped_stacked_same_site' : 0,
        'peptides_skipped_too_long' : 0,
        'peptides_skipped_electrician' : 0,
        'peptides_written_by_split' : Counter(),
        'rows_written_by_split' : Counter(),
        'null_counts' : Counter(),
        'skip_reasons' : Counter(),
    }

    chunk = []
    tick_count = 0
    try:
        for modified_sequence in iter_markdown_sequences( args.input_markdown ):
            if args.max_peptides is not None and stats[ 'total_peptides_seen' ] + len( chunk ) >= int( args.max_peptides ):
                break
            chunk.append( {
                'modified_sequence' : modified_sequence,
                'split' : choose_split( modified_sequence, args.test_fraction, args.seed ),
                'parsed' : parse_site_aware_modified_sequence( modified_sequence ),
            } )
            if len( chunk ) >= int( args.batch_size_peptides ):
                process_chunk( args, chunk, models, device, ccs_mean, ccs_std, writers, unsupported_writer, stats )
                chunk = []
                tick_count += 1
                sys.stdout.write( '.' )
                if tick_count % 10 == 0:
                    sys.stdout.write( ' ' )
                if tick_count % 100 == 0:
                    sys.stdout.write( '\n' )
                sys.stdout.flush()

        if len( chunk ) > 0:
            process_chunk( args, chunk, models, device, ccs_mean, ccs_std, writers, unsupported_writer, stats )
            tick_count += 1
            sys.stdout.write( '.' )
            if tick_count % 10 == 0:
                sys.stdout.write( ' ' )
            if tick_count % 100 == 0:
                sys.stdout.write( '\n' )
            sys.stdout.flush()
    finally:
        writers[ 'train' ].close()
        writers[ 'test' ].close()
        unsupported_writer.close()

    if tick_count % 100 != 0 and tick_count > 0:
        print()

    write_metadata( metadata_path, args, output_data_dir, writers, stats, ccs_mean, ccs_std )

    print( 'Distilled dataset generation complete' )
    print( '  peptides_seen=' + str( stats[ 'total_peptides_seen' ] ) )
    print( '  peptides_skipped_total=' + str( stats[ 'peptides_skipped_total' ] ) )
    print( '  peptides_written_total=' + str( stats[ 'peptides_written_by_split' ][ 'train' ] + stats[ 'peptides_written_by_split' ][ 'test' ] ) )
    print( '  peptides_skipped_stacked_same_site=' + str( stats[ 'peptides_skipped_stacked_same_site' ] ) )
    print( '  peptides_skipped_too_long=' + str( stats[ 'peptides_skipped_too_long' ] ) )
    print( '  peptides_skipped_electrician=' + str( stats[ 'peptides_skipped_electrician' ] ) )
    print( '  train_rows=' + str( stats[ 'rows_written_by_split' ][ 'train' ] ) )
    print( '  test_rows=' + str( stats[ 'rows_written_by_split' ][ 'test' ] ) )
    print( '  metadata=' + metadata_path )
    print( '  unsupported=' + unsupported_path )


if __name__ == '__main__':
    main()
