import argparse
import csv
import datetime
import glob
import hashlib
import json
import os

from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from sculptor_settings import charge_dist_len, max_peptide_len, metadata_filename
from sculptor_tensorize import ( aa_to_int,
                                 codedseq_to_array,
                                 nterm_unimod_map,
                                 residue_unimod_map,
                                 residues,
                                 return_charge_onehot,
                                 unimod_to_codedseq )


DEFAULT_INPUT_CSV = '/Users/searle.brian/Documents/testing/trainingdata/union_ccs.csv'
DEFAULT_OUTPUT_ROOT = '/Users/searle.brian/Documents/huggingface/data/IM2Deep_CCS'


class RunningStats( object ):
    def __init__( self ):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update( self, value ):
        x = float( value )
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def variance( self ):
        if self.count < 2:
            return 0.0
        return self.m2 / ( self.count - 1 )

    def std( self ):
        var = self.variance()
        if var <= 0.0:
            return 1.0
        return var ** 0.5


class SplitParquetWriter( object ):
    def __init__( self, split_name, output_data_dir, rows_per_shard ):
        self.split_name = split_name
        self.output_data_dir = output_data_dir
        self.rows_per_shard = int( rows_per_shard )

        self.seq_tokens = []
        self.charge_onehot = []
        self.ccs = []
        self.weight = []

        self.rows_written = 0
        self.shard_index = 0

    def append( self, seq_tokens, charge_onehot, ccs_value, weight_value ):
        self.seq_tokens.append( seq_tokens )
        self.charge_onehot.append( charge_onehot )
        self.ccs.append( float( ccs_value ) )
        self.weight.append( float( weight_value ) )

        if len( self.seq_tokens ) >= self.rows_per_shard:
            self.flush()

    def flush( self ):
        if len( self.seq_tokens ) == 0:
            return

        table = pa.Table.from_pydict( { 'seq_tokens' : self.seq_tokens,
                                        'charge_onehot' : self.charge_onehot,
                                        'ccs' : self.ccs,
                                        'weight' : self.weight, } )

        out_name = self.split_name + '-' + str(self.shard_index).zfill(3) + '.parquet'
        out_path = os.path.join( self.output_data_dir, out_name )
        pq.write_table( table, out_path, compression='zstd' )

        self.rows_written += len( self.seq_tokens )
        self.shard_index += 1

        self.seq_tokens = []
        self.charge_onehot = []
        self.ccs = []
        self.weight = []

    def close( self ):
        self.flush()


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Prepare Sculptor CCS parquet dataset from union_ccs.csv' )
    parser.add_argument( '--input_csv',
                         type=str,
                         default=DEFAULT_INPUT_CSV,
                         help='Path to union_ccs.csv' )
    parser.add_argument( '--output_root',
                         type=str,
                         default=DEFAULT_OUTPUT_ROOT,
                         help='Output root directory containing data/ parquet shards' )
    parser.add_argument( '--test_fraction',
                         type=float,
                         default=0.2,
                         help='Fraction of modified_sequence groups assigned to test' )
    parser.add_argument( '--rows_per_shard',
                         type=int,
                         default=200000,
                         help='Rows per output parquet shard' )
    parser.add_argument( '--max_rows',
                         type=int,
                         default=None,
                         help='Optional cap on input CSV rows (for smoke tests)' )
    parser.add_argument( '--overwrite',
                         action='store_true',
                         help='Overwrite existing train/test shard files in output_root/data' )
    return parser.parse_args( args )


def choose_split( modified_sequence, test_fraction ):
    digest = hashlib.md5( modified_sequence.encode( 'utf-8' ) ).hexdigest()
    bucket = int( digest, 16 ) % 10000000
    threshold = int( test_fraction * 10000000 )
    if bucket < threshold:
        return 'test'
    return 'train'


def clear_existing_outputs( output_data_dir, metadata_path ):
    patterns = [ 'train-*.parquet', 'test-*.parquet' ]
    for pattern in patterns:
        for path in glob.glob( os.path.join( output_data_dir, pattern ) ):
            os.remove( path )

    if os.path.isfile( metadata_path ):
        os.remove( metadata_path )


def ensure_output_dirs( output_root, overwrite ):
    output_data_dir = os.path.join( output_root, 'data' )
    os.makedirs( output_data_dir, exist_ok=True )

    metadata_path = os.path.join( output_root, metadata_filename )
    existing = glob.glob( os.path.join( output_data_dir, 'train-*.parquet' ) )
    existing += glob.glob( os.path.join( output_data_dir, 'test-*.parquet' ) )

    if len( existing ) > 0 and not overwrite:
        raise RuntimeError( 'Output directory already contains Sculptor parquet shards. '
                            'Use --overwrite to replace them: ' + output_data_dir )

    if overwrite:
        clear_existing_outputs( output_data_dir, metadata_path )

    return output_data_dir, metadata_path


def write_metadata( metadata_path,
                    input_csv,
                    output_root,
                    output_data_dir,
                    test_fraction,
                    rows_per_shard,
                    max_rows,
                    total_rows,
                    accepted_rows,
                    split_rows,
                    split_charge_counts,
                    skip_counts,
                    duplicate_key_rows,
                    train_stats,
                    train_writer,
                    test_writer ):
    residue_map_json = [ { 'residue' : aa, 'unimod' : unimod, 'token' : token }
                         for (aa, unimod), token in sorted( residue_unimod_map.items() ) ]

    metadata = { 'generated_at_utc' : datetime.datetime.now( datetime.timezone.utc ).strftime( '%Y-%m-%dT%H:%M:%SZ' ),
                 'input_csv' : os.path.abspath( input_csv ),
                 'output_root' : os.path.abspath( output_root ),
                 'output_data_dir' : os.path.abspath( output_data_dir ),
                 'max_peptide_len' : max_peptide_len,
                 'charge_dist_len' : charge_dist_len,
                 'test_fraction' : float( test_fraction ),
                 'rows_per_shard' : int( rows_per_shard ),
                 'max_rows' : max_rows,
                 'total_rows' : int( total_rows ),
                 'accepted_rows' : int( accepted_rows ),
                 'skipped_rows' : int( total_rows - accepted_rows ),
                 'split_rows' : { 'train' : int( split_rows.get( 'train', 0 ) ),
                                  'test' : int( split_rows.get( 'test', 0 ) ), },
                 'split_charge_counts' : {
                     'train' : { str(k) : int(v) for k, v in sorted( split_charge_counts['train'].items() ) },
                     'test' : { str(k) : int(v) for k, v in sorted( split_charge_counts['test'].items() ) },
                 },
                 'train_ccs_count' : int( train_stats.count ),
                 'train_ccs_mean' : float( train_stats.mean ),
                 'train_ccs_std' : float( train_stats.std() ),
                 'skip_counts' : { k : int(v) for k, v in sorted( skip_counts.items(), key=lambda kv: (-kv[1], kv[0]) ) },
                 'duplicate_modified_sequence_charge_rows' : int( duplicate_key_rows ),
                 'train_shards_written' : int( train_writer.shard_index ),
                 'test_shards_written' : int( test_writer.shard_index ),
                 'tokenizer' : {
                     'residues' : list( residues ),
                     'aa_to_int' : { k : int(v) for k, v in sorted( aa_to_int.items() ) },
                     'nterm_unimod_map' : dict( sorted( nterm_unimod_map.items() ) ),
                     'residue_unimod_map' : residue_map_json,
                 }, }

    with open( metadata_path, 'w' ) as f:
        json.dump( metadata, f, indent=2 )


def main():
    args = parse_args( os.sys.argv[1:] )

    if args.test_fraction <= 0.0 or args.test_fraction >= 1.0:
        raise ValueError( '--test_fraction must be in (0, 1)' )

    if not os.path.isfile( args.input_csv ):
        raise FileNotFoundError( 'Input CSV not found: ' + args.input_csv )

    output_data_dir, metadata_path = ensure_output_dirs( args.output_root, args.overwrite )

    train_writer = SplitParquetWriter( 'train', output_data_dir, args.rows_per_shard )
    test_writer = SplitParquetWriter( 'test', output_data_dir, args.rows_per_shard )
    writers = { 'train' : train_writer, 'test' : test_writer }

    total_rows = 0
    accepted_rows = 0
    split_rows = Counter()
    split_charge_counts = { 'train' : Counter(), 'test' : Counter() }
    skip_counts = Counter()

    seen_keys = set()
    duplicate_key_rows = 0

    train_stats = RunningStats()

    with open( args.input_csv, 'r', newline='' ) as handle:
        reader = csv.DictReader( handle )
        required_cols = { 'modified_sequence', 'charge', 'ccs' }
        if not required_cols.issubset( set( reader.fieldnames or [] ) ):
            raise ValueError( 'Input CSV missing required columns: ' + str( sorted( required_cols ) ) )

        for row in reader:
            if args.max_rows is not None and total_rows >= args.max_rows:
                break

            total_rows += 1
            modified_sequence = row[ 'modified_sequence' ].strip()

            try:
                charge = int( row[ 'charge' ] )
            except Exception:
                skip_counts[ 'bad_charge' ] += 1
                continue

            if charge < 1 or charge > charge_dist_len:
                skip_counts[ 'bad_charge' ] += 1
                continue

            try:
                ccs_value = float( row[ 'ccs' ] )
            except Exception:
                skip_counts[ 'bad_ccs' ] += 1
                continue

            coded = unimod_to_codedseq( modified_sequence,
                                        max_len=max_peptide_len,
                                        skip_counts=skip_counts )
            if coded is None:
                continue

            seq_tokens = codedseq_to_array( coded, max_size=max_peptide_len + 2 ).tolist()
            charge_onehot = return_charge_onehot( charge )

            split = choose_split( modified_sequence, args.test_fraction )
            writers[ split ].append( seq_tokens, charge_onehot, ccs_value, 1.0 )

            accepted_rows += 1
            split_rows[ split ] += 1
            split_charge_counts[ split ][ charge ] += 1

            if split == 'train':
                train_stats.update( ccs_value )

            key = modified_sequence + '|' + str(charge)
            if key in seen_keys:
                duplicate_key_rows += 1
            else:
                seen_keys.add( key )

            if accepted_rows % 200000 == 0:
                print( 'Accepted rows: ' + str(accepted_rows) )

    train_writer.close()
    test_writer.close()

    if train_stats.count == 0:
        raise RuntimeError( 'No accepted training rows found; dataset preparation failed' )

    write_metadata( metadata_path,
                    args.input_csv,
                    args.output_root,
                    output_data_dir,
                    args.test_fraction,
                    args.rows_per_shard,
                    args.max_rows,
                    total_rows,
                    accepted_rows,
                    split_rows,
                    split_charge_counts,
                    skip_counts,
                    duplicate_key_rows,
                    train_stats,
                    train_writer,
                    test_writer )

    print( 'Sculptor dataset preparation complete' )
    print( 'Input rows scanned: ' + str(total_rows) )
    print( 'Accepted rows: ' + str(accepted_rows) )
    print( 'Rejected rows: ' + str(total_rows - accepted_rows) )
    print( 'Train rows: ' + str(split_rows.get('train', 0)) +
           ' | Test rows: ' + str(split_rows.get('test', 0)) )
    print( 'Train mean CCS: ' + format( train_stats.mean, '.6f' ) +
           ' | Train std CCS: ' + format( train_stats.std(), '.6f' ) )
    print( 'Metadata: ' + metadata_path )


if __name__ == '__main__':
    main()
