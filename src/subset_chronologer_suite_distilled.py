"""Build a small unmodified z=2 subset from the full distilled dataset."""

import argparse
import glob
import json
import os
import random
import re

import pyarrow as pa
import pyarrow.parquet as pq


SRC = '/Users/searle.brian/Documents/huggingface/data/full_chronologer_suite_distilled'
DST = '/Users/searle.brian/Documents/huggingface/data/subset_chronologer_suite_distilled'
UNMOD_PATTERN = re.compile( r'^\[\]-[A-Z]+-\[\]$' )
TARGET_CHARGE = [ 0, 1, 0, 0, 0, 0 ]


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Sample a small unmodified z=2 subset from the distilled dataset.' )
    parser.add_argument( '--src_root', type=str, default=SRC,
                         help='Source distilled dataset root containing data/train-*.parquet and data/test-*.parquet.' )
    parser.add_argument( '--dst_root', type=str, default=DST,
                         help='Destination root for the sampled subset dataset.' )
    parser.add_argument( '--train_peptides', type=int, default=50000,
                         help='Number of unique qualifying peptides to sample from the train split.' )
    parser.add_argument( '--test_peptides', type=int, default=5000,
                         help='Number of unique qualifying peptides to sample from the test split.' )
    parser.add_argument( '--seed', type=int, default=1337,
                         help='Random seed for reproducible peptide sampling.' )
    parser.add_argument( '--rows_per_shard', type=int, default=200000,
                         help='Rows per output parquet shard.' )
    parser.add_argument( '--overwrite', action='store_true',
                         help='Overwrite any existing subset shards and metadata.' )
    return parser.parse_args( args )


def discover_split_files( dataset_root, split ):
    pattern = os.path.join( dataset_root, 'data', split + '-*.parquet' )
    return sorted( glob.glob( pattern ) )


def keep_row( seq, charge ):
    return UNMOD_PATTERN.match( seq ) is not None and list(charge) == TARGET_CHARGE


class SplitParquetWriter( object ):
    def __init__( self, split_name, output_data_dir, rows_per_shard ):
        self.split_name = split_name
        self.output_data_dir = output_data_dir
        self.rows_per_shard = int( rows_per_shard )
        self.shard_index = 0
        self.rows_written = 0
        self.buffers = None

    def append_table( self, table ):
        if self.buffers is None:
            self.buffers = []
        self.buffers.append( table )
        buffered_rows = sum( len(t) for t in self.buffers )
        if buffered_rows >= self.rows_per_shard:
            self.flush()

    def flush( self ):
        if not self.buffers:
            return
        combined = pa.concat_tables( self.buffers )
        offset = 0
        while offset < len( combined ):
            piece = combined.slice( offset, self.rows_per_shard )
            out_name = self.split_name + '-' + format( self.shard_index, '05d' ) + '.parquet'
            out_path = os.path.join( self.output_data_dir, out_name )
            pq.write_table( piece, out_path, compression='zstd' )
            self.rows_written += len( piece )
            self.shard_index += 1
            offset += len( piece )
        self.buffers = []

    def close( self ):
        self.flush()


def ensure_output_root( dst_root, overwrite ):
    data_dir = os.path.join( dst_root, 'data' )
    os.makedirs( data_dir, exist_ok=True )
    metadata_path = os.path.join( dst_root, 'subset_metadata.json' )

    existing = glob.glob( os.path.join( data_dir, 'train-*.parquet' ) )
    existing += glob.glob( os.path.join( data_dir, 'test-*.parquet' ) )
    if os.path.isfile( metadata_path ):
        existing.append( metadata_path )

    if len( existing ) > 0 and not overwrite:
        raise RuntimeError( 'Destination already contains subset output. Use --overwrite: ' + dst_root )

    if overwrite:
        for path in glob.glob( os.path.join( data_dir, 'train-*.parquet' ) ):
            os.remove( path )
        for path in glob.glob( os.path.join( data_dir, 'test-*.parquet' ) ):
            os.remove( path )
        if os.path.isfile( metadata_path ):
            os.remove( metadata_path )

    return data_dir, metadata_path


def collect_unique_peptides( parquet_files ):
    selected = set()
    for filepath in parquet_files:
        pf = pq.ParquetFile( filepath )
        for rg_idx in range( pf.metadata.num_row_groups ):
            table = pf.read_row_group( rg_idx, columns=[ 'modified_sequence', 'precursor_charge_onehot' ] )
            mod_seqs = table.column( 'modified_sequence' ).to_pylist()
            charges = table.column( 'precursor_charge_onehot' ).to_pylist()
            for seq, charge in zip( mod_seqs, charges ):
                if keep_row( seq, charge ):
                    selected.add( seq )
    return sorted( selected )


def sample_peptides( peptides, count, rng, split ):
    if len( peptides ) < count:
        raise RuntimeError( split + ' split only has ' + str(len(peptides)) +
                            ' qualifying peptides, requested ' + str(count) )
    return set( rng.sample( peptides, count ) )


def filter_split( parquet_files, chosen_peptides, split_name, writer ):
    rows_in = 0
    rows_out = 0
    peptides_written = set()

    for filepath in parquet_files:
        pf = pq.ParquetFile( filepath )
        kept_batches = []
        for rg_idx in range( pf.metadata.num_row_groups ):
            table = pf.read_row_group( rg_idx )
            rows_in += len( table )
            mod_seqs = table.column( 'modified_sequence' ).to_pylist()
            charges = table.column( 'precursor_charge_onehot' ).to_pylist()
            mask = []
            for seq, charge in zip( mod_seqs, charges ):
                keep = keep_row( seq, charge ) and seq in chosen_peptides
                mask.append( keep )
                if keep:
                    peptides_written.add( seq )
            filtered = table.filter( mask )
            if len( filtered ) > 0:
                rows_out += len( filtered )
                kept_batches.append( filtered )
        for batch in kept_batches:
            writer.append_table( batch )

    return {
        'rows_in' : rows_in,
        'rows_out' : rows_out,
        'unique_peptides_written' : len( peptides_written ),
    }


def write_metadata( metadata_path, args, train_stats, test_stats, train_candidates, test_candidates ):
    metadata = {
        'src_root' : os.path.abspath( args.src_root ),
        'dst_root' : os.path.abspath( args.dst_root ),
        'train_peptides_requested' : int( args.train_peptides ),
        'test_peptides_requested' : int( args.test_peptides ),
        'seed' : int( args.seed ),
        'rows_per_shard' : int( args.rows_per_shard ),
        'criteria' : {
            'modified_sequence_regex' : UNMOD_PATTERN.pattern,
            'precursor_charge_onehot' : list( TARGET_CHARGE ),
        },
        'candidate_unique_peptides' : {
            'train' : int( train_candidates ),
            'test' : int( test_candidates ),
        },
        'written' : {
            'train' : dict( train_stats ),
            'test' : dict( test_stats ),
        },
    }
    with open( metadata_path, 'w' ) as handle:
        json.dump( metadata, handle, indent=2 )


def main():
    args = parse_args( os.sys.argv[1:] )
    data_dir, metadata_path = ensure_output_root( args.dst_root, args.overwrite )

    train_files = discover_split_files( args.src_root, 'train' )
    test_files = discover_split_files( args.src_root, 'test' )
    if len( train_files ) == 0 or len( test_files ) == 0:
        raise RuntimeError( 'Could not find train/test parquet shards under ' + args.src_root )

    print( 'Scanning qualifying unique peptides...' )
    train_candidates = collect_unique_peptides( train_files )
    test_candidates = collect_unique_peptides( test_files )
    print( '  train candidates=' + str(len(train_candidates)) )
    print( '  test candidates=' + str(len(test_candidates)) )

    rng = random.Random( args.seed )
    chosen_train = sample_peptides( train_candidates, args.train_peptides, rng, 'train' )
    chosen_test = sample_peptides( test_candidates, args.test_peptides, rng, 'test' )

    train_writer = SplitParquetWriter( 'train', data_dir, args.rows_per_shard )
    test_writer = SplitParquetWriter( 'test', data_dir, args.rows_per_shard )
    try:
        print( 'Writing sampled train split...' )
        train_stats = filter_split( train_files, chosen_train, 'train', train_writer )
        print( 'Writing sampled test split...' )
        test_stats = filter_split( test_files, chosen_test, 'test', test_writer )
    finally:
        train_writer.close()
        test_writer.close()

    write_metadata( metadata_path, args, train_stats, test_stats, len(train_candidates), len(test_candidates) )

    print( 'Subset dataset complete' )
    print( '  train unique peptides=' + str(train_stats[ 'unique_peptides_written' ]) +
           ' rows=' + str(train_stats[ 'rows_out' ]) )
    print( '  test unique peptides=' + str(test_stats[ 'unique_peptides_written' ]) +
           ' rows=' + str(test_stats[ 'rows_out' ]) )
    print( '  output=' + args.dst_root )


if __name__ == '__main__':
    main()
