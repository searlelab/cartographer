"""Carve a peptide-level validation split from the train split of a distilled dataset."""

import argparse
import glob
import json
import os
import random
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = '/Users/searle.brian/Documents/huggingface/data/full_chronologer_suite_distilled'
DEFAULT_ROWS_PER_SHARD = 200000
DEFAULT_SEED = 1337
DEFAULT_METADATA_NAME = 'validation_split_metadata.json'


class SplitParquetWriter( object ):
    def __init__( self, split_name, output_dir, rows_per_shard ):
        self.split_name = split_name
        self.output_dir = output_dir
        self.rows_per_shard = int( rows_per_shard )
        self.shard_index = 0
        self.rows_written = 0
        self.tables = []

    def append_table( self, table ):
        if len( table ) == 0:
            return
        self.tables.append( table )
        buffered = sum( len(t) for t in self.tables )
        if buffered >= self.rows_per_shard:
            self.flush()

    def flush( self ):
        if len( self.tables ) == 0:
            return
        combined = pa.concat_tables( self.tables )
        offset = 0
        while offset < len( combined ):
            piece = combined.slice( offset, self.rows_per_shard )
            out_name = self.split_name + '-' + format( self.shard_index, '05d' ) + '.parquet'
            out_path = os.path.join( self.output_dir, out_name )
            pq.write_table( piece, out_path, compression='zstd' )
            self.rows_written += len( piece )
            self.shard_index += 1
            offset += len( piece )
        self.tables = []

    def close( self ):
        self.flush()


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Carve a peptide-level val split from train shards in a distilled dataset.' )
    parser.add_argument( '--dataset_root', type=str, default=DEFAULT_DATASET_ROOT,
                         help='Dataset root containing data/train-*.parquet and data/test-*.parquet.' )
    parser.add_argument( '--val_peptides', type=int, default=None,
                         help='Number of train peptides to move into val. Default matches test peptide count from metadata if available.' )
    parser.add_argument( '--rows_per_shard', type=int, default=DEFAULT_ROWS_PER_SHARD,
                         help='Rows per written train/val parquet shard.' )
    parser.add_argument( '--seed', type=int, default=DEFAULT_SEED,
                         help='Random seed for reproducible peptide sampling.' )
    parser.add_argument( '--overwrite', action='store_true',
                         help='Allow replacing existing val shards and train shards.' )
    return parser.parse_args( args )


def discover_split_files( dataset_root, split ):
    return sorted( glob.glob( os.path.join( dataset_root, 'data', split + '-*.parquet' ) ) )


def ensure_ready( dataset_root, overwrite ):
    data_dir = os.path.join( dataset_root, 'data' )
    if not os.path.isdir( data_dir ):
        raise RuntimeError( 'Missing data directory: ' + data_dir )

    train_files = discover_split_files( dataset_root, 'train' )
    test_files = discover_split_files( dataset_root, 'test' )
    val_files = discover_split_files( dataset_root, 'val' )

    if len( train_files ) == 0:
        raise RuntimeError( 'No train shards found under ' + data_dir )
    if len( test_files ) == 0:
        raise RuntimeError( 'No test shards found under ' + data_dir )
    if len( val_files ) > 0 and not overwrite:
        raise RuntimeError( 'Val shards already exist. Use --overwrite to replace them.' )

    return data_dir, train_files, test_files, val_files


def read_existing_metadata( dataset_root ):
    metadata_path = os.path.join( dataset_root, 'distilled_dataset_metadata.json' )
    if not os.path.isfile( metadata_path ):
        return None, metadata_path
    with open( metadata_path, 'r' ) as handle:
        return json.load( handle ), metadata_path


def determine_val_peptide_target( args, existing_metadata ):
    if args.val_peptides is not None:
        return int( args.val_peptides )
    if existing_metadata is not None:
        peptide_counts = existing_metadata.get( 'peptides_written_by_split', {} )
        if 'test' in peptide_counts:
            return int( peptide_counts[ 'test' ] )
    raise RuntimeError( 'Could not determine default val peptide target. Pass --val_peptides explicitly.' )


def collect_train_peptides( train_files ):
    peptides = set()
    counts = Counter()
    for filepath in train_files:
        pf = pq.ParquetFile( filepath )
        for rg_idx in range( pf.metadata.num_row_groups ):
            table = pf.read_row_group( rg_idx, columns=[ 'modified_sequence' ] )
            seqs = table.column( 'modified_sequence' ).to_pylist()
            peptides.update( seqs )
            counts.update( seqs )
    return sorted( peptides ), counts


def sample_val_peptides( train_peptides, target_count, seed ):
    if len( train_peptides ) < target_count:
        raise RuntimeError( 'Train split has only ' + str(len(train_peptides)) +
                            ' unique peptides; requested val=' + str(target_count) )
    rng = random.Random( int( seed ) )
    return set( rng.sample( train_peptides, target_count ) )


def rewrite_train_and_write_val( train_files, data_dir, val_peptides, rows_per_shard ):
    tmp_dir = os.path.join( data_dir, '_val_carve_tmp' )
    os.makedirs( tmp_dir, exist_ok=True )
    tmp_train_dir = os.path.join( tmp_dir, 'train' )
    tmp_val_dir = os.path.join( tmp_dir, 'val' )
    os.makedirs( tmp_train_dir, exist_ok=True )
    os.makedirs( tmp_val_dir, exist_ok=True )

    train_writer = SplitParquetWriter( 'train', tmp_train_dir, rows_per_shard )
    val_writer = SplitParquetWriter( 'val', tmp_val_dir, rows_per_shard )

    counts = {
        'train_rows_out' : 0,
        'val_rows_out' : 0,
        'train_peptides_out' : set(),
        'val_peptides_out' : set(),
    }

    try:
        for i, filepath in enumerate( train_files ):
            print( 'Rewriting train shard ' + str(i + 1) + '/' + str(len(train_files)) + ': ' + os.path.basename( filepath ) )
            pf = pq.ParquetFile( filepath )
            for rg_idx in range( pf.metadata.num_row_groups ):
                table = pf.read_row_group( rg_idx )
                seqs = table.column( 'modified_sequence' ).to_pylist()
                is_val = [ seq in val_peptides for seq in seqs ]
                is_train = [ not flag for flag in is_val ]

                val_table = table.filter( is_val )
                train_table = table.filter( is_train )

                if len( val_table ) > 0:
                    val_writer.append_table( val_table )
                    counts[ 'val_rows_out' ] += len( val_table )
                    counts[ 'val_peptides_out' ].update( val_table.column( 'modified_sequence' ).to_pylist() )

                if len( train_table ) > 0:
                    train_writer.append_table( train_table )
                    counts[ 'train_rows_out' ] += len( train_table )
                    counts[ 'train_peptides_out' ].update( train_table.column( 'modified_sequence' ).to_pylist() )
    finally:
        train_writer.close()
        val_writer.close()

    return tmp_dir, counts


def replace_split_files( data_dir, tmp_dir ):
    for path in glob.glob( os.path.join( data_dir, 'train-*.parquet' ) ):
        os.remove( path )
    for path in glob.glob( os.path.join( data_dir, 'val-*.parquet' ) ):
        os.remove( path )

    for path in sorted( glob.glob( os.path.join( tmp_dir, 'train', 'train-*.parquet' ) ) ):
        os.replace( path, os.path.join( data_dir, os.path.basename( path ) ) )
    for path in sorted( glob.glob( os.path.join( tmp_dir, 'val', 'val-*.parquet' ) ) ):
        os.replace( path, os.path.join( data_dir, os.path.basename( path ) ) )

    for subdir in [ os.path.join( tmp_dir, 'train' ), os.path.join( tmp_dir, 'val' ) ]:
        if os.path.isdir( subdir ):
            os.rmdir( subdir )
    if os.path.isdir( tmp_dir ):
        os.rmdir( tmp_dir )


def write_val_metadata( dataset_root,
                        args,
                        existing_metadata,
                        metadata_path,
                        test_files,
                        val_target,
                        counts ):
    split_metadata = {
        'dataset_root' : os.path.abspath( dataset_root ),
        'seed' : int( args.seed ),
        'val_peptides_requested' : int( val_target ),
        'rows_per_shard' : int( args.rows_per_shard ),
        'test_shards_unchanged' : len( test_files ),
        'train_rows_after' : int( counts[ 'train_rows_out' ] ),
        'val_rows_after' : int( counts[ 'val_rows_out' ] ),
        'train_peptides_after' : int( len( counts[ 'train_peptides_out' ] ) ),
        'val_peptides_after' : int( len( counts[ 'val_peptides_out' ] ) ),
    }

    val_metadata_path = os.path.join( dataset_root, DEFAULT_METADATA_NAME )
    with open( val_metadata_path, 'w' ) as handle:
        json.dump( split_metadata, handle, indent=2 )

    if existing_metadata is not None:
        existing_metadata[ 'rows_written_by_split' ][ 'train' ] = int( counts[ 'train_rows_out' ] )
        existing_metadata[ 'rows_written_by_split' ][ 'val' ] = int( counts[ 'val_rows_out' ] )
        existing_metadata[ 'peptides_written_by_split' ][ 'train' ] = int( len( counts[ 'train_peptides_out' ] ) )
        existing_metadata[ 'peptides_written_by_split' ][ 'val' ] = int( len( counts[ 'val_peptides_out' ] ) )
        existing_metadata[ 'train_shards_written' ] = len( discover_split_files( dataset_root, 'train' ) )
        existing_metadata[ 'val_shards_written' ] = len( discover_split_files( dataset_root, 'val' ) )
        with open( metadata_path, 'w' ) as handle:
            json.dump( existing_metadata, handle, indent=2 )


def main():
    args = parse_args( os.sys.argv[1:] )
    data_dir, train_files, test_files, _ = ensure_ready( args.dataset_root, args.overwrite )
    existing_metadata, metadata_path = read_existing_metadata( args.dataset_root )
    val_target = determine_val_peptide_target( args, existing_metadata )

    print( 'Collecting unique train peptides...' )
    train_peptides, train_counts = collect_train_peptides( train_files )
    print( '  unique train peptides=' + str(len(train_peptides)) )

    val_peptides = sample_val_peptides( train_peptides, val_target, args.seed )
    print( '  sampled val peptides=' + str(len(val_peptides)) )

    tmp_dir, counts = rewrite_train_and_write_val( train_files, data_dir, val_peptides, args.rows_per_shard )
    replace_split_files( data_dir, tmp_dir )

    train_overlap = counts[ 'train_peptides_out' ].intersection( counts[ 'val_peptides_out' ] )
    if len( train_overlap ) > 0:
        raise RuntimeError( 'Train/val overlap detected after rewrite: ' + str(len(train_overlap)) )
    if len( counts[ 'val_peptides_out' ] ) != val_target:
        raise RuntimeError( 'Val peptide count mismatch: expected ' + str(val_target) +
                            ', got ' + str(len(counts[ 'val_peptides_out' ])) )

    write_val_metadata( args.dataset_root,
                        args,
                        existing_metadata,
                        metadata_path,
                        test_files,
                        val_target,
                        counts )

    print( 'Validation carve complete' )
    print( '  train peptides=' + str(len(counts[ 'train_peptides_out' ])) +
           ' rows=' + str(counts[ 'train_rows_out' ]) )
    print( '  val peptides=' + str(len(counts[ 'val_peptides_out' ])) +
           ' rows=' + str(counts[ 'val_rows_out' ]) )
    print( '  test shards untouched=' + str(len(test_files)) )


if __name__ == '__main__':
    main()
