"""Create a validation split by renaming roughly half of the current test shards."""

import argparse
import glob
import json
import os
import random


DEFAULT_DATASET_ROOT = '/Users/searle.brian/Documents/huggingface/data/full_chronologer_suite_distilled'
DEFAULT_SEED = 1337
DEFAULT_VAL_FRACTION = 0.5
DEFAULT_METADATA_NAME = 'validation_split_metadata.json'


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Rename a seeded fraction of test shards to val shards in a distilled dataset.' )
    parser.add_argument( '--dataset_root', type=str, default=DEFAULT_DATASET_ROOT,
                         help='Dataset root containing data/train-*.parquet and data/test-*.parquet.' )
    parser.add_argument( '--seed', type=int, default=DEFAULT_SEED,
                         help='Random seed for reproducible shard selection.' )
    parser.add_argument( '--val_fraction', type=float, default=DEFAULT_VAL_FRACTION,
                         help='Approximate fraction of current test shards to rename as val (default 0.5).' )
    parser.add_argument( '--overwrite', action='store_true',
                         help='Allow replacing any existing val shards.' )
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


def choose_val_shards( test_files, val_fraction, seed ):
    if not ( 0.0 < float( val_fraction ) < 1.0 ):
        raise RuntimeError( '--val_fraction must be between 0 and 1' )

    rng = random.Random( int( seed ) )
    shuffled = list( test_files )
    rng.shuffle( shuffled )
    n_val = max( 1, int( round( len( shuffled ) * float( val_fraction ) ) ) )
    n_val = min( n_val, len( shuffled ) - 1 ) if len( shuffled ) > 1 else 1
    return sorted( shuffled[ :n_val ] )


def clear_existing_val_files( val_files ):
    for path in val_files:
        os.remove( path )


def rename_test_shards( dataset_root, val_files ):
    data_dir = os.path.join( dataset_root, 'data' )
    renamed = []
    for idx, src_path in enumerate( sorted( val_files ) ):
        dst_name = 'val-' + format( idx, '05d' ) + '.parquet'
        dst_path = os.path.join( data_dir, dst_name )
        os.replace( src_path, dst_path )
        renamed.append( dst_path )
    return renamed


def rewrite_metadata_counts( dataset_root ):
    metadata_path = os.path.join( dataset_root, 'distilled_dataset_metadata.json' )
    if not os.path.isfile( metadata_path ):
        return None

    with open( metadata_path, 'r' ) as handle:
        metadata = json.load( handle )

    data_dir = os.path.join( dataset_root, 'data' )
    split_counts = {}
    for split in [ 'train', 'val', 'test' ]:
        split_counts[ split ] = len( glob.glob( os.path.join( data_dir, split + '-*.parquet' ) ) )

    metadata[ 'train_shards_written' ] = int( split_counts[ 'train' ] )
    metadata[ 'val_shards_written' ] = int( split_counts[ 'val' ] )
    metadata[ 'test_shards_written' ] = int( split_counts[ 'test' ] )

    with open( metadata_path, 'w' ) as handle:
        json.dump( metadata, handle, indent=2 )

    return metadata_path


def write_split_metadata( dataset_root, seed, val_fraction, kept_test_count, renamed_val_count ):
    payload = {
        'dataset_root' : os.path.abspath( dataset_root ),
        'strategy' : 'seeded_test_shard_rename',
        'seed' : int( seed ),
        'val_fraction' : float( val_fraction ),
        'val_shards_written' : int( renamed_val_count ),
        'test_shards_remaining' : int( kept_test_count ),
    }
    metadata_path = os.path.join( dataset_root, DEFAULT_METADATA_NAME )
    with open( metadata_path, 'w' ) as handle:
        json.dump( payload, handle, indent=2 )
    return metadata_path


def main():
    args = parse_args( os.sys.argv[1:] )
    _, train_files, test_files, val_files = ensure_ready( args.dataset_root, args.overwrite )
    chosen_val_files = choose_val_shards( test_files, args.val_fraction, args.seed )

    if args.overwrite and len( val_files ) > 0:
        clear_existing_val_files( val_files )

    renamed_val_files = rename_test_shards( args.dataset_root, chosen_val_files )
    remaining_test_files = discover_split_files( args.dataset_root, 'test' )
    metadata_path = write_split_metadata( args.dataset_root,
                                          args.seed,
                                          args.val_fraction,
                                          len( remaining_test_files ),
                                          len( renamed_val_files ) )
    dataset_metadata_path = rewrite_metadata_counts( args.dataset_root )

    print( 'Validation shard rename complete' )
    print( '  train shards unchanged=' + str(len(train_files)) )
    print( '  val shards=' + str(len(renamed_val_files)) )
    print( '  test shards remaining=' + str(len(remaining_test_files)) )
    print( '  metadata=' + metadata_path )
    if dataset_metadata_path is not None:
        print( '  dataset metadata updated=' + dataset_metadata_path )


if __name__ == '__main__':
    main()
