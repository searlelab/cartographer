"""Temporarily rebalance Scout val/test splits by rows instead of whole shards."""

import argparse
import glob
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = os.path.join( '..', 'complete_chronologer_suite_distilled' )
DEFAULT_SEED = 1337
DEFAULT_ROWS_PER_SHARD = 200000
DEFAULT_BATCH_ROWS = 8192
DEFAULT_STAGE_DIRNAME = '_row_rebalance_stage'
DEFAULT_BACKUP_PREFIX = '_row_rebalance_backup_'
SPLIT_METADATA_NAME = 'validation_split_metadata.json'
DISTILLED_METADATA_NAME = 'distilled_dataset_metadata.json'
SCOUT_CACHE_NAME = 'scout_dataset_stats.json'

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


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Rewrite Scout val/test shards so the split is assigned by row instead of whole-shard rename.' )
    parser.add_argument( '--dataset_root', type=str, default=DEFAULT_DATASET_ROOT,
                         help='Dataset root containing data/train-*.parquet, data/val-*.parquet, and data/test-*.parquet.' )
    parser.add_argument( '--val_fraction', type=float, default=None,
                         help='Target fraction of combined val+test rows written to val. Default: read from validation metadata or infer from current shards.' )
    parser.add_argument( '--seed', type=int, default=DEFAULT_SEED,
                         help='Seed for deterministic row selection.' )
    parser.add_argument( '--rows_per_shard', type=int, default=None,
                         help='Rows per output shard. Default: metadata rows_per_shard or 200000.' )
    parser.add_argument( '--batch_rows', type=int, default=DEFAULT_BATCH_ROWS,
                         help='Batch size when streaming input parquet rows.' )
    parser.add_argument( '--overwrite', action='store_true',
                         help='Allow removing an old staging directory from a prior interrupted run.' )
    parser.add_argument( '--dry_run', action='store_true',
                         help='Report the planned rebalance without writing any files.' )
    parser.add_argument( '--repair_only', action='store_true',
                         help='Skip row rewriting and only refresh metadata/cache after a partially successful prior run.' )
    return parser.parse_args( args )


def discover_split_files( dataset_root, split_name ):
    pattern = os.path.join( dataset_root, 'data', split_name + '-*.parquet' )
    return sorted( glob.glob( pattern ) )


def ensure_ready( dataset_root ):
    data_dir = os.path.join( dataset_root, 'data' )
    if not os.path.isdir( data_dir ):
        raise RuntimeError( 'Missing data directory: ' + data_dir )

    train_files = discover_split_files( dataset_root, 'train' )
    val_files = discover_split_files( dataset_root, 'val' )
    test_files = discover_split_files( dataset_root, 'test' )

    if len( train_files ) == 0:
        raise RuntimeError( 'No train shards found under ' + data_dir )
    if len( val_files ) == 0:
        raise RuntimeError( 'No val shards found under ' + data_dir )
    if len( test_files ) == 0:
        raise RuntimeError( 'No test shards found under ' + data_dir )

    return data_dir, train_files, val_files, test_files


def load_json_if_exists( path ):
    if not os.path.isfile( path ):
        return None
    with open( path, 'r' ) as handle:
        return json.load( handle )


def ensure_path_writable( path ):
    if path is None or path == '' or not os.path.exists( path ):
        return
    try:
        os.chmod( path, 0o777 )
    except OSError:
        pass
    if os.name == 'nt':
        try:
            subprocess.run( [ 'attrib', '-R', path ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )
        except OSError:
            pass


def write_json_atomic( path, payload ):
    text = json.dumps( payload, indent=2 )
    parent_dir = os.path.dirname( path )
    if parent_dir != '':
        os.makedirs( parent_dir, exist_ok=True )
        ensure_path_writable( parent_dir )

    if os.path.exists( path ):
        ensure_path_writable( path )

    temp_path = path + '.tmp'
    ensure_path_writable( os.path.dirname( temp_path ) )
    try:
        with open( temp_path, 'w' ) as handle:
            handle.write( text )
        os.replace( temp_path, path )
        return
    except PermissionError:
        if os.path.exists( temp_path ):
            try:
                os.remove( temp_path )
            except OSError:
                pass
        if os.name != 'nt':
            raise

    with tempfile.NamedTemporaryFile( mode='w',
                                      suffix='.json',
                                      delete=False,
                                      dir=os.getcwd(),
                                      encoding='utf-8' ) as handle:
        handle.write( text )
        fallback_temp_path = handle.name

    try:
        subprocess.run(
            [
                'powershell',
                '-NoProfile',
                '-Command',
                "$text = Get-Content -LiteralPath '" + fallback_temp_path.replace( "'", "''" ) +
                "' -Raw; Set-Content -LiteralPath '" + path.replace( "'", "''" ) +
                "' -Value $text -Encoding utf8",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        try:
            os.remove( fallback_temp_path )
        except OSError:
            pass


def infer_rows_per_shard( dataset_root, requested_rows_per_shard ):
    if requested_rows_per_shard is not None:
        rows_per_shard = int( requested_rows_per_shard )
    else:
        metadata = load_json_if_exists( os.path.join( dataset_root, DISTILLED_METADATA_NAME ) ) or {}
        rows_per_shard = int( metadata.get( 'rows_per_shard', DEFAULT_ROWS_PER_SHARD ) )

    if rows_per_shard <= 0:
        raise RuntimeError( '--rows_per_shard must be > 0' )
    return rows_per_shard


def infer_val_fraction( dataset_root, requested_val_fraction, val_files, test_files ):
    if requested_val_fraction is not None:
        val_fraction = float( requested_val_fraction )
    else:
        split_metadata = load_json_if_exists( os.path.join( dataset_root, SPLIT_METADATA_NAME ) ) or {}
        val_fraction = split_metadata.get( 'val_fraction', None )
        if val_fraction is None:
            total_shards = len( val_files ) + len( test_files )
            val_fraction = float( len( val_files ) ) / float( total_shards )

    val_fraction = float( val_fraction )
    if not ( 0.0 < val_fraction < 1.0 ):
        raise RuntimeError( '--val_fraction must be between 0 and 1' )
    return val_fraction


def count_rows( parquet_files ):
    total_rows = 0
    per_file_rows = []
    for path in parquet_files:
        file_rows = int( pq.ParquetFile( path ).metadata.num_rows )
        total_rows += file_rows
        per_file_rows.append( ( path, file_rows ) )
    return total_rows, per_file_rows


def choose_val_row_indices( total_rows, val_fraction, seed ):
    if total_rows <= 1:
        raise RuntimeError( 'Need at least two combined val/test rows to rebalance.' )

    target_val_rows = int( round( float(total_rows) * float(val_fraction) ) )
    target_val_rows = max( 1, target_val_rows )
    target_val_rows = min( target_val_rows, total_rows - 1 )

    rng = random.Random( int( seed ) )
    chosen = rng.sample( range( total_rows ), target_val_rows )
    chosen.sort()
    return target_val_rows, chosen


def prepare_stage_dir( dataset_root, overwrite ):
    stage_dir = os.path.join( dataset_root, DEFAULT_STAGE_DIRNAME )
    if os.path.exists( stage_dir ):
        if stage_dir_is_empty( stage_dir ):
            remove_stage_dir_if_possible( stage_dir )
            if os.path.exists( stage_dir ):
                return stage_dir
        elif not overwrite:
            raise RuntimeError( 'Stage directory already exists. Use --overwrite to replace it: ' + stage_dir )
        else:
            shutil.rmtree( stage_dir )
    os.makedirs( stage_dir, exist_ok=False )
    return stage_dir


def stage_dir_is_empty( stage_dir ):
    if not os.path.isdir( stage_dir ):
        return True
    with os.scandir( stage_dir ) as entries:
        for _ in entries:
            return False
    return True


def remove_stage_dir_if_possible( stage_dir ):
    if not os.path.exists( stage_dir ):
        return True
    ensure_path_writable( stage_dir )
    try:
        os.rmdir( stage_dir )
        return True
    except OSError:
        return False


def read_rows_in_batches( parquet_path, batch_rows ):
    parquet_file = pq.ParquetFile( parquet_path )
    columns = [ 'modified_sequence',
                'precursor_charge_onehot',
                'charge_state_dist',
                'collision_energy_aligned_normed',
                'indexed_retention_time',
                'ccs',
                'intensities_raw' ]
    for record_batch in parquet_file.iter_batches( batch_size=int( batch_rows ), columns=columns ):
        table = pa.Table.from_batches( [ record_batch ] )
        yield {
            'modified_sequence' : table.column( 'modified_sequence' ).to_pylist(),
            'precursor_charge_onehot' : table.column( 'precursor_charge_onehot' ).to_pylist(),
            'charge_state_dist' : table.column( 'charge_state_dist' ).to_pylist(),
            'collision_energy_aligned_normed' : table.column( 'collision_energy_aligned_normed' ).to_pylist(),
            'indexed_retention_time' : table.column( 'indexed_retention_time' ).to_pylist(),
            'ccs' : table.column( 'ccs' ).to_pylist(),
            'intensities_raw' : table.column( 'intensities_raw' ).to_pylist(),
        }


def rewrite_val_test_pool( input_files, stage_dir, rows_per_shard, batch_rows, chosen_val_indices ):
    writers = {
        'val' : SplitParquetWriter( 'val', stage_dir, rows_per_shard ),
        'test' : SplitParquetWriter( 'test', stage_dir, rows_per_shard ),
    }
    next_val_pointer = 0
    global_row_index = 0

    try:
        for path in input_files:
            for batch in read_rows_in_batches( path, batch_rows ):
                n_rows = len( batch[ 'modified_sequence' ] )
                for local_idx in range( n_rows ):
                    is_val = next_val_pointer < len( chosen_val_indices ) and global_row_index == chosen_val_indices[ next_val_pointer ]
                    split_name = 'val' if is_val else 'test'
                    if is_val:
                        next_val_pointer += 1

                    row = {
                        'modified_sequence' : batch[ 'modified_sequence' ][ local_idx ],
                        'precursor_charge_onehot' : batch[ 'precursor_charge_onehot' ][ local_idx ],
                        'charge_state_dist' : batch[ 'charge_state_dist' ][ local_idx ],
                        'collision_energy_aligned_normed' : batch[ 'collision_energy_aligned_normed' ][ local_idx ],
                        'indexed_retention_time' : batch[ 'indexed_retention_time' ][ local_idx ],
                        'ccs' : batch[ 'ccs' ][ local_idx ],
                        'intensities_raw' : batch[ 'intensities_raw' ][ local_idx ],
                    }
                    writers[ split_name ].append( row )
                    global_row_index += 1
    finally:
        writers[ 'val' ].close()
        writers[ 'test' ].close()

    return {
        'rows_seen' : int( global_row_index ),
        'rows_written_by_split' : {
            'val' : int( writers[ 'val' ].rows_written ),
            'test' : int( writers[ 'test' ].rows_written ),
        },
        'shards_written_by_split' : {
            'val' : int( writers[ 'val' ].shard_index ),
            'test' : int( writers[ 'test' ].shard_index ),
        },
        'selected_rows_consumed' : int( next_val_pointer ),
    }


def finalize_swap( data_dir, val_files, test_files, stage_dir ):
    backup_dir = os.path.join( os.path.dirname( data_dir ),
                               DEFAULT_BACKUP_PREFIX + datetime.now().strftime( '%Y%m%d%H%M%S' ) )
    os.makedirs( backup_dir, exist_ok=False )

    for path in val_files + test_files:
        shutil.move( path, os.path.join( backup_dir, os.path.basename( path ) ) )

    staged_files = sorted( glob.glob( os.path.join( stage_dir, '*.parquet' ) ) )
    for path in staged_files:
        shutil.move( path, os.path.join( data_dir, os.path.basename( path ) ) )

    stage_removed = remove_stage_dir_if_possible( stage_dir )
    return backup_dir, stage_removed


def update_distilled_metadata( dataset_root, output_rows, output_shards ):
    metadata_path = os.path.join( dataset_root, DISTILLED_METADATA_NAME )
    metadata = load_json_if_exists( metadata_path )
    if metadata is None:
        return None

    rows_written_by_split = metadata.get( 'rows_written_by_split', {} )
    rows_written_by_split[ 'val' ] = int( output_rows[ 'val' ] )
    rows_written_by_split[ 'test' ] = int( output_rows[ 'test' ] )
    metadata[ 'rows_written_by_split' ] = rows_written_by_split

    metadata[ 'val_shards_written' ] = int( output_shards[ 'val' ] )
    metadata[ 'test_shards_written' ] = int( output_shards[ 'test' ] )

    write_json_atomic( metadata_path, metadata )
    return metadata_path


def write_split_metadata( dataset_root,
                          seed,
                          val_fraction,
                          input_rows,
                          output_rows,
                          output_shards,
                          backup_dir ):
    payload = {
        'dataset_root' : os.path.abspath( dataset_root ),
        'strategy' : 'seeded_row_rebalance_from_existing_val_test_pool',
        'seed' : int( seed ),
        'val_fraction' : float( val_fraction ),
        'input_rows_by_split' : {
            'val' : int( input_rows[ 'val' ] ),
            'test' : int( input_rows[ 'test' ] ),
        },
        'output_rows_by_split' : {
            'val' : int( output_rows[ 'val' ] ),
            'test' : int( output_rows[ 'test' ] ),
        },
        'output_shards_by_split' : {
            'val' : int( output_shards[ 'val' ] ),
            'test' : int( output_shards[ 'test' ] ),
        },
        'backup_dir' : os.path.abspath( backup_dir ),
    }
    metadata_path = os.path.join( dataset_root, SPLIT_METADATA_NAME )
    write_json_atomic( metadata_path, payload )
    return metadata_path


def clear_scout_cache( dataset_root ):
    cache_path = os.path.join( dataset_root, SCOUT_CACHE_NAME )
    if os.path.isfile( cache_path ):
        ensure_path_writable( os.path.dirname( cache_path ) )
        ensure_path_writable( cache_path )
        os.remove( cache_path )
        return cache_path
    return None


def finalize_metadata_and_cleanup( dataset_root,
                                   input_val_rows,
                                   input_test_rows,
                                   output_rows,
                                   output_shards,
                                   seed,
                                   val_fraction,
                                   backup_dir ):
    distilled_metadata_path = update_distilled_metadata( dataset_root, output_rows, output_shards )
    split_metadata_path = write_split_metadata( dataset_root,
                                                seed,
                                                val_fraction,
                                                { 'val' : input_val_rows, 'test' : input_test_rows },
                                                output_rows,
                                                output_shards,
                                                backup_dir )
    cleared_cache_path = clear_scout_cache( dataset_root )
    stage_dir = os.path.join( dataset_root, DEFAULT_STAGE_DIRNAME )
    stage_removed = True
    if os.path.isdir( stage_dir ) and stage_dir_is_empty( stage_dir ):
        stage_removed = remove_stage_dir_if_possible( stage_dir )
    return distilled_metadata_path, split_metadata_path, cleared_cache_path, stage_removed


def format_fraction( numer, denom ):
    if denom <= 0:
        return '0.000000'
    return format( float(numer) / float(denom), '.6f' )


def main():
    args = parse_args( os.sys.argv[1:] )
    data_dir, train_files, val_files, test_files = ensure_ready( args.dataset_root )

    rows_per_shard = infer_rows_per_shard( args.dataset_root, args.rows_per_shard )
    val_fraction = infer_val_fraction( args.dataset_root, args.val_fraction, val_files, test_files )
    if args.batch_rows <= 0:
        raise RuntimeError( '--batch_rows must be > 0' )

    input_val_rows, _ = count_rows( val_files )
    input_test_rows, _ = count_rows( test_files )
    total_rows = int( input_val_rows + input_test_rows )
    target_val_rows, chosen_val_indices = choose_val_row_indices( total_rows, val_fraction, args.seed )
    target_test_rows = total_rows - target_val_rows

    print( 'Row rebalance plan ready' )
    print( '  dataset_root=' + os.path.abspath( args.dataset_root ) )
    print( '  train shards unchanged=' + str(len(train_files)) )
    print( '  current val rows=' + str(input_val_rows) +
           ' (' + format_fraction( input_val_rows, total_rows ) + ')' )
    print( '  current test rows=' + str(input_test_rows) +
           ' (' + format_fraction( input_test_rows, total_rows ) + ')' )
    print( '  target val rows=' + str(target_val_rows) +
           ' (' + format_fraction( target_val_rows, total_rows ) + ')' )
    print( '  target test rows=' + str(target_test_rows) +
           ' (' + format_fraction( target_test_rows, total_rows ) + ')' )
    print( '  rows_per_shard=' + str(rows_per_shard) )
    print( '  seed=' + str(args.seed) )

    if args.dry_run:
        print( 'Dry run only; no files written.' )
        return

    if args.repair_only:
        backup_dirs = sorted( glob.glob( os.path.join( args.dataset_root, DEFAULT_BACKUP_PREFIX + '*' ) ) )
        if len( backup_dirs ) == 0:
            raise RuntimeError( 'No backup directory found for repair-only mode under ' + args.dataset_root )
        output_shards = { 'val' : len( val_files ), 'test' : len( test_files ) }
        output_rows = { 'val' : input_val_rows, 'test' : input_test_rows }
        distilled_metadata_path, split_metadata_path, cleared_cache_path, stage_removed = finalize_metadata_and_cleanup(
            args.dataset_root,
            input_val_rows,
            input_test_rows,
            output_rows,
            output_shards,
            args.seed,
            val_fraction,
            backup_dirs[ -1 ]
        )
        print( 'Repair-only finalize complete' )
        print( '  split metadata=' + split_metadata_path )
        if distilled_metadata_path is not None:
            print( '  distilled metadata updated=' + distilled_metadata_path )
        if cleared_cache_path is not None:
            print( '  scout cache removed=' + cleared_cache_path )
        if not stage_removed:
            print( '  stage directory retained=' + os.path.join( args.dataset_root, DEFAULT_STAGE_DIRNAME ) )
        return

    stage_dir = prepare_stage_dir( args.dataset_root, args.overwrite )
    input_files = sorted( val_files + test_files )
    rewrite_stats = rewrite_val_test_pool( input_files,
                                           stage_dir,
                                           rows_per_shard,
                                           args.batch_rows,
                                           chosen_val_indices )

    if rewrite_stats[ 'rows_seen' ] != total_rows:
        raise RuntimeError( 'Rewrite saw ' + str(rewrite_stats[ 'rows_seen' ]) +
                            ' rows but expected ' + str(total_rows) )
    if rewrite_stats[ 'selected_rows_consumed' ] != target_val_rows:
        raise RuntimeError( 'Rewrite consumed ' + str(rewrite_stats[ 'selected_rows_consumed' ]) +
                            ' selected val rows but expected ' + str(target_val_rows) )
    if rewrite_stats[ 'rows_written_by_split' ][ 'val' ] != target_val_rows:
        raise RuntimeError( 'Wrote ' + str(rewrite_stats[ 'rows_written_by_split' ][ 'val' ]) +
                            ' val rows but expected ' + str(target_val_rows) )
    if rewrite_stats[ 'rows_written_by_split' ][ 'test' ] != target_test_rows:
        raise RuntimeError( 'Wrote ' + str(rewrite_stats[ 'rows_written_by_split' ][ 'test' ]) +
                            ' test rows but expected ' + str(target_test_rows) )

    backup_dir, stage_removed = finalize_swap( data_dir, val_files, test_files, stage_dir )
    distilled_metadata_path, split_metadata_path, cleared_cache_path, stage_removed_post = finalize_metadata_and_cleanup(
        args.dataset_root,
        input_val_rows,
        input_test_rows,
        rewrite_stats[ 'rows_written_by_split' ],
        rewrite_stats[ 'shards_written_by_split' ],
        args.seed,
        val_fraction,
        backup_dir
    )
    stage_removed = stage_removed and stage_removed_post

    print( 'Row rebalance complete' )
    print( '  backup=' + backup_dir )
    print( '  val shards=' + str(rewrite_stats[ 'shards_written_by_split' ][ 'val' ]) +
           ', rows=' + str(rewrite_stats[ 'rows_written_by_split' ][ 'val' ]) )
    print( '  test shards=' + str(rewrite_stats[ 'shards_written_by_split' ][ 'test' ]) +
           ', rows=' + str(rewrite_stats[ 'rows_written_by_split' ][ 'test' ]) )
    print( '  split metadata=' + split_metadata_path )
    if distilled_metadata_path is not None:
        print( '  distilled metadata updated=' + distilled_metadata_path )
    if cleared_cache_path is not None:
        print( '  scout cache removed=' + cleared_cache_path )
    if not stage_removed:
        print( '  stage directory retained=' + os.path.join( args.dataset_root, DEFAULT_STAGE_DIRNAME ) )


if __name__ == '__main__':
    main()
