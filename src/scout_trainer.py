import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch.utils.data import DataLoader, TensorDataset

from constants import seed as default_seed
from scout_loader import ScoutDistilledDataset, discover_split_files, scan_distilled_dataset
from scout_loss import ScoutMultiTaskLoss
from scout_model import initialize_scout_model
from scout_settings import hyperparameters, max_peptide_len, metadata_filename, ms2_vector_len, n_ion_channels, progress_tick_rows, training_parameters
from tensorize import aa_to_int, nterm_unimod_map, residue_unimod_map, residues
from training_loop import resolve_device, train_model


MINI_EVAL_SAMPLE_ROWS = 8192
MINI_EVAL_INTERVAL_BATCHES = 10000
MINI_EVAL_SEED = default_seed
MINI_EVAL_BATCH_SIZE_CAP = 1024
EVAL_MIN_BATCH_SIZE = 1


def parse_args( args ):
    src_dir = os.path.dirname( os.path.abspath( __file__ ) )
    timestamp = datetime.now().strftime( '%Y%m%d%H%M%S' )
    default_out_filename = 'Scout_' + timestamp + '.pt'

    parser = argparse.ArgumentParser( description='Train Scout multi-task distilled peptide property model' )
    parser.add_argument( '--dataset_root',
                         type=str,
                         required=True,
                         help='Path to distilled Scout dataset root containing data/train-*.parquet, optional data/val-*.parquet, and data/test-*.parquet' )
    parser.add_argument( '--output_file',
                         type=str,
                         default=default_out_filename,
                         help='Model filename' )
    parser.add_argument( '--output_dir',
                         type=str,
                         default=os.path.join( src_dir, '..', 'models' ),
                         help='Directory to save model' )
    parser.add_argument( '--device',
                         type=str,
                         default='auto',
                         help='Device for training {auto, mps, cuda, cpu}' )
    parser.add_argument( '--num_workers',
                         type=int,
                         default=0,
                         help='DataLoader workers (default 0)' )
    parser.add_argument( '--patience',
                         type=int,
                         default=None,
                         help='Early stopping patience' )
    parser.add_argument( '--model_file',
                         type=str,
                         default=None,
                         help='Path to a source checkpoint for resume/warm-start' )
    parser.add_argument( '--start_epoch',
                         type=int,
                         default=1,
                         help='Epoch number to begin at (default: 1)' )
    parser.add_argument( '--n_epochs',
                         type=int,
                         default=None,
                         help='Total epoch count, overrides settings' )
    parser.add_argument( '--eval_batch_size',
                         type=int,
                         default=4096,
                         help='Batch size used for denormalized evaluation metrics' )
    return parser.parse_args( args )


def _top_counts( counter_dict, n=10 ):
    items = sorted( counter_dict.items(), key=lambda item: ( -item[1], item[0] ) )
    return items[:n]


def _serialize_training_parameters():
    serialized = dict( training_parameters )
    optimizer_obj = serialized.get( 'optimizer' )
    serialized[ 'optimizer' ] = optimizer_obj.__name__ if optimizer_obj is not None else 'None'
    return serialized


def _stats_to_summary( stats ):
    return { 'rows_total' : int( stats[ 'rows_total' ] ),
             'rows_tokenized' : int( stats[ 'rows_tokenized' ] ),
             'ms2_rows' : int( stats[ 'ms2_rows' ] ),
             'irt_rows' : int( stats[ 'irt_rows' ] ),
             'ccs_rows' : int( stats[ 'ccs_rows' ] ),
             'skip_counts' : dict( sorted( stats[ 'skip_counts' ].items() ) ), }


def _scalar_stats_from_train( train_stats ):
    def mean_and_std( total, total_sq, count ):
        if count <= 0:
            return 0.0, 1.0
        mean = float( total ) / float( count )
        variance = max( 0.0, float( total_sq ) / float( count ) - mean * mean )
        std = variance ** 0.5
        if std <= 0.0:
            std = 1.0
        return mean, std

    irt_mean, irt_std = mean_and_std( train_stats[ 'irt_sum' ], train_stats[ 'irt_sum_sq' ], train_stats[ 'irt_rows' ] )
    ccs_mean, ccs_std = mean_and_std( train_stats[ 'ccs_sum' ], train_stats[ 'ccs_sum_sq' ], train_stats[ 'ccs_rows' ] )
    return { 'irt_mean' : float( irt_mean ),
             'irt_std' : float( irt_std ),
             'ccs_mean' : float( ccs_mean ),
             'ccs_std' : float( ccs_std ), }


def _format_elapsed( start_time ):
    elapsed_seconds = max( 0, int( time.time() - start_time ) )
    return str( timedelta( seconds=elapsed_seconds ) )


def _format_rows_per_second( rows_processed, start_time ):
    elapsed_seconds = max( time.time() - start_time, 1e-9 )
    return format( float(rows_processed) / elapsed_seconds, '.1f' )


def _is_cuda_oom_error( error, device ):
    device_str = str( device )
    if not device_str.startswith( 'cuda' ):
        return False
    if isinstance( error, torch.cuda.OutOfMemoryError ):
        return True
    return 'out of memory' in str(error).lower()


def _evaluate_scout_loader( model,
                            loader,
                            scalar_stats,
                            device='cpu',
                            label=None,
                            expected_rows=None,
                            progress_tick_rows=0,
                            effective_batch_size=None,
                            start_time=None ):
    original_training = model.training
    original_device = next( model.parameters() ).device
    model.to( device )
    model.eval()
    if start_time is None:
        start_time = time.time()

    total_rows = 0
    next_progress_rows = int( progress_tick_rows ) if progress_tick_rows and progress_tick_rows > 0 else None
    if label is not None:
        rows_text = 'unknown'
        if expected_rows is not None:
            rows_text = str( int(expected_rows) )
        batch_text = 'unknown' if effective_batch_size is None else str( int(effective_batch_size) )
        print( label + ' start: device=' + str(device) +
               ', batch_size=' + batch_text +
               ', expected_rows=' + rows_text )

    totals = { 'ms2_cosine_sum' : 0.0,
               'ms2_count' : 0,
               'irt_abs_sum' : 0.0,
               'irt_sq_sum' : 0.0,
               'irt_count' : 0,
               'ccs_abs_sum' : 0.0,
               'ccs_sq_sum' : 0.0,
               'ccs_count' : 0, }

    with torch.inference_mode():
        for seq, charge, nce, target_bundle, mask_bundle in loader:
            seq = seq.to( device )
            charge = charge.to( device )
            nce = nce.to( device )
            target_bundle = target_bundle.to( device )
            mask_bundle = mask_bundle.to( device )
            pred = model( seq, charge, nce )
            total_rows += int( seq.shape[0] )

            if label is not None and next_progress_rows is not None:
                while total_rows >= next_progress_rows:
                    if expected_rows is not None and expected_rows > 0:
                        percent = min( 100.0, 100.0 * float(total_rows) / float(expected_rows) )
                        print( label + ' progress: ' +
                               format( percent, '6.2f' ) + '% (' +
                               str(total_rows) + '/' + str(int(expected_rows)) +
                               ' rows, ' + _format_rows_per_second( total_rows, start_time ) +
                               ' rows/s, elapsed ' + _format_elapsed( start_time ) + ')' )
                    else:
                        print( label + ' progress: ' + str(total_rows) +
                               ' rows, ' + _format_rows_per_second( total_rows, start_time ) +
                               ' rows/s, elapsed ' + _format_elapsed( start_time ) )
                    next_progress_rows += int( progress_tick_rows )

            mask_ms2 = mask_bundle[ :, 0 ] > 0.5
            if torch.any( mask_ms2 ):
                pred_ms2 = pred[ 'ms2' ][ mask_ms2 ]
                true_ms2 = target_bundle[ mask_ms2, :ms2_vector_len ]
                pred_ms2 = pred_ms2 / pred_ms2.norm( dim=1, keepdim=True ).clamp( min=1e-7 )
                true_ms2 = true_ms2 / true_ms2.norm( dim=1, keepdim=True ).clamp( min=1e-7 )
                totals[ 'ms2_cosine_sum' ] += float( torch.sum( torch.sum( pred_ms2 * true_ms2, dim=1 ) ).item() )
                totals[ 'ms2_count' ] += int( pred_ms2.shape[0] )

            mask_irt = mask_bundle[ :, 1 ] > 0.5
            if torch.any( mask_irt ):
                pred_irt = pred[ 'irt' ][ mask_irt, 0 ] * scalar_stats[ 'irt_std' ] + scalar_stats[ 'irt_mean' ]
                true_irt = target_bundle[ mask_irt, ms2_vector_len ] * scalar_stats[ 'irt_std' ] + scalar_stats[ 'irt_mean' ]
                diff_irt = pred_irt - true_irt
                totals[ 'irt_abs_sum' ] += float( torch.sum( torch.abs( diff_irt ) ).item() )
                totals[ 'irt_sq_sum' ] += float( torch.sum( diff_irt * diff_irt ).item() )
                totals[ 'irt_count' ] += int( diff_irt.shape[0] )

            mask_ccs = mask_bundle[ :, 2 ] > 0.5
            if torch.any( mask_ccs ):
                pred_ccs = pred[ 'ccs' ][ mask_ccs, 0 ] * scalar_stats[ 'ccs_std' ] + scalar_stats[ 'ccs_mean' ]
                true_ccs = target_bundle[ mask_ccs, ms2_vector_len + 1 ] * scalar_stats[ 'ccs_std' ] + scalar_stats[ 'ccs_mean' ]
                diff_ccs = pred_ccs - true_ccs
                totals[ 'ccs_abs_sum' ] += float( torch.sum( torch.abs( diff_ccs ) ).item() )
                totals[ 'ccs_sq_sum' ] += float( torch.sum( diff_ccs * diff_ccs ).item() )
                totals[ 'ccs_count' ] += int( diff_ccs.shape[0] )

    metrics = { 'test_ms2_cosine' : 0.0,
                'test_ms2_count' : int( totals[ 'ms2_count' ] ),
                'test_irt_mae' : 0.0,
                'test_irt_rmse' : 0.0,
                'test_irt_count' : int( totals[ 'irt_count' ] ),
                'test_ccs_mae' : 0.0,
                'test_ccs_rmse' : 0.0,
                'test_ccs_count' : int( totals[ 'ccs_count' ] ), }

    if totals[ 'ms2_count' ] > 0:
        metrics[ 'test_ms2_cosine' ] = totals[ 'ms2_cosine_sum' ] / totals[ 'ms2_count' ]
    if totals[ 'irt_count' ] > 0:
        metrics[ 'test_irt_mae' ] = totals[ 'irt_abs_sum' ] / totals[ 'irt_count' ]
        metrics[ 'test_irt_rmse' ] = ( totals[ 'irt_sq_sum' ] / totals[ 'irt_count' ] ) ** 0.5
    if totals[ 'ccs_count' ] > 0:
        metrics[ 'test_ccs_mae' ] = totals[ 'ccs_abs_sum' ] / totals[ 'ccs_count' ]
        metrics[ 'test_ccs_rmse' ] = ( totals[ 'ccs_sq_sum' ] / totals[ 'ccs_count' ] ) ** 0.5
    if label is not None:
        print( label + ' complete: processed ' + str(total_rows) +
               ' rows in ' + _format_elapsed( start_time ) +
               ' (' + _format_rows_per_second( total_rows, start_time ) + ' rows/s)' )
    model.to( original_device )
    if original_training:
        model.train()
    return metrics


def _compute_balanced_checkpoint_score( metrics ):
    ms2_component = ( 1.0 - float( metrics[ 'test_ms2_cosine' ] ) ) / 0.1
    rt_component = float( metrics[ 'test_irt_mae' ] ) / 1.0
    ccs_component = float( metrics[ 'test_ccs_mae' ] ) / 10.0
    return ( ( ms2_component ** 2 + rt_component ** 2 + ccs_component ** 2 ) / 3.0 ) ** 0.5


def _evaluate_scout_dataset_with_backoff( model,
                                          dataset,
                                          scalar_stats,
                                          batch_size,
                                          num_workers,
                                          device='cpu',
                                          label=None,
                                          expected_rows=None,
                                          progress_tick_rows=0 ):
    device = resolve_device( device )
    requested_batch_size = max( int(batch_size), 1 )
    current_batch_size = requested_batch_size
    attempted_batch_sizes = []

    while True:
        attempted_batch_sizes.append( int(current_batch_size) )
        loader = DataLoader( dataset, current_batch_size, shuffle=False, num_workers=num_workers )
        try:
            metrics = _evaluate_scout_loader( model,
                                             loader,
                                             scalar_stats,
                                             device=device,
                                             label=label,
                                             expected_rows=expected_rows,
                                             progress_tick_rows=progress_tick_rows,
                                             effective_batch_size=current_batch_size,
                                             start_time=time.time() )
            metrics[ 'effective_eval_batch_size' ] = int( current_batch_size )
            return metrics, int( current_batch_size )
        except RuntimeError as error:
            if not _is_cuda_oom_error( error, device ):
                raise
            if current_batch_size <= EVAL_MIN_BATCH_SIZE:
                attempts = ', '.join( str(v) for v in attempted_batch_sizes )
                raise RuntimeError( ( label or 'Scout evaluation' ) +
                                    ' failed after CUDA OOM retries with batch sizes: ' + attempts ) from error

            next_batch_size = max( EVAL_MIN_BATCH_SIZE, current_batch_size // 2 )
            if next_batch_size == current_batch_size:
                next_batch_size = max( EVAL_MIN_BATCH_SIZE, current_batch_size - 1 )
            print( ( label or 'Scout evaluation' ) +
                   ' CUDA OOM at batch_size=' + str(current_batch_size) +
                   '; retrying with batch_size=' + str(next_batch_size) )
            torch.cuda.empty_cache()
            current_batch_size = next_batch_size


def evaluate_scout( model,
                    test_files,
                    scalar_stats,
                    batch_size,
                    num_workers,
                    device='cpu',
                    label=None,
                    expected_rows=None,
                    progress_tick_rows=0,
                    return_effective_batch_size=False ):
    dataset = ScoutDistilledDataset( test_files,
                                     scalar_stats[ 'irt_mean' ],
                                     scalar_stats[ 'irt_std' ],
                                     scalar_stats[ 'ccs_mean' ],
                                     scalar_stats[ 'ccs_std' ],
                                     shuffle_files=False )
    metrics, effective_batch_size = _evaluate_scout_dataset_with_backoff( model,
                                                                          dataset,
                                                                          scalar_stats,
                                                                          batch_size,
                                                                          num_workers,
                                                                          device=device,
                                                                          label=label,
                                                                          expected_rows=expected_rows,
                                                                          progress_tick_rows=progress_tick_rows )
    if return_effective_batch_size:
        return metrics, effective_batch_size
    return metrics


def evaluate_scout_dataset( model,
                            dataset,
                            scalar_stats,
                            batch_size,
                            num_workers,
                            device='cpu',
                            label=None,
                            expected_rows=None,
                            progress_tick_rows=0,
                            return_effective_batch_size=False ):
    metrics, effective_batch_size = _evaluate_scout_dataset_with_backoff( model,
                                                                          dataset,
                                                                          scalar_stats,
                                                                          batch_size,
                                                                          num_workers,
                                                                          device=device,
                                                                          label=label,
                                                                          expected_rows=expected_rows,
                                                                          progress_tick_rows=progress_tick_rows )
    if return_effective_batch_size:
        return metrics, effective_batch_size
    return metrics


def build_fixed_mini_eval_loader( test_files, scalar_stats, batch_size, sample_rows=MINI_EVAL_SAMPLE_ROWS, seed=MINI_EVAL_SEED ):
    dataset = ScoutDistilledDataset( test_files,
                                     scalar_stats[ 'irt_mean' ],
                                     scalar_stats[ 'irt_std' ],
                                     scalar_stats[ 'ccs_mean' ],
                                     scalar_stats[ 'ccs_std' ],
                                     shuffle_files=False )
    rng = random.Random( int(seed) )
    reservoir = []
    total_seen = 0

    for row in dataset:
        total_seen += 1
        if len( reservoir ) < sample_rows:
            reservoir.append( row )
            continue

        replace_idx = rng.randint( 0, total_seen - 1 )
        if replace_idx < sample_rows:
            reservoir[ replace_idx ] = row

    if len( reservoir ) == 0:
        raise RuntimeError( 'Mini-eval sample is empty; no usable rows found.' )

    seq_tensor = torch.stack( [ row[0] for row in reservoir ] )
    charge_tensor = torch.stack( [ row[1] for row in reservoir ] )
    nce_tensor = torch.stack( [ row[2] for row in reservoir ] )
    target_tensor = torch.stack( [ row[3] for row in reservoir ] )
    mask_tensor = torch.stack( [ row[4] for row in reservoir ] )
    sample_dataset = TensorDataset( seq_tensor, charge_tensor, nce_tensor, target_tensor, mask_tensor )
    loader = DataLoader( sample_dataset, batch_size, shuffle=False )

    sample_info = { 'sample_rows_requested' : int( sample_rows ),
                    'sample_rows_actual' : int( len( reservoir ) ),
                    'rows_seen' : int( total_seen ),
                    'batch_size' : int( batch_size ),
                    'seed' : int( seed ),
                    'ms2_rows' : int( mask_tensor[:, 0].sum().item() ),
                    'irt_rows' : int( mask_tensor[:, 1].sum().item() ),
                    'ccs_rows' : int( mask_tensor[:, 2].sum().item() ), }
    return loader, sample_info


class ScoutMiniEvalReporter( object ):
    def __init__( self, loader, scalar_stats, eval_device, interval_batches=MINI_EVAL_INTERVAL_BATCHES ):
        self.loader = loader
        self.scalar_stats = scalar_stats
        self.eval_device = eval_device
        self.interval_batches = int( interval_batches )
        self.global_batches = 0
        self.start_time = time.time()
        self.rows_printed = 0
        self.header_every = 20

    def print_header( self ):
        print( 'Mini Eval (fixed seeded test sample)' )
        print( ' epoch | train_batch | elapsed  | ckpt_rms | ms2_cos  | n_ms2 | irt_mae  | n_irt | ccs_mae  | n_ccs ' )
        print( '-------|-------------|----------|----------|----------|-------|----------|-------|----------|-------' )

    def __call__( self, model, epoch, phase, batch_index, batch_size, batch_loss, device ):
        if phase != 'train':
            return

        self.global_batches += 1
        if self.global_batches % self.interval_batches != 0:
            return

        if self.rows_printed % self.header_every == 0:
            self.print_header()

        metrics = _evaluate_scout_loader( model, self.loader, self.scalar_stats, device=self.eval_device )
        checkpoint_score = _compute_balanced_checkpoint_score( metrics )
        elapsed = str( timedelta( seconds=int( time.time() - self.start_time ) ) )
        print( format( int(epoch), '6d' ) + ' | ' +
               format( int(self.global_batches), '11d' ) + ' | ' +
               elapsed.rjust(8) + ' | ' +
               format( checkpoint_score, '8.4f' ) + ' | ' +
               format( metrics[ 'test_ms2_cosine' ], '8.4f' ) + ' | ' +
               format( metrics[ 'test_ms2_count' ], '5d' ) + ' | ' +
               format( metrics[ 'test_irt_mae' ], '8.4f' ) + ' | ' +
               format( metrics[ 'test_irt_count' ], '5d' ) + ' | ' +
               format( metrics[ 'test_ccs_mae' ], '8.4f' ) + ' | ' +
               format( metrics[ 'test_ccs_count' ], '5d' ) )
        self.rows_printed += 1


class ScoutCheckpointMetric( object ):
    def __init__( self, scalar_stats, eval_device, eval_batch_size, num_workers, expected_rows, progress_tick_rows ):
        self.scalar_stats = scalar_stats
        self.eval_device = eval_device
        self.eval_batch_size = int( eval_batch_size )
        self.num_workers = int( num_workers )
        self.expected_rows = int( expected_rows ) if expected_rows is not None else None
        self.progress_tick_rows = int( progress_tick_rows )
        self.last_metrics = None
        self.last_effective_eval_batch_size = None

    def __call__( self, model, dataset, phase, device, epoch, epoch_loss ):
        metrics, effective_batch_size = evaluate_scout_dataset( model,
                                                                dataset,
                                                                self.scalar_stats,
                                                                self.eval_batch_size,
                                                                self.num_workers,
                                                                device=self.eval_device,
                                                                label='Val checkpoint',
                                                                expected_rows=self.expected_rows,
                                                                progress_tick_rows=self.progress_tick_rows,
                                                                return_effective_batch_size=True )
        score = _compute_balanced_checkpoint_score( metrics )
        self.last_metrics = dict( metrics )
        self.last_metrics[ 'balanced_checkpoint_score' ] = float( score )
        self.last_effective_eval_batch_size = int( effective_batch_size )
        print( 'Val checkpoint metrics: MS2 cosine=' + format( metrics[ 'test_ms2_cosine' ], '.6f' ) +
                ', iRT MAE=' + format( metrics[ 'test_irt_mae' ], '.6f' ) +
                ', CCS MAE=' + format( metrics[ 'test_ccs_mae' ], '.6f' ) +
                ', eval_batch_size=' + str( effective_batch_size ) )
        return float( score ), 'balanced_rms'


def write_metadata( metadata_path,
                    output_file_name,
                    dataset_root,
                    train_files,
                    val_files,
                    test_files,
                    scalar_stats,
                    train_scan,
                    val_scan,
                    test_scan,
                    metrics,
                    best_loss,
                    checkpoint_metrics=None ):
    residue_map_json = [ { 'residue' : aa, 'unimod' : unimod, 'token' : token }
                         for ( aa, unimod ), token in sorted( residue_unimod_map.items() ) ]
    metadata = { 'schema_version' : 1,
                 'created_at_utc' : datetime.now( timezone.utc ).strftime( '%Y-%m-%dT%H:%M:%SZ' ),
                 'model_checkpoint' : os.path.abspath( output_file_name ),
                 'dataset_root' : os.path.abspath( dataset_root ),
                 'splits' : { 'train_files' : [ os.path.abspath( p ) for p in train_files ],
                              'val_files' : [ os.path.abspath( p ) for p in val_files ],
                              'test_files' : [ os.path.abspath( p ) for p in test_files ], },
                 'tokenizer' : { 'residues' : list( residues ),
                                 'aa_to_int' : { key : value for key, value in sorted( aa_to_int.items() ) },
                                 'nterm_unimod_map' : dict( sorted( nterm_unimod_map.items() ) ),
                                 'residue_unimod_map' : residue_map_json, },
                 'scalar_stats' : { 'irt_mean' : float( scalar_stats[ 'irt_mean' ] ),
                                    'irt_std' : float( scalar_stats[ 'irt_std' ] ),
                                    'ccs_mean' : float( scalar_stats[ 'ccs_mean' ] ),
                                    'ccs_std' : float( scalar_stats[ 'ccs_std' ] ), },
                 'coverage' : { 'train' : _stats_to_summary( train_scan ),
                                'val' : _stats_to_summary( val_scan ),
                                'test' : _stats_to_summary( test_scan ), },
                 'hyperparameters' : { 'embed_dimension' : int( hyperparameters[ 'embed_dimension' ] ),
                                       'nce_encode_dimension' : int( hyperparameters[ 'nce_encode_dimension' ] ),
                                       'n_resnet_blocks' : int( hyperparameters[ 'n_resnet_blocks' ] ),
                                       'kernel_size' : int( hyperparameters[ 'kernel_size' ] ),
                                       'activation_function' : hyperparameters[ 'activation_function' ],
                                       'max_peptide_len' : int( max_peptide_len ),
                                       'n_ion_channels' : int( n_ion_channels ),
                                       'ms2_vector_len' : int( ms2_vector_len ), },
                 'training_parameters' : _serialize_training_parameters(),
                 'best_val_loss' : float( best_loss ),
                 'best_val_checkpoint_score' : float( best_loss ),
                 'best_val_checkpoint_metrics' : None if checkpoint_metrics is None else dict( checkpoint_metrics ),
                 'test_metrics' : dict( metrics ), }
    with open( metadata_path, 'w' ) as handle:
        json.dump( metadata, handle, indent=2, sort_keys=True )


def train_scout( dataset_root,
                 output_file_name,
                 device='auto',
                 num_workers=0,
                 patience=None,
                 model_file=None,
                 start_epoch=1,
                 n_epochs=None,
                  eval_batch_size=4096 ):
    print( 'Scout training initiated' )
    eval_device = resolve_device( device )

    train_files = discover_split_files( dataset_root, 'train' )
    val_files = discover_split_files( dataset_root, 'val' )
    test_files = discover_split_files( dataset_root, 'test' )

    if len( train_files ) == 0:
        raise RuntimeError( 'No train parquet files found in ' + dataset_root )
    if len( val_files ) == 0:
        raise RuntimeError( 'No val parquet files found in ' + dataset_root +
                            '; Scout now requires a validation split for checkpoint selection.' )
    if len( test_files ) == 0:
        raise RuntimeError( 'No test parquet files found in ' + dataset_root )

    print( 'Found ' + str(len(train_files)) + ' train shards, ' +
           str(len(val_files)) + ' val shards, ' +
           str(len(test_files)) + ' test shards' )
    print( 'Using train shards for fitting, val shards for in-training checkpoint selection, and test shards only for final evaluation.' )

    print( 'Scanning train split...' )
    train_scan = scan_distilled_dataset( train_files )
    print( 'Train tokenized rows=' + str( train_scan[ 'rows_tokenized' ] ) +
           ', MS2=' + str( train_scan[ 'ms2_rows' ] ) +
           ', iRT=' + str( train_scan[ 'irt_rows' ] ) +
           ', CCS=' + str( train_scan[ 'ccs_rows' ] ) )
    if len( train_scan[ 'skip_counts' ] ) > 0:
        parts = [ key + '=' + str(value) for key, value in _top_counts( train_scan[ 'skip_counts' ], n=12 ) ]
        print( 'Train top skip reasons: ' + ', '.join( parts ) )

    print( 'Scanning val split...' )
    val_scan = scan_distilled_dataset( val_files )
    print( 'Val tokenized rows=' + str( val_scan[ 'rows_tokenized' ] ) +
           ', MS2=' + str( val_scan[ 'ms2_rows' ] ) +
           ', iRT=' + str( val_scan[ 'irt_rows' ] ) +
           ', CCS=' + str( val_scan[ 'ccs_rows' ] ) )
    if len( val_scan[ 'skip_counts' ] ) > 0:
        parts = [ key + '=' + str(value) for key, value in _top_counts( val_scan[ 'skip_counts' ], n=12 ) ]
        print( 'Val top skip reasons: ' + ', '.join( parts ) )

    # Test coverage is logged for passive dashboarding and the final holdout report only.
    print( 'Scanning test split...' )
    test_scan = scan_distilled_dataset( test_files )
    print( 'Test tokenized rows=' + str( test_scan[ 'rows_tokenized' ] ) +
           ', MS2=' + str( test_scan[ 'ms2_rows' ] ) +
           ', iRT=' + str( test_scan[ 'irt_rows' ] ) +
           ', CCS=' + str( test_scan[ 'ccs_rows' ] ) )
    if len( test_scan[ 'skip_counts' ] ) > 0:
        parts = [ key + '=' + str(value) for key, value in _top_counts( test_scan[ 'skip_counts' ], n=12 ) ]
        print( 'Test top skip reasons: ' + ', '.join( parts ) )

    scalar_stats = _scalar_stats_from_train( train_scan )
    print( 'Train-only scalar normalization: iRT mean=' + format( scalar_stats[ 'irt_mean' ], '.6f' ) +
           ', std=' + format( scalar_stats[ 'irt_std' ], '.6f' ) +
           '; CCS mean=' + format( scalar_stats[ 'ccs_mean' ], '.6f' ) +
           ', std=' + format( scalar_stats[ 'ccs_std' ], '.6f' ) )

    # Test rows feed only the passive mini-eval dashboard; checkpoint selection stays on validation data.
    mini_eval_batch_size = max( 1, min( int(eval_batch_size), MINI_EVAL_BATCH_SIZE_CAP ) )
    mini_eval_loader, mini_eval_info = build_fixed_mini_eval_loader( test_files,
                                                                     scalar_stats,
                                                                     mini_eval_batch_size,
                                                                     sample_rows=MINI_EVAL_SAMPLE_ROWS,
                                                                     seed=MINI_EVAL_SEED )
    print( 'Mini-eval test sample: rows=' + str( mini_eval_info[ 'sample_rows_actual' ] ) +
           ' of ' + str( mini_eval_info[ 'rows_seen' ] ) +
           ', batch_size=' + str( mini_eval_info[ 'batch_size' ] ) +
           ', seed=' + str( mini_eval_info[ 'seed' ] ) +
           ', MS2=' + str( mini_eval_info[ 'ms2_rows' ] ) +
           ', iRT=' + str( mini_eval_info[ 'irt_rows' ] ) +
           ', CCS=' + str( mini_eval_info[ 'ccs_rows' ] ) )

    datasets = { 'train' : ScoutDistilledDataset( train_files,
                                                  scalar_stats[ 'irt_mean' ],
                                                  scalar_stats[ 'irt_std' ],
                                                  scalar_stats[ 'ccs_mean' ],
                                                  scalar_stats[ 'ccs_std' ],
                                                  shuffle_files=True ),
                 'val' : ScoutDistilledDataset( val_files,
                                                scalar_stats[ 'irt_mean' ],
                                                scalar_stats[ 'irt_std' ],
                                                scalar_stats[ 'ccs_mean' ],
                                                scalar_stats[ 'ccs_std' ],
                                                shuffle_files=False ), }

    model = initialize_scout_model( model_file=model_file, map_location='cpu' )
    loss_fx = ScoutMultiTaskLoss()
    optimizer = training_parameters[ 'optimizer' ]( list( model.parameters() ),
                                                    lr=training_parameters[ 'learning_rate' ] )
    num_epochs = n_epochs if n_epochs is not None else training_parameters[ 'n_epochs' ]
    mini_eval_reporter = ScoutMiniEvalReporter( mini_eval_loader,
                                                scalar_stats,
                                                eval_device,
                                                interval_batches=MINI_EVAL_INTERVAL_BATCHES )
    checkpoint_metric = ScoutCheckpointMetric( scalar_stats,
                                              eval_device,
                                              eval_batch_size,
                                              num_workers,
                                              val_scan[ 'rows_tokenized' ],
                                              progress_tick_rows )

    best_checkpoint_score = train_model( model,
                                         datasets,
                                         training_parameters[ 'initial_batch_size' ],
                                         training_parameters[ 'max_batch_size' ],
                                         training_parameters[ 'epochs_to_2x_batch' ],
                                         loss_fx,
                                         optimizer,
                                         num_epochs,
                                         device,
                                         device,
                                         output_file_name,
                                         progress_tick_rows=0,
                                         num_workers=num_workers,
                                         batch_callback=mini_eval_reporter,
                                         checkpoint_metric_callback=checkpoint_metric,
                                         report_epoch_loss=False,
                                         checkpoint_phase='val',
                                         skip_batch_phases={ 'val' },
                                         patience=patience,
                                         start_epoch=start_epoch )

    # Final test evaluation runs once after the best checkpoint is frozen from validation metrics.
    best_model = initialize_scout_model( model_file=output_file_name, map_location='cpu' )
    metrics, final_eval_batch_size = evaluate_scout( best_model,
                                                     test_files,
                                                     scalar_stats,
                                                     eval_batch_size,
                                                     num_workers,
                                                     device=eval_device,
                                                     label='Final test',
                                                     expected_rows=test_scan[ 'rows_tokenized' ],
                                                     progress_tick_rows=progress_tick_rows,
                                                     return_effective_batch_size=True )

    metadata_path = output_file_name + '.metadata.json'
    write_metadata( metadata_path,
                    output_file_name,
                    dataset_root,
                    train_files,
                    val_files,
                    test_files,
                    scalar_stats,
                    train_scan,
                    val_scan,
                    test_scan,
                    metrics,
                    best_checkpoint_score,
                    checkpoint_metrics=checkpoint_metric.last_metrics )

    print( 'Best val checkpoint score: ' + format( float(best_checkpoint_score), '.6f' ) )
    print( 'Final test eval batch size: ' + str( final_eval_batch_size ) )
    print( 'Test MS2 cosine: ' + format( metrics[ 'test_ms2_cosine' ], '.6f' ) +
           ' (n=' + str(metrics[ 'test_ms2_count' ]) + ')' )
    print( 'Test iRT MAE/RMSE: ' + format( metrics[ 'test_irt_mae' ], '.6f' ) +
           ' / ' + format( metrics[ 'test_irt_rmse' ], '.6f' ) +
           ' (n=' + str(metrics[ 'test_irt_count' ]) + ')' )
    print( 'Test CCS MAE/RMSE: ' + format( metrics[ 'test_ccs_mae' ], '.6f' ) +
           ' / ' + format( metrics[ 'test_ccs_rmse' ], '.6f' ) +
           ' (n=' + str(metrics[ 'test_ccs_count' ]) + ')' )
    print( 'Wrote metadata: ' + metadata_path )
    return { 'best_test_loss' : float( best_checkpoint_score ),
             'best_val_checkpoint_score' : float( best_checkpoint_score ),
             'metadata_path' : metadata_path,
             'metrics' : metrics, }


def main():
    args = parse_args( sys.argv[1:] )
    os.makedirs( args.output_dir, exist_ok=True )

    model_out_file = os.path.join( args.output_dir, args.output_file )
    if args.model_file is not None and os.path.abspath( model_out_file ) == os.path.abspath( args.model_file ):
        print( 'Error: --output_file resolves to the same path as --model_file. '
               'Use a different output filename to avoid overwriting the source model.' )
        sys.exit( 1 )

    train_scout( args.dataset_root,
                 model_out_file,
                 device=args.device,
                 num_workers=args.num_workers,
                 patience=args.patience,
                 model_file=args.model_file,
                 start_epoch=args.start_epoch,
                 n_epochs=args.n_epochs,
                 eval_batch_size=args.eval_batch_size )


if __name__ == '__main__':
    main()
    sys.exit()
