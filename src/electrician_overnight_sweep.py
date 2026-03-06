import argparse
import csv
import datetime
import json
import os
import random
import shutil
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from electrician_model import initialize_electrician_model
from electrician_settings import training_parameters, progress_tick_rows
from loss_functions import ChargeDistribution_CrossEntropy
from prospect_loader import ProspectChargeDataset, discover_split_files
from training_loop import train_model


DEFAULT_DATASET_ROOT = '/Users/searle.brian/Documents/huggingface/data/prospect-ptms-charge'

DEFAULT_DESIGNS = [
    {
        'name' : 'full_b3_e64_k7_d123',
        'arch' : { 'embed_dim' : 64, 'n_blocks' : 3, 'kernel' : 7,
                   'dilation_schedule' : [ 1, 2, 3 ], 'block_variant' : 'full',
                   'bottleneck_ratio' : 0.5, },
    },
    {
        'name' : 'full_b3_e48_k5_d123',
        'arch' : { 'embed_dim' : 48, 'n_blocks' : 3, 'kernel' : 5,
                   'dilation_schedule' : [ 1, 2, 3 ], 'block_variant' : 'full',
                   'bottleneck_ratio' : 0.5, },
    },
    {
        'name' : 'full_b3_e32_k5_d123',
        'arch' : { 'embed_dim' : 32, 'n_blocks' : 3, 'kernel' : 5,
                   'dilation_schedule' : [ 1, 2, 3 ], 'block_variant' : 'full',
                   'bottleneck_ratio' : 0.5, },
    },
    {
        'name' : 'full_b3_e24_k5_d123',
        'arch' : { 'embed_dim' : 24, 'n_blocks' : 3, 'kernel' : 5,
                   'dilation_schedule' : [ 1, 2, 3 ], 'block_variant' : 'full',
                   'bottleneck_ratio' : 0.5, },
    },
    {
        'name' : 'full_b2_e32_k5_d12',
        'arch' : { 'embed_dim' : 32, 'n_blocks' : 2, 'kernel' : 5,
                   'dilation_schedule' : [ 1, 2 ], 'block_variant' : 'full',
                   'bottleneck_ratio' : 0.5, },
    },
    {
        'name' : 'full_b2_e24_k5_d12',
        'arch' : { 'embed_dim' : 24, 'n_blocks' : 2, 'kernel' : 5,
                   'dilation_schedule' : [ 1, 2 ], 'block_variant' : 'full',
                   'bottleneck_ratio' : 0.5, },
    },
    {
        'name' : 'full_b2_e32_k3_d12',
        'arch' : { 'embed_dim' : 32, 'n_blocks' : 2, 'kernel' : 3,
                   'dilation_schedule' : [ 1, 2 ], 'block_variant' : 'full',
                   'bottleneck_ratio' : 0.5, },
    },
    {
        'name' : 'konly_b2_e32_k5_d12',
        'arch' : { 'embed_dim' : 32, 'n_blocks' : 2, 'kernel' : 5,
                   'dilation_schedule' : [ 1, 2 ], 'block_variant' : 'k_only',
                   'bottleneck_ratio' : 0.5, },
    },
    {
        'name' : 'konly_b2_e32_k5_d18',
        'arch' : { 'embed_dim' : 32, 'n_blocks' : 2, 'kernel' : 5,
                   'dilation_schedule' : [ 1, 8 ], 'block_variant' : 'k_only',
                   'bottleneck_ratio' : 0.5, },
    },
    {
        'name' : 'bneck05_b2_e32_k5_d12',
        'arch' : { 'embed_dim' : 32, 'n_blocks' : 2, 'kernel' : 5,
                   'dilation_schedule' : [ 1, 2 ], 'block_variant' : 'bottleneck',
                   'bottleneck_ratio' : 0.5, },
    },
]


class logger( object ):
    def __init__( self, log_file ):
        self.log_file = log_file

    def log( self, text ):
        timestamp = datetime.datetime.now().strftime( '%Y-%m-%d %H:%M:%S' )
        line = '[' + timestamp + '] ' + text
        print( line )
        with open( self.log_file, 'a' ) as f:
            f.write( line + '\n' )


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Overnight Electrician architecture sweep (GPU train / CPU test)' )
    parser.add_argument( '--dataset_root',
                         type=str,
                         default=DEFAULT_DATASET_ROOT,
                         help='Path to prospect-ptms-charge root directory' )
    parser.add_argument( '--device',
                         type=str,
                         default='mps',
                         help='Training device (default mps)' )
    parser.add_argument( '--n_epochs',
                         type=int,
                         default=15,
                         help='Epochs per training run' )
    parser.add_argument( '--n_replicates',
                         type=int,
                         default=5,
                         help='Replicates per architecture design' )
    parser.add_argument( '--output_dir',
                         type=str,
                         default=None,
                         help='Run output directory (default models/electrician_sweeps/<timestamp>)' )
    parser.add_argument( '--num_workers_train',
                         type=int,
                         default=0,
                         help='DataLoader workers during training' )
    parser.add_argument( '--num_workers_bench',
                         type=int,
                         default=0,
                         help='DataLoader workers during CPU benchmarking/evaluation' )
    parser.add_argument( '--cpu_batch_size',
                         type=int,
                         default=1024,
                         help='CPU evaluation/benchmark batch size' )
    parser.add_argument( '--cpu_num_threads',
                         type=int,
                         default=1,
                         help='torch CPU thread count during benchmarking/evaluation' )
    parser.add_argument( '--cpu_warmup_batches',
                         type=int,
                         default=3,
                         help='Warmup batch count before timed CPU loop' )
    parser.add_argument( '--designs_file',
                         type=str,
                         default=None,
                         help='Optional JSON file defining architecture designs' )
    parser.add_argument( '--max_designs',
                         type=int,
                         default=None,
                         help='Optional cap for number of designs (for smoke tests)' )
    parser.add_argument( '--seed',
                         type=int,
                         default=1337,
                         help='Base random seed' )
    parser.add_argument( '--dry_run',
                         action='store_true',
                         help='Print planned jobs and exit without training' )
    return parser.parse_args( args )


def sanitize_name( name ):
    safe = []
    for ch in name:
        if ch.isalnum() or ch in [ '-', '_' ]:
            safe.append( ch )
        else:
            safe.append( '_' )
    return ''.join( safe )


def set_all_seeds( seed ):
    random.seed( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )


def normalize_arch( arch ):
    alias = { 'kernel_size' : 'kernel' }
    normalized = {}
    for key, value in arch.items():
        normalized[ alias.get( key, key ) ] = value

    if 'dilation_schedule' in normalized and normalized[ 'dilation_schedule' ] is not None:
        normalized[ 'dilation_schedule' ] = [ int(d) for d in normalized[ 'dilation_schedule' ] ]
        if len( normalized[ 'dilation_schedule' ] ) == 0:
            raise ValueError( 'dilation_schedule must be non-empty' )
        if min( normalized[ 'dilation_schedule' ] ) <= 0:
            raise ValueError( 'dilation_schedule values must be >= 1' )
        if 'n_blocks' not in normalized:
            normalized[ 'n_blocks' ] = len( normalized[ 'dilation_schedule' ] )

    if 'block_variant' not in normalized:
        normalized[ 'block_variant' ] = 'full'
    if 'bottleneck_ratio' not in normalized:
        normalized[ 'bottleneck_ratio' ] = 0.5

    return normalized


def load_designs( designs_file ):
    if designs_file is None:
        return list( DEFAULT_DESIGNS )

    with open( designs_file, 'r' ) as f:
        payload = json.load( f )

    if isinstance( payload, dict ) and 'designs' in payload:
        payload = payload[ 'designs' ]

    if not isinstance( payload, list ):
        raise ValueError( 'designs_file must contain a list of design objects' )

    designs = []
    for i, design in enumerate( payload ):
        if not isinstance( design, dict ):
            raise ValueError( 'Design entry #' + str(i) + ' must be an object' )

        name = design.get( 'name', 'design_' + str(i+1) )

        if 'arch' in design:
            arch = design[ 'arch' ]
        else:
            arch = dict( design )
            arch.pop( 'name', None )

        designs.append( { 'name' : name,
                          'arch' : normalize_arch( arch ), } )

    return designs


def write_csv( rows, path, fieldnames ):
    with open( path, 'w', newline='' ) as f:
        writer = csv.DictWriter( f, fieldnames=fieldnames )
        writer.writeheader()
        for row in rows:
            writer.writerow( row )


def replicate_seed( base_seed, design_ix, replicate_ix ):
    return int( base_seed + design_ix * 1000 + replicate_ix )


def evaluate_and_benchmark_cpu( model, test_files, batch_size, num_workers, warmup_batches ):
    model.eval()

    warmup_loader = DataLoader( ProspectChargeDataset( test_files, shuffle_files=False ),
                                batch_size,
                                shuffle=False,
                                num_workers=num_workers, )
    warmups = []
    for i, batch in enumerate( warmup_loader ):
        warmups.append( batch )
        if i + 1 >= warmup_batches:
            break

    with torch.no_grad():
        for seq, _, _ in warmups:
            _ = model( seq )

    eval_loader = DataLoader( ProspectChargeDataset( test_files, shuffle_files=False ),
                              batch_size,
                              shuffle=False,
                              num_workers=num_workers, )

    total_samples = 0
    ce_sum = 0.0
    top1_correct = 0
    mae_charge_sum = 0.0
    eps = 1e-8
    charges = None

    model_forward_seconds = 0.0
    end_to_end_start = time.perf_counter()

    with torch.no_grad():
        for seq, true_dist, _ in eval_loader:
            fwd_start = time.perf_counter()
            pred = model( seq )
            model_forward_seconds += time.perf_counter() - fwd_start

            if charges is None:
                charges = torch.arange( 1,
                                        pred.shape[1] + 1,
                                        dtype=pred.dtype,
                                        device=pred.device )

            ce_batch = -torch.sum( true_dist * torch.log( pred.clamp( min=eps ) ), dim=1 )
            ce_sum += float( torch.sum( ce_batch ).item() )

            pred_top1 = torch.argmax( pred, dim=1 )
            true_top1 = torch.argmax( true_dist, dim=1 )
            top1_correct += int( torch.sum( pred_top1 == true_top1 ).item() )

            pred_expected_charge = torch.sum( pred * charges, dim=1 )
            true_expected_charge = torch.sum( true_dist * charges, dim=1 )
            mae_charge_sum += float( torch.sum( torch.abs( pred_expected_charge - true_expected_charge ) ).item() )

            total_samples += seq.shape[0]

    end_to_end_seconds = time.perf_counter() - end_to_end_start

    metrics = { 'n_samples' : total_samples,
                'test_ce' : ce_sum / max( 1, total_samples ),
                'top1_accuracy' : float(top1_correct) / max( 1, total_samples ),
                'expected_charge_mae' : mae_charge_sum / max( 1, total_samples ),
                'model_forward_ms_total' : model_forward_seconds * 1000.0,
                'end_to_end_ms_total' : end_to_end_seconds * 1000.0,
                'peptides_per_sec_forward' : float(total_samples) / max( 1e-9, model_forward_seconds ),
                'peptides_per_sec_end_to_end' : float(total_samples) / max( 1e-9, end_to_end_seconds ), }
    return metrics


def write_leaderboard( design_winners, path ):
    rows = [ r for r in design_winners if r.get( 'status' ) == 'ok' ]

    by_ce = sorted( rows, key=lambda r: float( r[ 'best_test_ce' ] ) )
    by_speed = sorted( rows, key=lambda r: -float( r.get( 'peptides_per_sec_forward', 0.0 ) ) )

    with open( path, 'w' ) as f:
        f.write( '# Electrician Sweep Leaderboard\n\n' )

        f.write( '## By Test CE\n\n' )
        f.write( '| rank | design | best_test_ce | top1_accuracy | expected_charge_mae | forward_pep_per_sec | end2end_pep_per_sec | checkpoint |\n' )
        f.write( '| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |\n' )
        for i, row in enumerate( by_ce ):
            f.write( '| ' + str(i+1) +
                     ' | ' + row[ 'design_name' ] +
                     ' | ' + format( float(row[ 'best_test_ce' ]), '.6f' ) +
                     ' | ' + format( float(row.get( 'top1_accuracy', 0.0 )), '.4f' ) +
                     ' | ' + format( float(row.get( 'expected_charge_mae', 0.0 )), '.4f' ) +
                     ' | ' + format( float(row.get( 'peptides_per_sec_forward', 0.0 )), '.1f' ) +
                     ' | ' + format( float(row.get( 'peptides_per_sec_end_to_end', 0.0 )), '.1f' ) +
                     ' | ' + row.get( 'winner_checkpoint', '' ) +
                     ' |\n' )

        f.write( '\n## By Forward Throughput (CPU)\n\n' )
        f.write( '| rank | design | forward_pep_per_sec | best_test_ce | top1_accuracy | expected_charge_mae |\n' )
        f.write( '| ---: | --- | ---: | ---: | ---: | ---: |\n' )
        for i, row in enumerate( by_speed ):
            f.write( '| ' + str(i+1) +
                     ' | ' + row[ 'design_name' ] +
                     ' | ' + format( float(row.get( 'peptides_per_sec_forward', 0.0 )), '.1f' ) +
                     ' | ' + format( float(row[ 'best_test_ce' ]), '.6f' ) +
                     ' | ' + format( float(row.get( 'top1_accuracy', 0.0 )), '.4f' ) +
                     ' | ' + format( float(row.get( 'expected_charge_mae', 0.0 )), '.4f' ) +
                     ' |\n' )


def validate_training_device( device ):
    if device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError( 'Requested --device mps, but torch.backends.mps.is_available() is False. '
                                'Use --device auto/cpu/cuda or run on a supported macOS + PyTorch build.' )
    elif device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError( 'Requested --device cuda, but torch.cuda.is_available() is False. '
                                'Use --device auto/cpu/mps or run on a CUDA-capable setup.' )


def main():
    args = parse_args( os.sys.argv[1:] )

    timestamp = datetime.datetime.now().strftime( '%Y%m%d_%H%M%S' )
    if args.output_dir is None:
        run_dir = os.path.join( 'models', 'electrician_sweeps', timestamp )
    else:
        run_dir = args.output_dir

    os.makedirs( run_dir, exist_ok=True )
    checkpoints_dir = os.path.join( run_dir, 'checkpoints' )
    winners_dir = os.path.join( run_dir, 'winners' )
    os.makedirs( checkpoints_dir, exist_ok=True )
    os.makedirs( winners_dir, exist_ok=True )

    log = logger( os.path.join( run_dir, 'stdout.log' ) )

    try:
        torch.set_num_threads( int(args.cpu_num_threads) )
        try:
            torch.set_num_interop_threads( max( 1, int(args.cpu_num_threads) ) )
        except RuntimeError:
            pass
    except Exception:
        pass

    train_files = discover_split_files( args.dataset_root, 'train' )
    test_files = discover_split_files( args.dataset_root, 'test' )
    if len( train_files ) == 0:
        raise RuntimeError( 'No train parquet shards found under ' + args.dataset_root )
    if len( test_files ) == 0:
        raise RuntimeError( 'No test parquet shards found under ' + args.dataset_root )

    designs = load_designs( args.designs_file )
    if args.max_designs is not None:
        designs = designs[ :args.max_designs ]

    log.log( 'Training device: ' + args.device + ' | Epoch test-phase device: cpu | Final benchmark device: cpu' )
    log.log( 'Dataset root: ' + args.dataset_root )
    log.log( 'Train shards: ' + str(len(train_files)) + ' | Test shards: ' + str(len(test_files)) )
    log.log( 'Design count: ' + str(len(designs)) + ' | Replicates: ' + str(args.n_replicates) +
             ' | Epochs: ' + str(args.n_epochs) )

    run_config = { 'timestamp' : timestamp,
                   'args' : vars( args ),
                   'train_shards' : train_files,
                   'test_shards' : test_files,
                   'designs' : designs, }
    with open( os.path.join( run_dir, 'run_config.json' ), 'w' ) as f:
        json.dump( run_config, f, indent=2 )

    if args.dry_run:
        for d in designs:
            log.log( '[dry-run] ' + d[ 'name' ] + ' arch=' + json.dumps( d[ 'arch' ], sort_keys=True ) )
        log.log( 'Dry run complete. No training executed.' )
        return

    validate_training_device( args.device )

    replicate_rows = []

    for rep in range( 1, args.n_replicates + 1 ):
        log.log( 'Replicate pass ' + str(rep) + '/' + str(args.n_replicates) )
        for design_ix, design in enumerate( designs ):
            design_name = sanitize_name( design[ 'name' ] )
            arch = normalize_arch( design[ 'arch' ] )
            rep_seed = replicate_seed( args.seed, design_ix, rep )
            checkpoint = os.path.join( checkpoints_dir,
                                       design_name + '__rep' + str(rep).zfill(2) + '.pt' )

            row = { 'design_name' : design_name,
                    'design_index' : design_ix,
                    'replicate' : rep,
                    'seed' : rep_seed,
                    'status' : 'ok',
                    'error' : '',
                    'best_test_ce' : '',
                    'train_seconds' : '',
                    'checkpoint' : checkpoint,
                    'arch_json' : json.dumps( arch, sort_keys=True ), }

            log.log( '  design ' + str(design_ix+1) + '/' + str(len(designs)) + ': ' + design_name +
                     ' seed=' + str(rep_seed) )

            try:
                set_all_seeds( rep_seed )

                datasets = { 'train' : ProspectChargeDataset( train_files, shuffle_files=True ),
                             'test' : ProspectChargeDataset( test_files, shuffle_files=False ), }

                model = initialize_electrician_model( arch_overrides=arch )
                loss_fx = ChargeDistribution_CrossEntropy()
                optimizer = training_parameters[ 'optimizer' ]( list( model.parameters() ),
                                                                lr=training_parameters[ 'learning_rate' ], )

                train_start = time.time()
                best_test_ce = train_model( model,
                                            datasets,
                                            training_parameters[ 'initial_batch_size' ],
                                            training_parameters[ 'max_batch_size' ],
                                            training_parameters[ 'epochs_to_2x_batch' ],
                                            loss_fx,
                                            optimizer,
                                            args.n_epochs,
                                            args.device,
                                            'cpu',
                                            checkpoint,
                                            progress_tick_rows=progress_tick_rows,
                                            num_workers=args.num_workers_train,
                                            start_epoch=1, )
                train_seconds = time.time() - train_start

                row[ 'best_test_ce' ] = format( float(best_test_ce), '.8f' )
                row[ 'train_seconds' ] = format( train_seconds, '.3f' )
                log.log( '    best_test_ce=' + row[ 'best_test_ce' ] +
                         ' train_seconds=' + row[ 'train_seconds' ] )
            except Exception as e:
                row[ 'status' ] = 'failed'
                row[ 'error' ] = str(e)
                log.log( '    FAILED: ' + str(e) )
                log.log( traceback.format_exc() )

            replicate_rows.append( row )

            write_csv( replicate_rows,
                       os.path.join( run_dir, 'replicates.csv' ),
                       [ 'design_name', 'design_index', 'replicate', 'seed', 'status', 'error',
                         'best_test_ce', 'train_seconds', 'checkpoint', 'arch_json' ] )

    design_winners = []
    winner_source_paths = set()

    for design_ix, design in enumerate( designs ):
        design_name = sanitize_name( design[ 'name' ] )
        arch = normalize_arch( design[ 'arch' ] )
        valid_rows = [ r for r in replicate_rows
                       if r[ 'design_name' ] == design_name and
                          r[ 'status' ] == 'ok' and
                          r[ 'best_test_ce' ] != '' and
                          os.path.isfile( r[ 'checkpoint' ] ) ]

        base = { 'design_name' : design_name,
                 'design_index' : design_ix,
                 'status' : 'ok',
                 'error' : '',
                 'winner_replicate' : '',
                 'winner_seed' : '',
                 'best_test_ce' : '',
                 'winner_checkpoint' : '',
                 'arch_json' : json.dumps( arch, sort_keys=True ),
                 'test_ce' : '',
                 'top1_accuracy' : '',
                 'expected_charge_mae' : '',
                 'n_samples' : '',
                 'model_forward_ms_total' : '',
                 'end_to_end_ms_total' : '',
                 'peptides_per_sec_forward' : '',
                 'peptides_per_sec_end_to_end' : '', }

        if len( valid_rows ) == 0:
            base[ 'status' ] = 'failed'
            base[ 'error' ] = 'No successful replicate checkpoints for design'
            design_winners.append( base )
            log.log( 'No winner for design ' + design_name )
            continue

        winner = min( valid_rows, key=lambda r: float( r[ 'best_test_ce' ] ) )
        winner_source_paths.add( winner[ 'checkpoint' ] )

        winner_copy = os.path.join( winners_dir, design_name + '.pt' )
        shutil.copy2( winner[ 'checkpoint' ], winner_copy )

        base[ 'winner_replicate' ] = winner[ 'replicate' ]
        base[ 'winner_seed' ] = winner[ 'seed' ]
        base[ 'best_test_ce' ] = winner[ 'best_test_ce' ]
        base[ 'winner_checkpoint' ] = winner_copy

        log.log( 'CPU evaluation + benchmark for winner: ' + design_name )
        model = initialize_electrician_model( model_file=winner_copy,
                                              arch_overrides=arch,
                                              map_location='cpu', ).to( 'cpu' )

        metrics = evaluate_and_benchmark_cpu( model,
                                              test_files,
                                              args.cpu_batch_size,
                                              args.num_workers_bench,
                                              args.cpu_warmup_batches, )

        base[ 'test_ce' ] = format( metrics[ 'test_ce' ], '.8f' )
        base[ 'top1_accuracy' ] = format( metrics[ 'top1_accuracy' ], '.8f' )
        base[ 'expected_charge_mae' ] = format( metrics[ 'expected_charge_mae' ], '.8f' )
        base[ 'n_samples' ] = metrics[ 'n_samples' ]
        base[ 'model_forward_ms_total' ] = format( metrics[ 'model_forward_ms_total' ], '.3f' )
        base[ 'end_to_end_ms_total' ] = format( metrics[ 'end_to_end_ms_total' ], '.3f' )
        base[ 'peptides_per_sec_forward' ] = format( metrics[ 'peptides_per_sec_forward' ], '.3f' )
        base[ 'peptides_per_sec_end_to_end' ] = format( metrics[ 'peptides_per_sec_end_to_end' ], '.3f' )

        log.log( '  test_ce=' + base[ 'test_ce' ] +
                 ' top1=' + base[ 'top1_accuracy' ] +
                 ' mae=' + base[ 'expected_charge_mae' ] +
                 ' forward_pep_s=' + base[ 'peptides_per_sec_forward' ] )

        design_winners.append( base )

    ok_winners = [ r for r in design_winners if r[ 'status' ] == 'ok' and r[ 'best_test_ce' ] != '' ]
    if len( ok_winners ) > 0:
        global_best = min( ok_winners, key=lambda r: float( r[ 'best_test_ce' ] ) )
        global_best_src = global_best[ 'winner_checkpoint' ]
        global_best_dst = os.path.join( run_dir, 'global_best.pt' )
        shutil.copy2( global_best_src, global_best_dst )
        log.log( 'Global best: ' + global_best[ 'design_name' ] +
                 ' CE=' + global_best[ 'best_test_ce' ] +
                 ' -> ' + global_best_dst )
    else:
        log.log( 'No successful winners; global_best.pt not created' )

    for row in replicate_rows:
        path = row.get( 'checkpoint', '' )
        if path == '' or not os.path.isfile( path ):
            continue
        if path in winner_source_paths:
            continue
        os.remove( path )

    write_csv( replicate_rows,
               os.path.join( run_dir, 'replicates.csv' ),
               [ 'design_name', 'design_index', 'replicate', 'seed', 'status', 'error',
                 'best_test_ce', 'train_seconds', 'checkpoint', 'arch_json' ] )

    write_csv( design_winners,
               os.path.join( run_dir, 'design_winners.csv' ),
               [ 'design_name', 'design_index', 'status', 'error',
                 'winner_replicate', 'winner_seed', 'best_test_ce', 'winner_checkpoint', 'arch_json',
                 'test_ce', 'top1_accuracy', 'expected_charge_mae', 'n_samples',
                 'model_forward_ms_total', 'end_to_end_ms_total',
                 'peptides_per_sec_forward', 'peptides_per_sec_end_to_end' ] )

    write_leaderboard( design_winners,
                       os.path.join( run_dir, 'leaderboard.md' ) )

    log.log( 'Sweep complete. Outputs written to ' + run_dir )


if __name__ == '__main__':
    main()
