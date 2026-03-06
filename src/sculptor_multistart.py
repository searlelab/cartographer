import argparse
import csv
import datetime
import json
import os
import shutil
import sys
import traceback

from sculptor_trainer import train_sculptor


def default_designs():
    designs = []
    for embed_dim in [ 24, 32, 48, 64 ]:
        for kernel in [ 5, 7 ]:
            for dilation_schedule in [ [ 1, 2, 3 ], [ 1, 4, 8 ] ]:
                d_label = ''.join( [ str(d) for d in dilation_schedule ] )
                name = 'full_b3_e' + str(embed_dim) + '_k' + str(kernel) + '_d' + d_label
                arch = { 'embed_dim' : embed_dim,
                         'n_blocks' : 3,
                         'kernel' : kernel,
                         'dilation_schedule' : dilation_schedule,
                         'block_variant' : 'full',
                         'bottleneck_ratio' : 0.5, }
                designs.append( { 'name' : name, 'arch' : arch } )
    return designs


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
        return default_designs()

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

        name = design.get( 'name', 'design_' + str(i + 1) )

        if 'arch' in design:
            arch = design[ 'arch' ]
        else:
            arch = dict( design )
            arch.pop( 'name', None )

        designs.append( { 'name' : name, 'arch' : normalize_arch( arch ) } )

    return designs


def sanitize_name( name ):
    safe = []
    for ch in name:
        if ch.isalnum() or ch in [ '-', '_' ]:
            safe.append( ch )
        else:
            safe.append( '_' )
    return ''.join( safe )


def write_csv( rows, path, fieldnames ):
    with open( path, 'w', newline='' ) as f:
        writer = csv.DictWriter( f, fieldnames=fieldnames )
        writer.writeheader()
        for row in rows:
            writer.writerow( row )


def parse_args( args ):
    timestamp = datetime.datetime.now().strftime( '%Y%m%d%H%M%S' )
    default_out_filename = 'Sculptor_' + timestamp + '.pt'

    parser = argparse.ArgumentParser( description='Sculptor multistart architecture sweep' )
    parser.add_argument( '--dataset_root',
                         type=str,
                         required=True,
                         help='Path to Sculptor dataset root directory' )
    parser.add_argument( '--output_file',
                         type=str,
                         default=default_out_filename,
                         help='Output filename for global best model' )
    parser.add_argument( '--output_dir',
                         type=str,
                         default=None,
                         help='Run output directory (default models/sculptor_sweeps/<timestamp>)' )
    parser.add_argument( '--device',
                         type=str,
                         default='auto',
                         help='Training device {auto, mps, cuda, cpu}' )
    parser.add_argument( '--num_workers',
                         type=int,
                         default=0,
                         help='DataLoader workers' )
    parser.add_argument( '--patience',
                         type=int,
                         default=None,
                         help='Early stopping patience' )
    parser.add_argument( '--n_epochs',
                         type=int,
                         default=None,
                         help='Epoch count override for each run' )
    parser.add_argument( '--n_starts',
                         type=int,
                         default=3,
                         help='Independent starts per architecture design' )
    parser.add_argument( '--designs_file',
                         type=str,
                         default=None,
                         help='Optional JSON file defining architecture designs' )
    parser.add_argument( '--max_designs',
                         type=int,
                         default=None,
                         help='Optional cap on design count (for smoke tests)' )
    parser.add_argument( '--metadata_file',
                         type=str,
                         default=None,
                         help='Optional metadata JSON path override' )
    parser.add_argument( '--eval_batch_size',
                         type=int,
                         default=4096,
                         help='Eval batch size for MAE/RMSE summary' )
    parser.add_argument( '--dry_run',
                         action='store_true',
                         help='Print planned runs and exit without training' )
    return parser.parse_args( args )


def write_leaderboard( winners, path ):
    rows = [ r for r in winners if r.get( 'status' ) == 'ok' ]
    rows = sorted( rows, key=lambda r: float( r[ 'best_test_loss' ] ) )

    with open( path, 'w' ) as f:
        f.write( '# Sculptor Leaderboard\n\n' )
        if len( rows ) == 0:
            f.write( 'No successful designs.\n' )
            return

        f.write( '| rank | design | best_test_loss | test_mae_ccs | test_rmse_ccs | winner_checkpoint |\n' )
        f.write( '| ---: | --- | ---: | ---: | ---: | --- |\n' )

        for i, row in enumerate( rows ):
            f.write( '| ' + str(i + 1) +
                     ' | ' + row[ 'design_name' ] +
                     ' | ' + format( float(row[ 'best_test_loss' ]), '.6f' ) +
                     ' | ' + format( float(row[ 'test_mae_ccs' ]), '.6f' ) +
                     ' | ' + format( float(row[ 'test_rmse_ccs' ]), '.6f' ) +
                     ' | ' + row[ 'winner_checkpoint' ] +
                     ' |\n' )


def main():
    args = parse_args( sys.argv[1:] )

    timestamp = datetime.datetime.now().strftime( '%Y%m%d_%H%M%S' )
    if args.output_dir is None:
        run_dir = os.path.join( 'models', 'sculptor_sweeps', timestamp )
    else:
        run_dir = args.output_dir

    os.makedirs( run_dir, exist_ok=True )

    checkpoints_dir = os.path.join( run_dir, 'checkpoints' )
    winners_dir = os.path.join( run_dir, 'winners' )
    os.makedirs( checkpoints_dir, exist_ok=True )
    os.makedirs( winners_dir, exist_ok=True )

    designs = load_designs( args.designs_file )
    if args.max_designs is not None:
        designs = designs[ : args.max_designs ]

    print( 'Sculptor multistart' )
    print( 'Dataset root: ' + args.dataset_root )
    print( 'Design count: ' + str(len(designs)) )
    print( 'Starts per design: ' + str(args.n_starts) )
    print( 'Run directory: ' + run_dir )

    if args.dry_run:
        for d in designs:
            print( '[dry-run] ' + d[ 'name' ] + ' arch=' + json.dumps( d[ 'arch' ], sort_keys=True ) )
        print( 'Dry run complete' )
        return

    replicate_rows = []
    winner_rows = []
    winner_source_paths = set()

    for design_ix, design in enumerate( designs ):
        design_name = sanitize_name( design[ 'name' ] )
        arch = normalize_arch( design[ 'arch' ] )

        print( '\n' + '=' * 72 )
        print( 'Design ' + str(design_ix + 1) + ' of ' + str(len(designs)) + ': ' + design_name )
        print( 'Architecture: ' + json.dumps( arch, sort_keys=True ) )
        print( '=' * 72 )

        best_row = None

        for run in range( 1, args.n_starts + 1 ):
            run_ckpt = os.path.join( checkpoints_dir,
                                     design_name + '__run' + str(run).zfill(2) + '.pt' )

            row = { 'design_name' : design_name,
                    'design_index' : design_ix,
                    'run' : run,
                    'status' : 'ok',
                    'error' : '',
                    'best_test_loss' : '',
                    'test_mae_ccs' : '',
                    'test_rmse_ccs' : '',
                    'n_test' : '',
                    'checkpoint' : run_ckpt,
                    'arch_json' : json.dumps( arch, sort_keys=True ), }

            print( 'Run ' + str(run) + ' of ' + str(args.n_starts) + ' for ' + design_name )

            try:
                metrics = train_sculptor( args.dataset_root,
                                          run_ckpt,
                                          device=args.device,
                                          num_workers=args.num_workers,
                                          patience=args.patience,
                                          model_file=None,
                                          start_epoch=1,
                                          n_epochs=args.n_epochs,
                                          arch_overrides=arch,
                                          metadata_file=args.metadata_file,
                                          eval_batch_size=args.eval_batch_size )

                row[ 'best_test_loss' ] = format( float(metrics[ 'best_test_loss' ]), '.8f' )
                row[ 'test_mae_ccs' ] = format( float(metrics[ 'test_mae' ]), '.8f' )
                row[ 'test_rmse_ccs' ] = format( float(metrics[ 'test_rmse' ]), '.8f' )
                row[ 'n_test' ] = int( metrics[ 'n_test' ] )

                if best_row is None or float( row[ 'best_test_loss' ] ) < float( best_row[ 'best_test_loss' ] ):
                    best_row = dict( row )
            except Exception as exc:
                row[ 'status' ] = 'failed'
                row[ 'error' ] = str( exc )
                print( 'FAILED: ' + str(exc) )
                print( traceback.format_exc() )

            replicate_rows.append( row )

        if best_row is None:
            winner_rows.append( { 'design_name' : design_name,
                                  'design_index' : design_ix,
                                  'status' : 'failed',
                                  'error' : 'No successful runs for design',
                                  'winner_run' : '',
                                  'best_test_loss' : '',
                                  'test_mae_ccs' : '',
                                  'test_rmse_ccs' : '',
                                  'n_test' : '',
                                  'winner_checkpoint' : '',
                                  'arch_json' : json.dumps( arch, sort_keys=True ), } )
            continue

        winner_src = best_row[ 'checkpoint' ]
        winner_dst = os.path.join( winners_dir, design_name + '.pt' )
        shutil.copy2( winner_src, winner_dst )
        winner_source_paths.add( winner_src )

        winner_rows.append( { 'design_name' : design_name,
                              'design_index' : design_ix,
                              'status' : 'ok',
                              'error' : '',
                              'winner_run' : best_row[ 'run' ],
                              'best_test_loss' : best_row[ 'best_test_loss' ],
                              'test_mae_ccs' : best_row[ 'test_mae_ccs' ],
                              'test_rmse_ccs' : best_row[ 'test_rmse_ccs' ],
                              'n_test' : best_row[ 'n_test' ],
                              'winner_checkpoint' : winner_dst,
                              'arch_json' : json.dumps( arch, sort_keys=True ), } )

    ok_winners = [ row for row in winner_rows if row[ 'status' ] == 'ok' ]
    if len( ok_winners ) > 0:
        global_best = min( ok_winners, key=lambda row: float( row[ 'best_test_loss' ] ) )
        global_best_src = global_best[ 'winner_checkpoint' ]

        global_best_path = os.path.join( run_dir, 'global_best.pt' )
        shutil.copy2( global_best_src, global_best_path )

        final_out = os.path.join( run_dir, args.output_file )
        shutil.copy2( global_best_src, final_out )

        print( '\nGlobal best design: ' + global_best[ 'design_name' ] )
        print( 'Best normalized test loss: ' + global_best[ 'best_test_loss' ] )
        print( 'Global best checkpoint: ' + global_best_path )
        print( 'Final output model: ' + final_out )
    else:
        print( '\nNo successful design runs; no global best model written' )

    for row in replicate_rows:
        path = row.get( 'checkpoint', '' )
        if path == '' or not os.path.isfile( path ):
            continue
        if path in winner_source_paths:
            continue
        os.remove( path )

    replicate_csv = os.path.join( run_dir, 'replicates.csv' )
    winners_csv = os.path.join( run_dir, 'design_winners.csv' )
    leaderboard_md = os.path.join( run_dir, 'leaderboard.md' )

    write_csv( replicate_rows,
               replicate_csv,
               [ 'design_name', 'design_index', 'run', 'status', 'error',
                 'best_test_loss', 'test_mae_ccs', 'test_rmse_ccs', 'n_test',
                 'checkpoint', 'arch_json' ] )

    write_csv( winner_rows,
               winners_csv,
               [ 'design_name', 'design_index', 'status', 'error', 'winner_run',
                 'best_test_loss', 'test_mae_ccs', 'test_rmse_ccs', 'n_test',
                 'winner_checkpoint', 'arch_json' ] )

    write_leaderboard( winner_rows, leaderboard_md )

    print( '\nOutputs:' )
    print( '  ' + replicate_csv )
    print( '  ' + winners_csv )
    print( '  ' + leaderboard_md )


if __name__ == '__main__':
    main()
    sys.exit()
