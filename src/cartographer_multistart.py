import argparse
import os
import sys
from datetime import datetime

from cartographer_trainer import train_cartographer


def parse_args( args ):
    src_dir = os.path.dirname( os.path.abspath( __file__ ) )
    timestamp = datetime.now().strftime( '%Y%m%d%H%M%S' )
    default_out_filename = 'Cartographer_' + timestamp + '.pt'

    parser = argparse.ArgumentParser(
        description='Train Cartographer multiple times and keep the best model'
    )
    parser.add_argument( '--dataset_root',
                         type=str,
                         required=True,
                         help='Path to prospect-ptms-ms2 dataset directory with data/train-*.parquet, optional data/val-*.parquet, and data/test-*.parquet' )
    parser.add_argument( '--output_file',
                         type=str,
                         default=default_out_filename,
                         help='Model filename' )
    parser.add_argument( '--output_dir',
                         type=str,
                         default=os.path.join( src_dir, '..', 'models' ),
                         help='Directory to save model (default is Cartographer/models)' )
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
                         help='Early stopping: exit if no improvement for this many epochs' )
    parser.add_argument( '--n_epochs',
                         type=int,
                         default=None,
                         help='Total epoch count, overrides settings (default: use training_parameters)' )
    parser.add_argument( '--n_starts',
                         type=int,
                         default=10,
                         help='Number of independent training runs (default 10)' )
    parser.add_argument( '--prtc_report',
                         type=str,
                         default=None,
                         help='Optional TSV path; writes one file per run with _runXX suffix' )
    return parser.parse_args( args )


def per_run_path( base_path, run_idx ):
    if base_path is None:
        return None
    root, ext = os.path.splitext( base_path )
    if ext == '':
        ext = '.tsv'
    return root + '_run' + str(run_idx).zfill(2) + ext


def main():
    args = parse_args( sys.argv[1:] )
    if args.n_starts < 1:
        raise ValueError( '--n_starts must be >= 1' )

    os.makedirs( args.output_dir, exist_ok=True )

    final_out = os.path.join( args.output_dir, args.output_file )

    base_name, ext = os.path.splitext( args.output_file )
    if ext == '':
        ext = '.pt'

    run_files = []
    for run in range( 1, args.n_starts + 1 ):
        run_file = os.path.join( args.output_dir,
                                 base_name + '_run' + str(run).zfill(2) + ext )
        run_files.append( run_file )

    best_loss = None
    best_file = None

    for run in range( 1, args.n_starts + 1 ):
        run_file = run_files[ run - 1 ]
        run_prtc = per_run_path( args.prtc_report, run )

        print( '\n' + '=' * 60 )
        print( 'MULTI-START RUN ' + str(run) + ' of ' + str(args.n_starts) )
        print( '=' * 60 + '\n' )

        run_loss = train_cartographer( args.dataset_root,
                                       run_file,
                                       device=args.device,
                                       num_workers=args.num_workers,
                                       prtc_report=run_prtc,
                                       patience=args.patience,
                                       model_file=None,
                                       start_epoch=1,
                                       n_epochs=args.n_epochs )

        print( '\nRun ' + str(run) + ' best test loss: ' + format( run_loss, '.6f' ) )

        if best_loss is None or run_loss < best_loss:
            best_loss = run_loss
            best_file = run_file
            print( '>>> New best across all runs!' )
        else:
            print( 'Global best remains ' + format( best_loss, '.6f' ) )

    print( '\n' + '=' * 60 )
    print( 'MULTI-START COMPLETE' )
    print( 'Best test loss: ' + format( best_loss, '.6f' ) )
    print( '=' * 60 )

    os.rename( best_file, final_out )
    print( 'Best model saved to ' + final_out )

    for run_file in run_files:
        if os.path.isfile( run_file ):
            os.remove( run_file )
            print( 'Removed ' + run_file )

    print( 'Done' )


if __name__ == '__main__':
    main()
    sys.exit()
