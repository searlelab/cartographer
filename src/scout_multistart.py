import argparse
import os
import sys
from datetime import datetime

from scout_trainer import train_scout


def parse_args( args ):
    src_dir = os.path.dirname( os.path.abspath( __file__ ) )
    timestamp = datetime.now().strftime( '%Y%m%d%H%M%S' )
    default_out_filename = 'Scout_' + timestamp + '.pt'

    parser = argparse.ArgumentParser(
        description='Train Scout multiple times and keep the best model'
    )
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
    parser.add_argument( '--n_epochs',
                         type=int,
                         default=None,
                         help='Total epoch count, overrides settings' )
    parser.add_argument( '--eval_batch_size',
                         type=int,
                         default=4096,
                         help='Batch size used for denormalized evaluation metrics' )
    parser.add_argument( '--n_starts',
                         type=int,
                         default=10,
                         help='Number of independent training runs (default 10)' )
    return parser.parse_args( args )


def _metadata_path( model_path ):
    return model_path + '.metadata.json'


def main():
    args = parse_args( sys.argv[1:] )
    if args.n_starts < 1:
        raise ValueError( '--n_starts must be >= 1' )

    os.makedirs( args.output_dir, exist_ok=True )

    final_out = os.path.join( args.output_dir, args.output_file )
    final_metadata = _metadata_path( final_out )

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
    best_metadata = None

    for run in range( 1, args.n_starts + 1 ):
        run_file = run_files[ run - 1 ]

        print( '\n' + '=' * 60 )
        print( 'MULTI-START RUN ' + str(run) + ' of ' + str(args.n_starts) )
        print( '=' * 60 + '\n' )

        run_result = train_scout( args.dataset_root,
                                  run_file,
                                  device=args.device,
                                  num_workers=args.num_workers,
                                  patience=args.patience,
                                  model_file=None,
                                  start_epoch=1,
                                  n_epochs=args.n_epochs,
                                  eval_batch_size=args.eval_batch_size )

        run_loss = float( run_result[ 'best_test_loss' ] )
        run_metadata = run_result.get( 'metadata_path', _metadata_path( run_file ) )

        print( '\nRun ' + str(run) + ' best checkpoint score: ' + format( run_loss, '.6f' ) )

        if best_loss is None or run_loss < best_loss:
            best_loss = run_loss
            best_file = run_file
            best_metadata = run_metadata
            print( '>>> New best across all runs!' )
        else:
            print( 'Global best remains ' + format( best_loss, '.6f' ) )

    print( '\n' + '=' * 60 )
    print( 'MULTI-START COMPLETE' )
    print( 'Best checkpoint score: ' + format( best_loss, '.6f' ) )
    print( '=' * 60 )

    os.replace( best_file, final_out )
    print( 'Best model saved to ' + final_out )

    if best_metadata is not None and os.path.isfile( best_metadata ):
        os.replace( best_metadata, final_metadata )
        print( 'Best metadata saved to ' + final_metadata )

    for run_file in run_files:
        if os.path.isfile( run_file ):
            os.remove( run_file )
            print( 'Removed ' + run_file )

        run_metadata = _metadata_path( run_file )
        if os.path.isfile( run_metadata ):
            os.remove( run_metadata )
            print( 'Removed ' + run_metadata )

    print( 'Done' )


if __name__ == '__main__':
    main()
    sys.exit()
