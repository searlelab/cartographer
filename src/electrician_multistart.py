import os, sys, argparse
from datetime import datetime

from electrician_trainer import train_electrician


def parse_args( args ):
    src_dir = os.path.dirname( os.path.abspath( __file__ ) )
    timestamp = datetime.now().strftime( '%Y%m%d%H%M%S' )
    default_out_filename = 'Electrician_' + timestamp + '.pt'
    parser = argparse.ArgumentParser(
        description='Train Electrician multiple times and keep the best model' )
    parser.add_argument('--dataset_root',
                        type=str,
                        required=True,
                        help='Path to prospect-ptms-charge dataset directory')
    parser.add_argument('--output_file',
                        type=str,
                        help='Model filename',
                        default=default_out_filename)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to save model (default is Electrician/models)',
                        default=os.path.join( src_dir, '..', 'models' ))
    parser.add_argument('--device',
                        type=str,
                        help='Device for training {auto, mps, cuda, cpu}',
                        default='auto')
    parser.add_argument('--num_workers',
                        type=int,
                        help='DataLoader workers (default 0)',
                        default=0)
    parser.add_argument('--prtc_report',
                        type=str,
                        help='TSV file to log PRTC peptide charge predictions each epoch',
                        default=None)
    parser.add_argument('--patience',
                        type=int,
                        help='Early stopping: exit if no improvement for this many epochs',
                        default=None)
    parser.add_argument('--n_starts',
                        type=int,
                        help='Number of independent training runs (default 10)',
                        default=10)
    return parser.parse_args( args )


def main():
    args = parse_args( sys.argv[1:] )
    os.makedirs( args.output_dir, exist_ok=True )

    final_out = os.path.join( args.output_dir, args.output_file )
    n_starts = args.n_starts

    best_loss = None
    best_file = None

    # Temp file paths for each run
    run_files = []
    for run in range( 1, n_starts + 1 ):
        base, ext = os.path.splitext( args.output_file )
        run_file = os.path.join( args.output_dir, base + '_run' + str(run) + ext )
        run_files.append( run_file )

    for run in range( 1, n_starts + 1 ):
        run_file = run_files[ run - 1 ]
        print( '\n' + '=' * 60 )
        print( 'MULTI-START RUN ' + str(run) + ' of ' + str(n_starts) )
        print( '=' * 60 + '\n' )

        run_loss = train_electrician( args.dataset_root,
                                      run_file,
                                      device=args.device,
                                      num_workers=args.num_workers,
                                      prtc_report=args.prtc_report,
                                      patience=args.patience, )

        print( '\nRun ' + str(run) + ' best test loss: ' + format( run_loss, '.6f' ) )

        if best_loss is None or run_loss < best_loss:
            best_loss = run_loss
            best_file = run_file
            print( '>>> New best across all runs!' )
        else:
            print( 'Global best remains ' + format( best_loss, '.6f' ) )

    # Keep only the best model
    print( '\n' + '=' * 60 )
    print( 'MULTI-START COMPLETE' )
    print( 'Best test loss: ' + format( best_loss, '.6f' ) )
    print( '=' * 60 )

    os.rename( best_file, final_out )
    print( 'Best model saved to ' + final_out )

    # Clean up non-best run files
    for run_file in run_files:
        if os.path.isfile( run_file ):
            os.remove( run_file )
            print( 'Removed ' + run_file )

    print( 'Done' )


if __name__ == "__main__":
    main()
    sys.exit()
