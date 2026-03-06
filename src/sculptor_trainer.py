import argparse
import os
import sys
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch.utils.data import DataLoader

from sculptor_loader import SculptorCCSDataset, discover_split_files
from sculptor_loss import CCS_HuberLoss
from sculptor_model import initialize_sculptor_model
from sculptor_settings import metadata_filename, progress_tick_rows, training_parameters
from training_loop import train_model


def parse_args( args ):
    src_dir = os.path.dirname( os.path.abspath( __file__ ) )
    timestamp = datetime.now().strftime( '%Y%m%d%H%M%S' )
    default_out_filename = 'Sculptor_' + timestamp + '.pt'

    parser = argparse.ArgumentParser( description='Train Sculptor CCS regression model' )
    parser.add_argument( '--dataset_root',
                         type=str,
                         required=True,
                         help='Path to Sculptor dataset root containing data/train-*.parquet and metadata JSON' )
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
                         help='Early stopping: no improvement for this many epochs' )
    parser.add_argument( '--model_file',
                         type=str,
                         default=None,
                         help='Path to source checkpoint for resume/warm-start' )
    parser.add_argument( '--start_epoch',
                         type=int,
                         default=1,
                         help='Epoch number to begin at (default: 1)' )
    parser.add_argument( '--n_epochs',
                         type=int,
                         default=None,
                         help='Total epoch count, overrides settings' )
    parser.add_argument( '--metadata_file',
                         type=str,
                         default=None,
                         help='Path to Sculptor metadata JSON (default: dataset_root/' + metadata_filename + ')' )
    parser.add_argument( '--eval_batch_size',
                         type=int,
                         default=4096,
                         help='Batch size used for denormalized test MAE/RMSE reporting' )
    return parser.parse_args( args )


def load_metadata( dataset_root, metadata_file=None ):
    candidates = []
    if metadata_file is not None:
        candidates.append( metadata_file )
    candidates.append( os.path.join( dataset_root, metadata_filename ) )

    for path in candidates:
        if os.path.isfile( path ):
            import json
            with open( path, 'r' ) as f:
                return json.load( f ), path

    raise FileNotFoundError( 'Could not locate Sculptor metadata JSON. Tried: ' + ', '.join( candidates ) )


def evaluate_metrics( model, test_files, ccs_mean, ccs_std, batch_size, num_workers ):
    dataset = SculptorCCSDataset( test_files,
                                  ccs_mean,
                                  ccs_std,
                                  shuffle_files=False )
    loader = DataLoader( dataset,
                         batch_size,
                         shuffle=False,
                         num_workers=num_workers )

    model.eval()

    abs_err_sum = 0.0
    sq_err_sum = 0.0
    total = 0

    with torch.no_grad():
        for seq, charge, true_norm, _ in loader:
            pred_norm = model( seq, charge )

            pred_ccs = pred_norm * ccs_std + ccs_mean
            true_ccs = true_norm * ccs_std + ccs_mean

            diff = pred_ccs - true_ccs
            abs_err_sum += float( torch.sum( torch.abs( diff ) ).item() )
            sq_err_sum += float( torch.sum( diff * diff ).item() )
            total += int( diff.shape[0] )

    if total == 0:
        return { 'mae' : 0.0, 'rmse' : 0.0, 'n_test' : 0 }

    mae = abs_err_sum / total
    rmse = ( sq_err_sum / total ) ** 0.5
    return { 'mae' : mae, 'rmse' : rmse, 'n_test' : total }


def train_sculptor( dataset_root,
                    output_file_name,
                    device='auto',
                    num_workers=0,
                    patience=None,
                    model_file=None,
                    start_epoch=1,
                    n_epochs=None,
                    arch_overrides=None,
                    metadata_file=None,
                    eval_batch_size=4096, ):
    print( 'Sculptor training initiated' )

    metadata, metadata_path = load_metadata( dataset_root, metadata_file )
    ccs_mean = float( metadata[ 'train_ccs_mean' ] )
    ccs_std = float( metadata[ 'train_ccs_std' ] )
    if ccs_std <= 0.0:
        ccs_std = 1.0

    print( 'Using metadata: ' + metadata_path )
    print( 'CCS normalization: mean=' + format( ccs_mean, '.6f' ) +
           ', std=' + format( ccs_std, '.6f' ) )

    train_files = discover_split_files( dataset_root, 'train' )
    test_files = discover_split_files( dataset_root, 'test' )

    print( 'Found ' + str(len(train_files)) + ' train shards, ' +
           str(len(test_files)) + ' test shards' )

    assert len( train_files ) > 0, 'No train parquet files found in ' + dataset_root
    assert len( test_files ) > 0, 'No test parquet files found in ' + dataset_root

    datasets = { 'train' : SculptorCCSDataset( train_files,
                                               ccs_mean,
                                               ccs_std,
                                               shuffle_files=True ),
                 'test' : SculptorCCSDataset( test_files,
                                              ccs_mean,
                                              ccs_std,
                                              shuffle_files=False ), }

    model = initialize_sculptor_model( model_file=model_file,
                                       arch_overrides=arch_overrides )

    loss_fx = CCS_HuberLoss()

    optimizer = training_parameters[ 'optimizer' ]( list( model.parameters() ),
                                                    lr=training_parameters[ 'learning_rate' ] )

    num_epochs = n_epochs if n_epochs is not None else training_parameters[ 'n_epochs' ]

    best_loss = train_model( model,
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
                             progress_tick_rows=progress_tick_rows,
                             num_workers=num_workers,
                             patience=patience,
                             start_epoch=start_epoch )

    best_model = initialize_sculptor_model( model_file=output_file_name,
                                            arch_overrides=arch_overrides,
                                            map_location='cpu' )
    metrics = evaluate_metrics( best_model,
                                test_files,
                                ccs_mean,
                                ccs_std,
                                eval_batch_size,
                                num_workers )

    print( 'Best normalized test loss: ' + format( float(best_loss), '.6f' ) )
    print( 'Test MAE (CCS): ' + format( metrics[ 'mae' ], '.6f' ) )
    print( 'Test RMSE (CCS): ' + format( metrics[ 'rmse' ], '.6f' ) )
    print( 'Test sample count: ' + str(metrics[ 'n_test' ]) )

    return { 'best_test_loss' : float( best_loss ),
             'test_mae' : float( metrics[ 'mae' ] ),
             'test_rmse' : float( metrics[ 'rmse' ] ),
             'n_test' : int( metrics[ 'n_test' ] ), }


def main():
    args = parse_args( sys.argv[1:] )
    os.makedirs( args.output_dir, exist_ok=True )

    model_out_file = os.path.join( args.output_dir, args.output_file )

    if args.model_file is not None:
        if os.path.abspath( model_out_file ) == os.path.abspath( args.model_file ):
            print( 'Error: --output_file resolves to the same path as --model_file. '
                   'Use a different output filename to avoid overwriting the source model.' )
            sys.exit( 1 )

    train_sculptor( args.dataset_root,
                    model_out_file,
                    device=args.device,
                    num_workers=args.num_workers,
                    patience=args.patience,
                    model_file=args.model_file,
                    start_epoch=args.start_epoch,
                    n_epochs=args.n_epochs,
                    arch_overrides=None,
                    metadata_file=args.metadata_file,
                    eval_batch_size=args.eval_batch_size )


if __name__ == '__main__':
    main()
    sys.exit()
