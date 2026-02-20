import os, sys, argparse
from datetime import datetime
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from cartographer_settings import training_parameters, progress_tick_rows, max_peptide_len
from cartographer_model import initialize_cartographer_model
from loss_functions import Spectrum_masked_negLogit
from training_loop import train_model, resolve_device
from prospect_loader import ProspectMS2Dataset, discover_split_files
from tensorize import codedseq_to_array

import torch


PRTC_PEPTIDES = [ 'SSAAPPPPPR', 'GISNEGQNASIK', 'HVLTSIGEK', 'DIPVPKPK',
                  'IGDYAGIK', 'TASEFDSAIAQDK', 'SAAGAFGPELSR', 'ELGQSGVDTYLQTK',
                  'GLILVGGYGTR', 'GILFVGSGVSGGEEGAR', 'SFANQPLEVVYSK',
                  'LTILEELR', 'NGFILDGFPR', 'ELASGLSFPVGFK', 'LSSEAPALFQFDLK', ]

PRTC_NCE = 0.33   # NCE 33, normalized /100
PRTC_CHARGE = 2    # +2H, one-hot index 1


def build_prtc_inputs( device ):
    """Pre-build PRTC peptide tensors for inference."""
    seq_size = max_peptide_len + 2
    n = len( PRTC_PEPTIDES )

    seq_arrays = []
    for pep in PRTC_PEPTIDES:
        coded = '-' + pep + '_'
        seq_arrays.append( codedseq_to_array( coded, max_size=seq_size ) )

    seq_t = torch.from_numpy( np.array( seq_arrays, 'int64' ) ).to( device )
    charge_t = torch.zeros( n, 6, device=device )
    charge_t[ :, PRTC_CHARGE - 1 ] = 1.0
    nce_t = torch.full( (n, 1), PRTC_NCE, device=device )

    return seq_t, charge_t, nce_t


def make_prtc_callback( prtc_file, device ):
    """Return a callback that writes PRTC predictions to TSV after each epoch."""
    device = resolve_device( device )
    seq_t, charge_t, nce_t = build_prtc_inputs( device )

    def callback( model, epoch ):
        model.to( device )
        model.eval()
        with torch.no_grad():
            pred = model( seq_t, charge_t, nce_t )
        pred_np = pred.cpu().numpy()

        with open( prtc_file, 'a' ) as f:
            for i, pep in enumerate( PRTC_PEPTIDES ):
                vals = '\t'.join( format( v, '.6f' ) for v in pred_np[i] )
                f.write( str(epoch) + '\t' + pep + '\t' + vals + '\n' )

    return callback


def parse_args(args):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime( '%Y%m%d%H%M%S' )
    default_out_filename = 'Cartographer_'+timestamp+'.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root',
                        type=str,
                        required=True,
                        help='Path to prospect-ptms-ms2 dataset directory')
    parser.add_argument('--output_file',
                        type=str,
                        help='Model filename',
                        default=default_out_filename)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to save model (default is Cartographer/models)',
                        default=os.path.join(src_dir,'..','models'))
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
                        help='TSV file to log PRTC peptide predictions each epoch',
                        default=None)
    parser.add_argument('--model_file',
                        type=str,
                        help='Path to a .pt state dict to resume from (default: train from scratch)',
                        default=None)
    parser.add_argument('--start_epoch',
                        type=int,
                        help='Epoch number to begin at (default: 1)',
                        default=1)
    parser.add_argument('--n_epochs',
                        type=int,
                        help='Total epoch count, overrides settings (default: use training_parameters)',
                        default=None)
    return parser.parse_args(args)


def train_cartographer( dataset_root, output_file_name, device='auto', num_workers=0,
                        prtc_report=None, model_file=None, start_epoch=1, n_epochs=None, ):
    print( 'Cartographer training initiated' )

    # Discover pre-split parquet shards
    train_files = discover_split_files( dataset_root, 'train' )
    test_files = discover_split_files( dataset_root, 'test' )
    print( 'Found ' + str(len(train_files)) + ' train shards, ' +
           str(len(test_files)) + ' test shards' )
    assert len(train_files) > 0, 'No train parquet files found in ' + dataset_root
    assert len(test_files) > 0, 'No test parquet files found in ' + dataset_root

    datasets = { 'train' : ProspectMS2Dataset( train_files, shuffle_files=True ),
                 'test'  : ProspectMS2Dataset( test_files,  shuffle_files=False ), }
    print( 'Datasets created' )

    model = initialize_cartographer_model( frag_type='beam', model_file=model_file )

    loss_fx = Spectrum_masked_negLogit( )

    parameters = list(model.parameters())
    optimizer = training_parameters[ 'optimizer' ]( parameters,
                                                    lr = training_parameters[ 'learning_rate' ], )

    # Override device from settings if specified via CLI
    train_device = device
    eval_device = device

    # PRTC report callback
    epoch_callback = None
    if prtc_report is not None:
        # Truncate file and write header
        with open( prtc_report, 'w' ) as f:
            f.write( 'epoch\tpeptide\t' + '\t'.join( 'i' + str(i) for i in range(174) ) + '\n' )
        epoch_callback = make_prtc_callback( prtc_report, device )
        print( 'PRTC report: ' + prtc_report )

    num_epochs = n_epochs if n_epochs is not None else training_parameters[ 'n_epochs' ]

    print( 'Ready to begin Cartographer training' )
    final_loss = train_model( model,
                              datasets,
                              training_parameters[ 'initial_batch_size' ],
                              training_parameters[ 'max_batch_size' ],
                              training_parameters[ 'epochs_to_2x_batch' ],
                              loss_fx,
                              optimizer,
                              num_epochs,
                              train_device,
                              eval_device,
                              output_file_name,
                              progress_tick_rows=progress_tick_rows,
                              num_workers=num_workers,
                              epoch_callback=epoch_callback,
                              start_epoch=start_epoch, )
    return final_loss

def main():
    args = parse_args(sys.argv[1:])
    model_out_file = os.path.join( args.output_dir, args.output_file, )
    os.makedirs( args.output_dir, exist_ok=True )

    # Safety check: prevent overwriting the source model
    if args.model_file is not None:
        if os.path.abspath( model_out_file ) == os.path.abspath( args.model_file ):
            print( 'Error: --output_file resolves to the same path as --model_file ('
                   + os.path.abspath( model_out_file ) + '). '
                   'Use a different output filename to avoid overwriting the source model.' )
            sys.exit(1)

    train_cartographer( args.dataset_root, model_out_file,
                        device=args.device, num_workers=args.num_workers,
                        prtc_report=args.prtc_report,
                        model_file=args.model_file,
                        start_epoch=args.start_epoch,
                        n_epochs=args.n_epochs, )


if __name__ == "__main__":
    main()
    sys.exit()
