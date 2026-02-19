import os, sys, argparse
from datetime import datetime
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from electrician_settings import training_parameters, progress_tick_rows, max_peptide_len
from electrician_model import initialize_electrician_model
from loss_functions import ChargeDistribution_CrossEntropy
from training_loop import train_model, resolve_device
from prospect_loader import ProspectChargeDataset, discover_split_files
from tensorize import codedseq_to_array

import torch


PRTC_PEPTIDES = [ 'SSAAPPPPPR', 'GISNEGQNASIK', 'HVLTSIGEK', 'DIPVPKPK',
                  'IGDYAGIK', 'TASEFDSAIAQDK', 'SAAGAFGPELSR', 'ELGQSGVDTYLQTK',
                  'GLILVGGYGTR', 'GILFVGSGVSGGEEGAR', 'SFANQPLEVVYSK',
                  'LTILEELR', 'NGFILDGFPR', 'ELASGLSFPVGFK', 'LSSEAPALFQFDLK', ]


def build_prtc_inputs( device ):
    """Pre-build PRTC peptide tensors for inference."""
    seq_size = max_peptide_len + 2
    n = len( PRTC_PEPTIDES )

    seq_arrays = []
    for pep in PRTC_PEPTIDES:
        coded = '-' + pep + '_'
        seq_arrays.append( codedseq_to_array( coded, max_size=seq_size ) )

    seq_t = torch.from_numpy( np.array( seq_arrays, 'int64' ) ).to( device )
    return seq_t


def make_prtc_callback( prtc_file, device ):
    """Return a callback that writes PRTC charge predictions to TSV after each epoch."""
    device = resolve_device( device )
    seq_t = build_prtc_inputs( device )

    def callback( model, epoch ):
        model.to( device )
        model.eval()
        with torch.no_grad():
            pred = model( seq_t )
        pred_np = pred.cpu().numpy()

        with open( prtc_file, 'a' ) as f:
            for i, pep in enumerate( PRTC_PEPTIDES ):
                vals = '\t'.join( format( v, '.6f' ) for v in pred_np[i] )
                f.write( str(epoch) + '\t' + pep + '\t' + vals + '\n' )

    return callback


def parse_args(args):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime( '%Y%m%d%H%M%S' )
    default_out_filename = 'Electrician_'+timestamp+'.pt'
    parser = argparse.ArgumentParser()
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
                        help='TSV file to log PRTC peptide charge predictions each epoch',
                        default=None)
    return parser.parse_args(args)


def train_electrician( dataset_root, output_file_name, device='auto', num_workers=0, prtc_report=None, ):
    print( 'Electrician training initiated' )

    # Discover pre-split parquet shards
    train_files = discover_split_files( dataset_root, 'train' )
    test_files = discover_split_files( dataset_root, 'test' )
    print( 'Found ' + str(len(train_files)) + ' train shards, ' +
           str(len(test_files)) + ' test shards' )
    assert len(train_files) > 0, 'No train parquet files found in ' + dataset_root
    assert len(test_files) > 0, 'No test parquet files found in ' + dataset_root

    datasets = { 'train' : ProspectChargeDataset( train_files, shuffle_files=True ),
                 'test'  : ProspectChargeDataset( test_files,  shuffle_files=False ), }
    print( 'Datasets created' )

    model = initialize_electrician_model()

    loss_fx = ChargeDistribution_CrossEntropy( )

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
            f.write( 'epoch\tpeptide\tz1\tz2\tz3\tz4\tz5\tz6\n' )
        epoch_callback = make_prtc_callback( prtc_report, device )
        print( 'PRTC report: ' + prtc_report )

    print( 'Ready to begin Electrician training' )
    final_loss = train_model( model,
                              datasets,
                              training_parameters[ 'initial_batch_size' ],
                              training_parameters[ 'max_batch_size' ],
                              training_parameters[ 'epochs_to_2x_batch' ],
                              loss_fx,
                              optimizer,
                              training_parameters[ 'n_epochs'],
                              train_device,
                              eval_device,
                              output_file_name,
                              progress_tick_rows=progress_tick_rows,
                              num_workers=num_workers,
                              epoch_callback=epoch_callback, )
    return final_loss

def main():
    args = parse_args(sys.argv[1:])
    model_out_file = os.path.join( args.output_dir, args.output_file, )
    os.makedirs( args.output_dir, exist_ok=True )
    train_electrician( args.dataset_root, model_out_file,
                        device=args.device, num_workers=args.num_workers,
                        prtc_report=args.prtc_report, )


if __name__ == "__main__":
    main()
    sys.exit()
