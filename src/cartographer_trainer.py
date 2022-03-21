import os, sys, pickle, random, re, argparse
from datetime import datetime
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from cartographer_settings import hyperparameters, training_parameters
from tensorize import residues
from cartographer_model import cartographer_model
from loss_functions import Spectrum_masked_negLogit
from training_loop import train_model

import torch
from torch.utils.data import TensorDataset

from constants import cartographer_ptdata_loc, seed, validation_fraction, max_peptide_len, min_precursor_charge, max_precursor_charge
from tensorize import char_mapper

def parse_args(args):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime( '%Y%m%d%H%M%S' )
    default_out_filename = 'Cartographer_'+timestamp+'.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file',
                        type=str,
                        help='Model filename',
                        default=default_out_filename)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to save model (default is Cartographer/models)',
                        default=os.path.join(src_dir,'..','models'))
    return parser.parse_args(args)


def load_data( data_dir, ): ## WILL NEED TO FIX WITH SPLIT_TRAIN_TEST_DATA
    files = [ f for f in os.listdir(data_dir) if f[-4:] == '.pkl' ]
    data = {}
    for f in files:
        ## NAMING SCHEME: FILE NAMES ARE DATASET, FRAGMENTATION TYPE, AND NCE SEPARATED BY _
        dataset, frag_type, nce = f.split('_')[:3]
        nce = float( nce )    
        
        f_data = pickle.load( open( os.path.join( data_dir, f ) ,'rb') )
        f_data = dict( [ i for i in f_data.items()
                       if len(i[0][0][1:-1]) <= max_peptide_len and
                          int(i[0][1]) >= min_precursor_charge and 
                          int(i[0][1]) <= max_precursor_charge ] )
        ## NEW FILTER ON LADDER ION NUMBER
        f_data = dict( [ i for i in f_data.items()
                       if np.sum(i[1]['ion_array'] >= 0.01) >= 3 ] )
        data[(dataset,frag_type,nce)] = f_data
        
    return data
      
        

def split_train_test_data( data, test_frac, random_seed, ): # NEED TO FIX INPUTS
    proc_data = dict( [ (d,dict([ (v,[[],[],[],[]]) for v in ['x','y','w'] ])) for d in ['train','test'] ] )
    random.seed( random_seed )
    for d in data:
        data_set, frag_type, nce = d
        
        all_keys = list( data[d] )
        random.shuffle( all_keys )
        test_idx = int( np.round( len(all_keys)*test_frac ) )
        keys = { 'train' : all_keys[test_idx:],
                 'test'  : all_keys[:test_idx] }
        
        ## EVENTUALLY NEED TO MOVE THIS CODE TO TENSORIZE
        ## LEAVE HERE FOR NOW FOR TESTING PURPOSES
        ## PROBABLY RE-ORGANIZE SO THAT IT IS ALREADY IN TENSOR FORM
        ## WHEN IT COMES INTO TRAIN-TEST SPLIT
        
        for data_type in keys:
            seq_array = [ char_mapper( re.sub(r'\[.+?\]',
                                          '',
                                          data[d][x]['peptide_mod_seq'] ) ) 
                      for x in keys[data_type] ]
            pz_array = [ [1 if data[d][x]['precursor_z'] == z else 0 
                        for z in range(min_precursor_charge,max_precursor_charge+1)] for x in keys[data_type] ]
        
            frag_array = [ [ 1 if frag_type == 'HCD' else 0 ] 
                        for _ in keys[data_type] ]
            nce_array = [ [ float(nce) if frag_type == 'HCD' else 0.0 ] 
                        for _ in keys[data_type] ]
        
            weights = [ [ 1.0 ] for _ in keys[data_type] ]
            
            intensities = [ list(data[d][x]['ion_array']) for x in keys[data_type] ]
        
            proc_data[data_type]['x'][0] += seq_array
            proc_data[data_type]['x'][1] += pz_array #features
            proc_data[data_type]['x'][2] += nce_array
            proc_data[data_type]['x'][3] += frag_array
            proc_data[data_type]['y'][0] += intensities
            proc_data[data_type]['w'][0] += weights
            
    for data_type in proc_data:
        proc_data[data_type]['x'] = [ torch.Tensor( np.array(x) ) for x in proc_data[data_type]['x'] ]
        proc_data[data_type]['x'][0] = proc_data[data_type]['x'][0].to( torch.int64 )
        #data[data_type]['x'][1] = data[data_type]['x'][1].to( torch.int64 )
        proc_data[data_type]['x'][3] = proc_data[data_type]['x'][3].to( torch.bool  )
        proc_data[data_type]['y'] = torch.Tensor( np.array( proc_data[data_type]['y'][0] ) )
        #norm_weights = np.array( data[data_type]['w'][0] ) / np.mean( data[data_type]['w'][0] )
        #print( norm_weights )
        proc_data[data_type]['w'] = torch.Tensor( np.array( proc_data[data_type]['w'][0] ) )
    #print( proc_data['train']['x'] )
    #print( proc_data['train']['y'] )
    #print( proc_data['train']['w'] )
    datasets = dict( [ (dt, TensorDataset( *proc_data[dt]['x'], proc_data[dt]['y'], proc_data[dt]['w'] ) ) 
                    for dt in proc_data ] ) 
        
    return datasets


def train_cartographer( output_file_name, test_frac=validation_fraction, random_seed = seed, ):
    # Prepare data
    print( 'Cartographer training initiated' )
    data = load_data( cartographer_ptdata_loc )
    print( 'Cartographer PT data loaded' )
    datasets = split_train_test_data( data, test_frac, random_seed )
    print( 'Training and testing data split' )
    model = cartographer_model( max_peptide_len + 2,
                                len( residues ) + 1, 
                                max_precursor_charge - min_precursor_charge + 1,
                                hyperparameters[ 'embed_dimension' ],
                                hyperparameters[ 'nce_encode_dimension' ],
                                hyperparameters[ 'n_resnet_blocks' ],
                                hyperparameters[ 'kernel_size' ],
                                training_parameters[ 'dropout_rate' ],
                                hyperparameters[ 'activation_function' ], )
    
    loss_fx = Spectrum_masked_negLogit( )
    
    parameters = list(model.parameters())
    optimizer = training_parameters[ 'optimizer' ]( parameters,
                                                    lr = training_parameters[ 'learning_rate' ], )
    
    print( 'Ready to begin Cartographer training' )
    final_loss = train_model( model, 
                              datasets, 
                              training_parameters[ 'initial_batch_size' ], 
                              training_parameters[ 'max_batch_size' ], 
                              training_parameters[ 'epochs_to_2x_batch' ],
                              loss_fx, 
                              optimizer,
                              training_parameters[ 'n_epochs'], 
                              training_parameters[ 'train_device' ], 
                              training_parameters[ 'eval_device' ],
                              output_file_name, )
    return final_loss
    
def main():
    args = parse_args(sys.argv[1:])
    model_out_file = os.path.join( args.output_dir, args.output_file, )
    train_cartographer( model_out_file, )


if __name__ == "__main__":
    main()
    sys.exit()
