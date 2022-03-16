
import numpy as np

from local_io import read_rt_database
from constants import chronologer_db_loc, seed, validation_fraction, max_peptide_len 
from chronologer_settings import hyperparameters, training_parameters
from tensorize import hi_db_to_tensors, residues
from chronologer_model import chronologer_model
from loss_functions import RT_masked_negLogL
from training_loop import train_model


def split_train_test_data( db, test_frac, random_seed, ):
    db = db.sample( frac=1, random_state=random_seed, )
    db = db.reset_index() # Reset now that its shuffled
    split_idx = int( np.round( test_frac * len(db) ) )
    split_data = { 'train' : hi_db_to_tensors( db.iloc[ split_idx:, ] ),
                   'test'  : hi_db_to_tensors( db.iloc[ :split_idx, ] ), }
    return split_data
    


def train_chronologer( device='cpu', test_frac=validation_fraction, random_seed = seed, ):
    
    # Prepare data
    print( 'Chronologer training initiated' )
    chronologer_db = read_rt_database( chronologer_db_loc, )
    print( 'Chronologer database successfully loaded' )
    datasets = split_train_test_data( chronologer_db, test_frac, random_seed )
    print('Training and testing data split')
    model = chronologer_model( max_peptide_len + 2,
                               len( residues ) + 1,
                               hyperparameters[ 'embed_dimension' ],
                               hyperparameters[ 'n_resnet_blocks' ],
                               hyperparameters[ 'kernel_size' ],
                               training_parameters[ 'dropout_rate'],
                               hyperparameters[ 'activation_function' ], )
    
    n_sources = len( set( chronologer_db.Source ) )
    loss_fx = RT_masked_negLogL( n_sources, )
    
    parameters = list(model.parameters()) + list(loss_fx.parameters())
    optimizer = training_parameters[ 'optimizer' ]( parameters,
                                                    lr = training_parameters[ 'learning_rate' ], )
    
    ## ## ##
    ## ## ##
    output_file_name = 'TEST.PT' ## NEED TO FIX
    ## ## ##
    ## ## ##
    print( 'Ready to begin Chronologer training' )
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
    
    
    
    
train_chronologer()
