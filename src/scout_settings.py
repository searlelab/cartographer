import torch


hyperparameters = { 'embed_dimension' : 64,
                    'nce_encode_dimension' : 16,
                    'n_resnet_blocks' : 3,
                    'kernel_size' : 7,
                    'activation_function' : 'relu',
                  }

training_parameters = { 'n_epochs' : 200,
                        'learning_rate' : 1e-3,
                        'dropout_rate' : 0.18,
                        'initial_batch_size' : 64,
                        'epochs_to_2x_batch' : 60,
                        'max_batch_size' : 16384,
                        'optimizer' : torch.optim.Adam,
                        'train_device' : 'auto',
                        'eval_device' : 'auto', }

max_peptide_len = 31
n_ion_channels = 4
ms2_vector_len = 116
progress_tick_rows = 2**18

metadata_filename = 'Scout_metadata.json'
