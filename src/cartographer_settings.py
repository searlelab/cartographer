import torch


hyperparameters = { 'embed_dimension' : 128,
                    'nce_encode_dimension' : 32,
                    'n_resnet_blocks' : 3,
                    'kernel_size' : 9,
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

# Prosit-specific constants (separate from constants.py to preserve Chronologer defaults)
max_peptide_len = 31         # Prosit max (constants.py stays 35 for Chronologer)
n_ion_channels = 6           # y+1, y+2, y+3, b+1, b+2, b+3
ms2_vector_len = 174         # 6 * 29
progress_tick_rows = 2**18   # 262,144 rows per tick
