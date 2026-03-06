import torch


hyperparameters = { 'embed_dimension' : 64,
                    'n_resnet_blocks' : 3,
                    'kernel_size' : 7,
                    'activation_function' : 'relu',
                  }

training_parameters = { 'n_epochs' : 200,
                        'learning_rate' : 1e-3,
                        'dropout_rate' : 0.10,
                        'initial_batch_size' : 64,
                        'epochs_to_2x_batch' : 60,
                        'max_batch_size' : 16384,
                        'optimizer' : torch.optim.Adam,
                        'train_device' : 'auto',
                        'eval_device' : 'auto', }

# Prosit-specific constants (separate from constants.py to preserve Chronologer defaults)
max_peptide_len = 31         # Prosit max (same as Cartographer)
charge_dist_len = 6          # charge states +1 through +6
progress_tick_rows = 2**18   # 262,144 rows per tick
