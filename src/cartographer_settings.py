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
                        'train_device' : 'cuda',
                        'eval_device' : 'cpu', }

