import torch


hyperparameters = { 'embed_dimension' : 64,
                    'n_heads' : 1,
                    'ff_neurons' : 64,
                    'activation_function' : 'relu',
                    'n_layers' : 3,
                  }

training_parameters = { 'n_epochs' : 100,
                        'learning_rate' : 1e-3,
                        'dropout_rate' : 0.1,
                        'initial_batch_size' : 64,
                        'epochs_to_2x_batch' : 30,
                        'max_batch_size' : 1024,
                        'optimizer' : torch.optim.SGD,
                        'train_device' : 'cuda',
                        'eval_device' : 'cuda', }

