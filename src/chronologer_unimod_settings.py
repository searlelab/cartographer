import torch


epsilon = 1e-7
train_fdr = 0.01

chronologer_min_peptide_len = 6
chronologer_max_peptide_len = 50

default_target_column = 'indexed_retention_time'
default_source_column = 'package'
default_modified_sequence_column = 'modified_sequence'

source_min_rows_per_split = 2
progress_tick_rows = 100000


hyperparameters = {
    'embed_dimension': 64,
    'n_resnet_blocks': 3,
    'kernel_size': 7,
    'activation_function': 'relu',
}


training_parameters = {
    'n_epochs': 100,
    'learning_rate': 1e-3,
    'dropout_rate': 0.1,
    'initial_batch_size': 64,
    'epochs_to_2x_batch': 30,
    'max_batch_size': 1024,
    'optimizer': torch.optim.Adam,
    'train_device': 'cuda',
    'eval_device': 'cpu',
    'loss_family': 'laplace',
}

