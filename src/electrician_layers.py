

import torch
import torch.nn as nn
from core_layers import resnet_block, activation_func



class sequence_encoder( nn.Module ):
    """Standalone peptide sequence encoder for charge state prediction.
    Mirrors precursor_encoder in cartographer_layers.py but without
    charge or NCE embedding (since charge is the prediction target)."""
    def __init__( self, vec_length, n_states, embed_dim, output_dim, n_blocks, kernel, drop_rate, act_fx, ):
        super().__init__()
        self.seq_embed = nn.Embedding( n_states,
                                       embed_dim,
                                       padding_idx=0, )
        self.resnet_blocks = nn.Sequential( *[ resnet_block( embed_dim,
                                                             embed_dim,
                                                             kernel,
                                                             d+1,
                                                             act_fx, )
                                               for d in range(n_blocks) ] )
        self.dropout = nn.Dropout( drop_rate )
        self.flatten = nn.Flatten()
        self.activate = activation_func( act_fx )
        self.output = nn.Linear( vec_length * embed_dim, output_dim, )

    def forward( self, seq ):
        x = self.seq_embed( seq )
        x.transpose_( 1, -1 )
        x = self.resnet_blocks( x )
        x = self.dropout( x )
        x = self.flatten( x )
        return self.output( x )


