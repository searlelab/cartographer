
import torch
import torch.nn as nn
from core_layers import resnet_block


class chronologer_model( nn.Module ):
    def __init__( self, vec_length, n_states, embed_dim, n_blocks, kernel, drop_rate, act_fx, ):
        super().__init__()
        self.seq_embed = nn.Embedding( n_states, embed_dim, padding_idx=0, )
        torch.nn.init.kaiming_normal_( self.seq_embed.weight, nonlinearity='linear', )
        self.resnet_blocks = nn.Sequential( *[ resnet_block( embed_dim, 
                                                             embed_dim,
                                                             kernel, 
                                                             d+1, 
                                                             act_fx, ) for d in range(n_blocks) ] )
        self.dropout = nn.Dropout( drop_rate, )
        self.flatten = nn.Flatten()
        self.output = nn.Linear( vec_length * embed_dim, 1, )
        nn.init.xavier_normal_( self.output.weight, )
        nn.init.constant_( self.output.bias.data, 0.0, )
        
    def forward( self, x, ):
        x = self.seq_embed( x ).transpose( 1, -1, )
        x = self.resnet_blocks( x )
        x = self.dropout( x )
        x = self.flatten( x )
        return self.output( x )
        