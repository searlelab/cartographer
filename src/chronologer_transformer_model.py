
import torch
import torch.nn as nn
from transformer_layers import PositionalEncoder


class chronologer_model( nn.Module ):
    def __init__( self, vec_length, n_states, embed_dim, n_heads, ff_neurons, drop_rate, act_fx, n_layers, ):
        super().__init__()
        self.seq_embed = nn.Embedding( n_states, embed_dim, padding_idx=0, )
        torch.nn.init.kaiming_normal_( self.seq_embed.weight, nonlinearity='linear', )

        self.pos_encoder = PositionalEncoder( embed_dim, drop_rate, vec_length, )

        transformer_layer = nn.TransformerEncoderLayer( embed_dim, n_heads, 
                                                        dim_feedforward=ff_neurons, dropout=drop_rate, 
                                                        activation=act_fx, batch_first=True,)
        self.transformer_encoder = nn.TransformerEncoder( transformer_layer, n_layers )

        #self.dropout = nn.Dropout( drop_rate, )
        self.flatten = nn.Flatten()
        self.output = nn.Linear( vec_length * embed_dim, 1, )
        nn.init.xavier_normal_( self.output.weight, )
        nn.init.constant_( self.output.bias.data, 0.0, )

    def forward( self, x, ):
        x = self.seq_embed( x )
        x = self.pos_encoder( x )
        x = self.transformer_encoder( x )
        x = self.flatten( x )
        return self.output( x )
        