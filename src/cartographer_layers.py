

import torch
import torch.nn as nn
from .resnet_layers import process_unit, resnet_block, activation_func




class precursor_encoder( nn.Module ):
    def __init__( self, vec_length, n_states, n_charges, embed_dim, output_dim, n_blocks, kernel, drop_rate, act_fx, ):
        super().__init__()
        self.seq_embed = nn.Embedding( n_states, 
                                       embed_dim, 
                                       padding_idx=0, )
        self.pz_linear_1= nn.Linear( n_charges, 
                                     vec_length, )
        self.pz_linear_2 = nn.Linear( 1, 
                                      embed_dim, )
        self.resnet_blocks = nn.Sequential( *[ resnet_block( 'conv', 
                                                             embed_dim, 
                                                             embed_dim, 
                                                             act_fx, 
                                                             kernel, 
                                                             d+1, ) 
                                               for d in range(n_blocks) ] )
        self.dropout = nn.Dropout( drop_rate )
        self.flatten = nn.Flatten()
        self.norm = nn.BatchNorm1d( vec_length )
        self.activate = activation_func( act_fx )
        self.output = nn.Linear( vec_length * embed_dim, output_dim, )

    def pz_embed( self, pz ):
        x = self.pz_linear_1(  pz )
        x.unsqueeze_( -1 )
        return self.pz_linear_2( x )
    
    def forward(self, seq, pz):
        seq_embed = self.seq_embed( seq )
        pz_embed = self.pz_embed( pz )
        x = seq_embed * pz_embed
        x.transpose_( 1, -1 )
        x = self.resnet_blocks( x )
        x = self.dropout( x )
        x = self.flatten( x )
        return self.output( x )


class hcd_decoder( nn.Module ):
    def __init__( self, input_dim, regression_dim, output_dim, n_blocks, act_fx, ):
        super().__init__()
        self.ion_embed = nn.Linear( input_dim, regression_dim, )
        self.nce_embed = nn.Sequential( process_unit( 'linear', 1, 32, ),
                                        nn.Linear( 32, regression_dim, ), 
                                      )
        self.resnet_blocks = nn.Sequential( *[ resnet_block( 'linear', 
                                                             regression_dim, 
                                                             regression_dim, 
                                                             act_fx, )
                                               for _ in range( n_blocks ) ] )
        self.output = nn.Linear( regression_dim, output_dim, )

    def forward( self, ion_encoded, nce, ):
        ion_embed = self.ion_embed( ion_encoded, )
        nce_embed = self.nce_embed( nce, )
        x = ion_embed * nce_embed
        x = self.resnet_blocks( x )
        return self.output( x )


class cid_decoder( nn.Module ):
    def __init__( self, input_dim, regression_dim, output_dim, n_blocks, act_fx, ):
        super().__init__()
        self.ion_embed = nn.Linear( input_dim, regression_dim, )
        self.resnet_blocks = nn.Sequential( *[ resnet_block( 'linear', 
                                                             regression_dim, 
                                                             regression_dim, 
                                                             act_fx, )
                                               for _ in range( n_blocks ) ] )
        self.output = nn.Linear( regression_dim, output_dim, )

    def forward( self, ion_encoded, ):
        x = self.ion_embed( ion_encoded, )
        x = self.resnet_blocks( x )
        return self.output( x )



