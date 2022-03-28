import torch
import torch.nn as nn
from core_layers import resnet_block

from cartographer_settings import hyperparameters, training_parameters
from tensorize import residues
from constants import max_peptide_len, min_precursor_charge, max_precursor_charge


class precursor_charge_embed( nn.Module ):
    def __init__( self, vec_length, n_charges, embed_dim, ):
        super().__init__()
        self.embed_dim1 = nn.Linear( n_charges, vec_length, )
        self.embed_dim2 = nn.Linear( 1, embed_dim, )
        
    def forward( self, x ):
        x = self.embed_dim1( x )
        x.unsqueeze_( -1 )
        return self.embed_dim2( x )

class nce_embed( nn.Module ):
    def __init__( self, vec_length, embed_dim, nce_dim, ):
        super().__init__()
        self.embed_dim1 = nn.Sequential( nn.Linear( 1, nce_dim, ),
                                         nn.ReLU(),
                                         nn.Linear( nce_dim, vec_length, ) )
        self.embed_dim2 = nn.Linear( 1, embed_dim, )
    
    def forward( self, x, ):
        x = self.embed_dim1( x )
        x.unsqueeze_( -1 )
        return self.embed_dim2( x )

class cartographer_model( nn.Module ):
    def __init__( self, vec_length, n_states, n_charges, embed_dim, nce_dim, n_blocks, kernel, drop_rate, act_fx, ):
        super().__init__()
        self.seq_embed = nn.Embedding( n_states, embed_dim, padding_idx=0, )
        self.charge_embed = precursor_charge_embed( vec_length, n_charges, embed_dim, )
        self.nce_embed = nce_embed( vec_length, embed_dim, nce_dim, )
        self.resnet_blocks = nn.Sequential( *[ resnet_block( embed_dim, 
                                                             embed_dim,
                                                             kernel, 
                                                             d+1, 
                                                             act_fx, ) for d in range(n_blocks) ] )
        self.dropout = nn.Dropout( drop_rate )
        self.hcd_conv = nn.Conv1d( embed_dim, 4, kernel_size=4, )
        self.cid_conv = nn.Conv1d( embed_dim, 4, kernel_size=4, )
        
    
    def feat_embed( self, seq, z, nce, ):
        seq_embed = self.seq_embed( seq )
        charge_embed = self.charge_embed( z )
        nce_embed = self.nce_embed( nce )
        return seq_embed * charge_embed * nce_embed
    
    def encoder( self, seq, z, nce, ):
        x = self.feat_embed( seq, z, nce, )
        x.transpose_( 1, -1 ) # Rotate seq dimension from dim1 to dim2
        x = self.resnet_blocks( x )
        return self.dropout( x )
        
    def normalize( self, x, ):
        return x.clamp( min=0.0, ) / x.amax( dim=(1,-1), keepdim=True, )
    
    def hcd_decoder( self, x, ):
        x = self.hcd_conv( x )
        return self.normalize( x )
    
    def cid_decoder( self, x, ):
        x = self.cid_conv( x )
        return self.normalize( x )
    
    def output( self, x, hcd_bool, ):
        hcd = self.hcd_decoder( x )
        cid = self.cid_decoder( x )
        hcd_bool.unsqueeze_( -1, )
        return hcd * hcd_bool + cid * ~hcd_bool
    
    def forward(self, seq, z, nce, hcd_bool, ):
        x = self.encoder( seq, z, nce, )
        return self.output( x, hcd_bool, )
    

class cartographer_cid( cartographer_model ):
    def forward( self, seq, z, ):
        x = self.encoder( seq, z, torch.zeros(z.size(0).unsqueeze(-1)), )
        return self.cid_decoder( x )


class cartographer_hcd( cartographer_model ):
    def forward( self, seq, z, nce, ):
        x = self.encoder( seq, z, nce, )
        return self.hcd_decoder( x )
    

def initialize_cartographer_model( frag_type = None, model_file = None, ):
    if frag_type == 'beam':
        cartographer = cartographer_hcd
    elif frag_type == 'resonance':
        cartographer = cartographer_cid
    else:
        cartographer = cartographer_model
    
    model = cartographer( max_peptide_len + 2,
                          len( residues ) + 1, 
                          max_precursor_charge - min_precursor_charge + 1,
                          hyperparameters[ 'embed_dimension' ],
                          hyperparameters[ 'nce_encode_dimension' ],
                          hyperparameters[ 'n_resnet_blocks' ],
                          hyperparameters[ 'kernel_size' ],
                          training_parameters[ 'dropout_rate' ],
                          hyperparameters[ 'activation_function' ], )
    if model_file:
        model.load_state_dict( torch.load( model_file ), strict=True, )
    model.eval()
    return model
