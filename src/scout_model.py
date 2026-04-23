import torch
import torch.nn as nn

import scout_settings
from constants import max_precursor_charge, min_precursor_charge
from core_layers import resnet_block
from scout_settings import hyperparameters, ms2_vector_len, n_ion_channels, training_parameters
from tensorize import residues


def _init_linear( layer ):
    nn.init.xavier_normal_( layer.weight )
    nn.init.constant_( layer.bias.data, 0.0 )
    return layer


class precursor_charge_embed( nn.Module ):
    def __init__( self, vec_length, n_charges, embed_dim ):
        super().__init__()
        self.embed_dim1 = nn.Linear( n_charges, vec_length )
        self.embed_dim2 = nn.Linear( 1, embed_dim )

    def forward( self, x ):
        x = self.embed_dim1( x )
        x = x.unsqueeze( -1 )
        return self.embed_dim2( x )


class nce_embed( nn.Module ):
    def __init__( self, vec_length, embed_dim, nce_dim ):
        super().__init__()
        self.embed_dim1 = nn.Sequential( nn.Linear( 1, nce_dim ),
                                         nn.ReLU(),
                                         nn.Linear( nce_dim, vec_length ) )
        self.embed_dim2 = nn.Linear( 1, embed_dim )

    def forward( self, x ):
        x = self.embed_dim1( x )
        x = x.unsqueeze( -1 )
        return self.embed_dim2( x )


class scout_shared_encoder( nn.Module ):
    def __init__( self, vec_length, n_states, n_charges, embed_dim, nce_dim, n_blocks, kernel, drop_rate, act_fx ):
        super().__init__()
        self.seq_embed = nn.Embedding( n_states, embed_dim, padding_idx=0 )
        self.charge_embed = precursor_charge_embed( vec_length, n_charges, embed_dim )
        self.nce_embed = nce_embed( vec_length, embed_dim, nce_dim )
        self.resnet_blocks = nn.Sequential( *[ resnet_block( embed_dim,
                                                             embed_dim,
                                                             kernel,
                                                             d + 1,
                                                             act_fx )
                                               for d in range( n_blocks ) ] )
        self.dropout = nn.Dropout( drop_rate )
        self.flatten = nn.Flatten()

    def forward( self, seq, charge, nce ):
        x = self.seq_embed( seq )
        x = x * self.charge_embed( charge ) * self.nce_embed( nce )
        x = x.transpose( 1, -1 )
        x = self.resnet_blocks( x )
        x = self.dropout( x )
        pooled = self.flatten( x )
        return x, pooled


class scout_ms2_head( nn.Module ):
    def __init__( self, embed_dim, n_channels ):
        super().__init__()
        self.output = nn.Conv1d( embed_dim, n_channels, kernel_size=4 )

    def normalize( self, x ):
        return x.clamp( min=0.0 ) / x.amax( dim=(1, -1), keepdim=True ).clamp( min=1e-7 )

    def forward( self, seq_features ):
        x = self.output( seq_features )
        x = self.normalize( x )
        return x.flatten( 1 )[ :, :ms2_vector_len ]


class scout_irt_head( nn.Module ):
    def __init__( self, pooled_dim, embed_dim, act_fx ):
        super().__init__()
        self.layers = nn.Sequential( _init_linear( nn.Linear( pooled_dim, embed_dim ) ),
                                     nn.ReLU() if act_fx == 'relu' else nn.Identity(),
                                     _init_linear( nn.Linear( embed_dim, 1 ) ) )

    def forward( self, pooled_features ):
        return self.layers( pooled_features )


class scout_ccs_head( nn.Module ):
    def __init__( self, pooled_dim, embed_dim, n_charges, act_fx ):
        super().__init__()
        self.layers = nn.Sequential( _init_linear( nn.Linear( pooled_dim + n_charges, embed_dim ) ),
                                     nn.ReLU() if act_fx == 'relu' else nn.Identity(),
                                     _init_linear( nn.Linear( embed_dim, 1 ) ) )

    def forward( self, pooled_features, charge ):
        return self.layers( torch.cat( [ pooled_features, charge ], dim=1 ) )


class scout_model( nn.Module ):
    def __init__( self, vec_length, n_states, n_charges, embed_dim, nce_dim, n_blocks, kernel, drop_rate, act_fx ):
        super().__init__()
        self.encoder = scout_shared_encoder( vec_length,
                                             n_states,
                                             n_charges,
                                             embed_dim,
                                             nce_dim,
                                             n_blocks,
                                             kernel,
                                             drop_rate,
                                             act_fx )
        pooled_dim = vec_length * embed_dim
        self.ms2_head = scout_ms2_head( embed_dim, n_ion_channels )
        self.irt_head = scout_irt_head( pooled_dim, embed_dim, act_fx )
        self.ccs_head = scout_ccs_head( pooled_dim, embed_dim, n_charges, act_fx )

    def forward_shared( self, seq, charge, nce ):
        seq_features, pooled_features = self.encoder( seq, charge, nce )
        return { 'seq_features' : seq_features,
                 'pooled_features' : pooled_features }

    def forward( self, seq, charge, nce ):
        shared = self.forward_shared( seq, charge, nce )
        seq_features = shared[ 'seq_features' ]
        pooled_features = shared[ 'pooled_features' ]
        return { 'ms2' : self.ms2_head( seq_features ),
                 'irt' : self.irt_head( pooled_features ),
                 'ccs' : self.ccs_head( pooled_features, charge ), }


class scout_torchscript_wrapper( nn.Module ):
    def __init__( self, model ):
        super().__init__()
        self.model = model

    def forward( self, seq, charge, nce ):
        outputs = self.model( seq, charge, nce )
        return outputs[ 'ms2' ], outputs[ 'irt' ], outputs[ 'ccs' ]


def initialize_scout_model( model_file=None, map_location=None ):
    model = scout_model( scout_settings.max_peptide_len + 2,
                         len( residues ) + 1,
                         max_precursor_charge - min_precursor_charge + 1,
                         hyperparameters[ 'embed_dimension' ],
                         hyperparameters[ 'nce_encode_dimension' ],
                         hyperparameters[ 'n_resnet_blocks' ],
                         hyperparameters[ 'kernel_size' ],
                         training_parameters[ 'dropout_rate' ],
                         hyperparameters[ 'activation_function' ] )
    if model_file:
        model.load_state_dict( torch.load( model_file, map_location=map_location ), strict=True )
    return model
