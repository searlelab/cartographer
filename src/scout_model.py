import torch
import torch.nn as nn

from electrician_settings import charge_dist_len
import scout_settings
from constants import max_precursor_charge, min_precursor_charge
from core_layers import resnet_block
from scout_settings import hyperparameters, ms2_vector_len, n_ion_channels, training_parameters
from tensorize import residues


def _init_linear( layer ):
    nn.init.xavier_normal_( layer.weight )
    nn.init.constant_( layer.bias.data, 0.0 )
    return layer


class scout_attention_pool( nn.Module ):
    def __init__( self, embed_dim ):
        super().__init__()
        self.score = _init_linear( nn.Linear( embed_dim, 1 ) )

    def forward( self, seq_features, seq ):
        mask = seq.ne( 0 )
        logits = self.score( seq_features.transpose( 1, 2 ) ).squeeze( -1 )
        logits = logits.masked_fill( ~mask, float( '-inf' ) )
        valid_rows = mask.any( dim=1, keepdim=True )
        logits = torch.where( valid_rows, logits, torch.zeros_like( logits ) )
        weights = torch.softmax( logits, dim=1 )
        weights = weights * mask.to( weights.dtype )
        weights = weights / weights.sum( dim=1, keepdim=True ).clamp( min=1e-7 )
        return torch.sum( seq_features * weights.unsqueeze( 1 ), dim=-1 )


class scout_film_conditioner( nn.Module ):
    def __init__( self, n_charges, embed_dim, act_fx ):
        super().__init__()
        self.layers = nn.Sequential( _init_linear( nn.Linear( n_charges * 2 + 1, embed_dim ) ),
                                     nn.ReLU() if act_fx == 'relu' else nn.Identity(),
                                     _init_linear( nn.Linear( embed_dim, embed_dim * 2 ) ) )

    def forward( self, charge, nce ):
        cond = torch.cat( [ charge, nce, charge * nce ], dim=1 )
        gamma, beta = self.layers( cond ).chunk( 2, dim=1 )
        return gamma.unsqueeze( -1 ), beta.unsqueeze( -1 )


class scout_shared_encoder( nn.Module ):
    def __init__( self, n_states, embed_dim, n_blocks, kernel, drop_rate, act_fx ):
        super().__init__()
        self.seq_embed = nn.Embedding( n_states, embed_dim, padding_idx=0 )
        self.resnet_blocks = nn.Sequential( *[ resnet_block( embed_dim,
                                                             embed_dim,
                                                             kernel,
                                                             d + 1,
                                                             act_fx )
                                               for d in range( n_blocks ) ] )
        self.dropout = nn.Dropout( drop_rate )
        self.attention_pool = scout_attention_pool( embed_dim )

    def forward( self, seq ):
        x = self.seq_embed( seq )
        x = x.transpose( 1, -1 )
        x = self.resnet_blocks( x )
        x = self.dropout( x )
        pooled = self.attention_pool( x, seq )
        return x, pooled


class scout_ms2_head( nn.Module ):
    def __init__( self, embed_dim, n_channels, n_charges, kernel, act_fx ):
        super().__init__()
        self.conditioner = scout_film_conditioner( n_charges, embed_dim, act_fx )
        self.resnet_block = resnet_block( embed_dim, embed_dim, kernel, 1, act_fx )
        self.output = nn.Conv1d( embed_dim, n_channels, kernel_size=4 )

    def normalize( self, x ):
        return x.clamp( min=0.0 ) / x.amax( dim=(1, -1), keepdim=True ).clamp( min=1e-7 )

    def forward( self, seq_features, charge, nce ):
        gamma, beta = self.conditioner( charge, nce )
        x = seq_features * ( 1.0 + gamma ) + beta
        x = self.resnet_block( x )
        x = self.output( x )
        x = self.normalize( x )
        return x.flatten( 1 )[ :, :ms2_vector_len ]


class scout_irt_head( nn.Module ):
    def __init__( self, pooled_dim, embed_dim, act_fx ):
        super().__init__()
        hidden_dim = max( 1, embed_dim // 2 )
        self.layers = nn.Sequential( _init_linear( nn.Linear( pooled_dim, embed_dim ) ),
                                     nn.ReLU() if act_fx == 'relu' else nn.Identity(),
                                     _init_linear( nn.Linear( embed_dim, hidden_dim ) ),
                                     nn.ReLU() if act_fx == 'relu' else nn.Identity(),
                                     _init_linear( nn.Linear( hidden_dim, 1 ) ) )

    def forward( self, pooled_features ):
        return self.layers( pooled_features )


class scout_ccs_head( nn.Module ):
    def __init__( self, pooled_dim, embed_dim, n_charges, act_fx ):
        super().__init__()
        hidden_dim = max( 1, embed_dim // 2 )
        self.layers = nn.Sequential( _init_linear( nn.Linear( pooled_dim + n_charges, embed_dim ) ),
                                     nn.ReLU() if act_fx == 'relu' else nn.Identity(),
                                     _init_linear( nn.Linear( embed_dim, hidden_dim ) ),
                                     nn.ReLU() if act_fx == 'relu' else nn.Identity(),
                                     _init_linear( nn.Linear( hidden_dim, 1 ) ) )

    def forward( self, pooled_features, charge ):
        return self.layers( torch.cat( [ pooled_features, charge ], dim=1 ) )


class scout_charge_dist_head( nn.Module ):
    def __init__( self, pooled_dim, embed_dim, n_charges, act_fx ):
        super().__init__()
        hidden_dim = max( 1, embed_dim // 2 )
        self.layers = nn.Sequential( _init_linear( nn.Linear( pooled_dim, embed_dim ) ),
                                     nn.ReLU() if act_fx == 'relu' else nn.Identity(),
                                     _init_linear( nn.Linear( embed_dim, hidden_dim ) ),
                                     nn.ReLU() if act_fx == 'relu' else nn.Identity(),
                                     _init_linear( nn.Linear( hidden_dim, n_charges ) ) )

    def forward( self, pooled_features ):
        return torch.softmax( self.layers( pooled_features ), dim=1 )


class scout_model( nn.Module ):
    def __init__( self, vec_length, n_states, n_charges, embed_dim, nce_dim, n_blocks, kernel, drop_rate, act_fx ):
        super().__init__()
        self.encoder = scout_shared_encoder( n_states,
                                             embed_dim,
                                             n_blocks,
                                             kernel,
                                             drop_rate,
                                             act_fx )
        self.ms2_head = scout_ms2_head( embed_dim, n_ion_channels, n_charges, kernel, act_fx )
        self.irt_head = scout_irt_head( embed_dim, embed_dim, act_fx )
        self.ccs_head = scout_ccs_head( embed_dim, embed_dim, n_charges, act_fx )
        self.charge_dist_head = scout_charge_dist_head( embed_dim, embed_dim, charge_dist_len, act_fx )

    def forward_shared( self, seq, charge, nce ):
        seq_features, pooled_features = self.encoder( seq )
        return { 'seq_features' : seq_features,
                 'pooled_features' : pooled_features }

    def forward( self, seq, charge, nce ):
        shared = self.forward_shared( seq, charge, nce )
        seq_features = shared[ 'seq_features' ]
        pooled_features = shared[ 'pooled_features' ]
        return { 'ms2' : self.ms2_head( seq_features, charge, nce ),
                 'irt' : self.irt_head( pooled_features ),
                 'ccs' : self.ccs_head( pooled_features, charge ),
                 'charge_dist' : self.charge_dist_head( pooled_features ), }


class scout_torchscript_wrapper( nn.Module ):
    def __init__( self, model ):
        super().__init__()
        self.model = model

    def forward( self, seq, charge, nce ):
        outputs = self.model( seq, charge, nce )
        return outputs[ 'ms2' ], outputs[ 'irt' ], outputs[ 'ccs' ], outputs[ 'charge_dist' ]


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
