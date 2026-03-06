import torch
import torch.nn as nn

import sculptor_settings
from core_layers import build_resnet_block
from sculptor_settings import hyperparameters, training_parameters, charge_dist_len
from sculptor_tensorize import residues


class sculptor_model( nn.Module ):
    def __init__( self,
                  vec_length,
                  n_states,
                  embed_dim,
                  n_blocks,
                  kernel,
                  drop_rate,
                  act_fx,
                  n_charges=6,
                  dilation_schedule=None,
                  block_variant='full',
                  bottleneck_ratio=0.5, ):
        super().__init__()
        self.seq_embed = nn.Embedding( n_states, embed_dim, padding_idx=0 )
        torch.nn.init.kaiming_normal_( self.seq_embed.weight, nonlinearity='linear' )

        if dilation_schedule is None:
            dilation_schedule = [ d + 1 for d in range(n_blocks) ]
        else:
            assert len( dilation_schedule ) > 0, 'dilation_schedule must be non-empty'
            dilation_schedule = [ int(d) for d in dilation_schedule ]
            assert min( dilation_schedule ) > 0, 'dilation_schedule values must be >= 1'

        self.resnet_blocks = nn.Sequential( *[ build_resnet_block( block_variant,
                                                                    embed_dim,
                                                                    embed_dim,
                                                                    kernel,
                                                                    d_rate,
                                                                    act_fx,
                                                                    bottleneck_ratio, )
                                                for d_rate in dilation_schedule ] )
        self.dropout = nn.Dropout( drop_rate )
        self.flatten = nn.Flatten()
        self.output = nn.Linear( vec_length * embed_dim + n_charges, 1 )

        nn.init.xavier_normal_( self.output.weight )
        nn.init.constant_( self.output.bias.data, 0.0 )

    def forward( self, seq, charge ):
        x = self.seq_embed( seq ).transpose( 1, -1 )
        x = self.resnet_blocks( x )
        x = self.dropout( x )
        x = self.flatten( x )
        charge = charge.to( dtype=x.dtype )
        x = torch.cat( [ x, charge ], dim=1 )
        return self.output( x )


def initialize_sculptor_model( model_file=None, arch_overrides=None, map_location=None ):
    arch = { 'embed_dim' : hyperparameters[ 'embed_dimension' ],
             'n_blocks' : hyperparameters[ 'n_resnet_blocks' ],
             'kernel' : hyperparameters[ 'kernel_size' ],
             'drop_rate' : training_parameters[ 'dropout_rate' ],
             'act_fx' : hyperparameters[ 'activation_function' ],
             'n_charges' : charge_dist_len,
             'dilation_schedule' : None,
             'block_variant' : 'full',
             'bottleneck_ratio' : 0.5, }

    if arch_overrides is not None:
        alias = { 'kernel_size' : 'kernel' }
        normalized = {}
        for key, value in arch_overrides.items():
            normalized[ alias.get( key, key ) ] = value

        unknown = sorted( set( normalized ) - set( arch ) )
        if len( unknown ) > 0:
            raise ValueError( 'Unknown Sculptor architecture override(s): ' + ', '.join( unknown ) )

        arch.update( normalized )

    model = sculptor_model( sculptor_settings.max_peptide_len + 2,
                            len( residues ) + 1,
                            arch[ 'embed_dim' ],
                            arch[ 'n_blocks' ],
                            arch[ 'kernel' ],
                            arch[ 'drop_rate' ],
                            arch[ 'act_fx' ],
                            arch[ 'n_charges' ],
                            arch[ 'dilation_schedule' ],
                            arch[ 'block_variant' ],
                            arch[ 'bottleneck_ratio' ], )

    if model_file:
        model.load_state_dict( torch.load( model_file, map_location=map_location ), strict=True )

    return model
