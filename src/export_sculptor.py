import argparse
import json
import os
import sys

import torch

from sculptor_model import initialize_sculptor_model
from sculptor_settings import charge_dist_len, max_peptide_len
from sculptor_tensorize import aa_to_int, nterm_unimod_map, residue_unimod_map, residues


def parse_dilation_schedule( raw_value ):
    if raw_value is None:
        return None
    value = raw_value.strip()
    if value == '':
        return None
    parts = [ part.strip() for part in value.split( ',' ) if part.strip() != '' ]
    if len( parts ) == 0:
        return None
    try:
        dilations = [ int(part) for part in parts ]
    except ValueError as exc:
        raise ValueError( 'dilation_schedule must be a comma-separated list of integers' ) from exc
    if min( dilations ) < 1:
        raise ValueError( 'dilation_schedule values must be >= 1' )
    return dilations


def build_arch_overrides( args ):
    arch = {}
    if args.embed_dim is not None:
        arch[ 'embed_dim' ] = int( args.embed_dim )
    if args.n_blocks is not None:
        arch[ 'n_blocks' ] = int( args.n_blocks )
    if args.kernel is not None:
        arch[ 'kernel' ] = int( args.kernel )
    if args.block_variant is not None:
        arch[ 'block_variant' ] = args.block_variant
    if args.bottleneck_ratio is not None:
        arch[ 'bottleneck_ratio' ] = float( args.bottleneck_ratio )
    dilation_schedule = parse_dilation_schedule( args.dilation_schedule )
    if dilation_schedule is not None:
        arch[ 'dilation_schedule' ] = dilation_schedule
    if len( arch ) == 0:
        return None
    return arch


def build_preprocessing_metadata( model_file ):
    seq_len = max_peptide_len + 2
    residue_map_json = [ { 'residue' : aa, 'unimod' : unimod, 'token' : token }
                         for (aa, unimod), token in sorted( residue_unimod_map.items() ) ]

    metadata = {
        'model_file'      : os.path.abspath( model_file ),
        'model_family'    : 'sculptor',
        'residues'        : list( residues ),
        'aa_to_int'       : { k : v for k, v in sorted( aa_to_int.items() ) },
        'padding_index'   : 0,
        'max_peptide_len' : max_peptide_len,
        'charge_dist_len' : charge_dist_len,
        'input_names'     : [ 'tokens', 'charge_onehot' ],
        'input_shapes'    : [ [ 'batch', seq_len ], [ 'batch', charge_dist_len ] ],
        'input_dtypes'    : [ 'int64', 'float32' ],
        'output_name'     : 'ccs_norm',
        'output_shape'    : [ 'batch', 1 ],
        'output_dtype'    : 'float32',
        'nterm_unimod_map' : dict( sorted( nterm_unimod_map.items() ) ),
        'residue_unimod_map' : residue_map_json,
    }
    return metadata


def build_example_inputs( batch_size=2 ):
    seq_len = max_peptide_len + 2
    seq = torch.zeros( batch_size, seq_len, dtype=torch.int64 )
    seq[ :, :8 ] = torch.tensor( [ 1, 2, 3, 4, 5, 6, 7, 8 ] )

    charge = torch.zeros( batch_size, charge_dist_len, dtype=torch.float32 )
    charge[ :, 1 ] = 1.0

    return seq, charge


def export_torchscript( model, output_path ):
    model.eval()
    example_inputs = build_example_inputs()
    with torch.no_grad():
        traced = torch.jit.trace( model, example_inputs )
    traced.save( output_path )
    print( 'TorchScript model saved to ' + output_path )
    return traced


def validate_export( model, traced_model, n_tests=5 ):
    model.eval()
    max_diff = 0.0
    seq_len = max_peptide_len + 2

    for _ in range( n_tests ):
        rand_len = torch.randint( 8, seq_len - 1, (1,) ).item()
        seq = torch.zeros( 4, seq_len, dtype=torch.int64 )
        seq[ :, :rand_len ] = torch.randint( 1, len(residues), (4, rand_len) )

        charge = torch.zeros( 4, charge_dist_len, dtype=torch.float32 )
        z = torch.randint( 0, charge_dist_len, (4,) )
        charge[ torch.arange( 4 ), z ] = 1.0

        with torch.no_grad():
            py_out = model( seq, charge )
            ts_out = traced_model( seq, charge )

        diff = torch.abs( py_out - ts_out ).max().item()
        max_diff = max( max_diff, diff )

    print( 'Validation max absolute diff: ' + format( max_diff, '.2e' ) )
    if max_diff > 1e-5:
        print( 'WARNING: TorchScript outputs differ from Python model by more than 1e-5' )
    else:
        print( 'Validation passed' )
    return max_diff


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Export Sculptor model to TorchScript' )
    parser.add_argument( '--model_file',
                         type=str,
                         required=True,
                         help='Path to trained .pt state dict file' )
    parser.add_argument( '--output_dir',
                         type=str,
                         default=None,
                         help='Output directory (default: same as model_file)' )
    parser.add_argument( '--embed_dim', type=int, default=None,
                         help='Override embedding dimension for model construction' )
    parser.add_argument( '--n_blocks', type=int, default=None,
                         help='Override number of residual blocks when dilation_schedule is not provided' )
    parser.add_argument( '--kernel', type=int, default=None,
                         help='Override residual convolution kernel size' )
    parser.add_argument( '--dilation_schedule', type=str, default=None,
                         help='Comma-separated dilation schedule override (e.g., "1,4,8")' )
    parser.add_argument( '--block_variant', type=str, default=None, choices=[ 'full', 'k_only', 'bottleneck' ],
                         help='Residual block variant override' )
    parser.add_argument( '--bottleneck_ratio', type=float, default=None,
                         help='Bottleneck ratio override (used when block_variant=bottleneck)' )
    parser.add_argument( '--skip_validation', action='store_true',
                         help='Skip TorchScript validation' )
    return parser.parse_args( args )


def main():
    args = parse_args( sys.argv[1:] )

    model_file = args.model_file
    if not os.path.isfile( model_file ):
        print( 'ERROR: Model file not found: ' + model_file )
        sys.exit( 1 )

    output_dir = args.output_dir or os.path.dirname( os.path.abspath( model_file ) )
    os.makedirs( output_dir, exist_ok=True )

    base_name = os.path.splitext( os.path.basename( model_file ) )[0]
    ts_path = os.path.join( output_dir, base_name + '.torchscript.pt' )
    json_path = os.path.join( output_dir, base_name + '.preprocessing.json' )
    arch_overrides = build_arch_overrides( args )

    print( 'Loading model from ' + model_file )
    if arch_overrides is not None:
        print( 'Using architecture overrides: ' + json.dumps( arch_overrides ) )
    model = initialize_sculptor_model( model_file=model_file,
                                       arch_overrides=arch_overrides,
                                       map_location='cpu' )
    model.eval()

    traced = export_torchscript( model, ts_path )

    metadata = build_preprocessing_metadata( model_file )
    if arch_overrides is not None:
        metadata[ 'architecture_overrides' ] = arch_overrides
    with open( json_path, 'w' ) as f:
        json.dump( metadata, f, indent=2 )
    print( 'Preprocessing metadata saved to ' + json_path )

    if not args.skip_validation:
        validate_export( model, traced )

    print( 'Export complete' )


if __name__ == '__main__':
    main()
