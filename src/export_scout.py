import argparse
import json
import os
import sys

import torch

from scout_model import initialize_scout_model, scout_torchscript_wrapper
from scout_settings import max_peptide_len, ms2_vector_len
from tensorize import aa_to_int, nterm_unimod_map, residue_unimod_map, residues


def build_preprocessing_metadata( model_file ):
    metadata_file = model_file + '.metadata.json'
    scalar_stats = { 'irt_mean' : 0.0, 'irt_std' : 1.0, 'ccs_mean' : 0.0, 'ccs_std' : 1.0 }
    if os.path.isfile( metadata_file ):
        with open( metadata_file, 'r' ) as handle:
            payload = json.load( handle )
        scalar_stats.update( payload.get( 'scalar_stats', {} ) )

    residue_map_json = [ { 'residue' : aa, 'unimod' : unimod, 'token' : token }
                         for ( aa, unimod ), token in sorted( residue_unimod_map.items() ) ]

    return { 'model_file' : os.path.abspath( model_file ),
             'model_family' : 'scout',
             'padding_index' : 0,
             'residues' : list( residues ),
             'aa_to_int' : { key : value for key, value in sorted( aa_to_int.items() ) },
             'nterm_unimod_map' : dict( sorted( nterm_unimod_map.items() ) ),
             'residue_unimod_map' : residue_map_json,
             'max_peptide_len' : max_peptide_len,
             'input_names' : [ 'tokens', 'precursor_charge_onehot', 'nce' ],
             'input_shapes' : [ [ 'batch', max_peptide_len + 2 ],
                                [ 'batch', 6 ],
                                [ 'batch', 1 ] ],
             'input_dtypes' : [ 'int64', 'float32', 'float32' ],
             'output_names' : [ 'ms2', 'irt', 'ccs' ],
             'output_shapes' : [ [ 'batch', ms2_vector_len ],
                                 [ 'batch', 1 ],
                                 [ 'batch', 1 ] ],
             'output_dtypes' : [ 'float32', 'float32', 'float32' ],
             'scalar_stats' : scalar_stats,
             'ms2_ion_order' : 'y1(1+), y1(2+), b1(1+), b1(2+), y2(1+), y2(2+), b2(1+), b2(2+), ...', }


def build_example_inputs( batch_size=2 ):
    seq = torch.zeros( batch_size, max_peptide_len + 2, dtype=torch.int64 )
    seq[ :, :8 ] = torch.tensor( [ 1, 2, 3, 4, 5, 6, 7, 8 ] )

    charge = torch.zeros( batch_size, 6, dtype=torch.float32 )
    charge[ :, 1 ] = 1.0

    nce = torch.full( ( batch_size, 1 ), 0.30, dtype=torch.float32 )
    return seq, charge, nce


def export_torchscript( model, output_path ):
    model.eval()
    wrapper = scout_torchscript_wrapper( model )
    example_inputs = build_example_inputs()
    with torch.no_grad():
        traced = torch.jit.trace( wrapper, example_inputs )
    traced.save( output_path )
    print( 'TorchScript model saved to ' + output_path )
    return traced


def validate_export( model, traced_model, n_tests=5 ):
    model.eval()
    max_diff = 0.0
    for _ in range( n_tests ):
        seq, charge, nce = build_example_inputs( batch_size=4 )
        rand_len = torch.randint( 8, max_peptide_len + 1, ( 1, ) ).item()
        seq[ :, :rand_len ] = torch.randint( 1, len( residues ), ( 4, rand_len ) )
        seq[ :, rand_len: ] = 0

        with torch.no_grad():
            py_out = model( seq, charge, nce )
            ts_out = traced_model( seq, charge, nce )

        diffs = [ torch.abs( py_out[ 'ms2' ] - ts_out[0] ).max().item(),
                  torch.abs( py_out[ 'irt' ] - ts_out[1] ).max().item(),
                  torch.abs( py_out[ 'ccs' ] - ts_out[2] ).max().item() ]
        max_diff = max( [ max_diff ] + diffs )

    print( 'Validation max absolute diff: ' + format( max_diff, '.2e' ) )
    if max_diff > 1e-5:
        print( 'WARNING: TorchScript outputs differ from Python model by more than 1e-5' )
    else:
        print( 'Validation passed' )
    return max_diff


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Export Scout model to TorchScript' )
    parser.add_argument( '--model_file', type=str, required=True,
                         help='Path to trained .pt state dict file' )
    parser.add_argument( '--output_dir', type=str, default=None,
                         help='Output directory (default: same as model_file)' )
    parser.add_argument( '--skip_validation', action='store_true',
                         help='Skip TorchScript validation' )
    return parser.parse_args( args )


def main():
    args = parse_args( sys.argv[1:] )
    if not os.path.isfile( args.model_file ):
        print( 'ERROR: Model file not found: ' + args.model_file )
        sys.exit( 1 )

    output_dir = args.output_dir or os.path.dirname( os.path.abspath( args.model_file ) )
    os.makedirs( output_dir, exist_ok=True )

    base_name = os.path.splitext( os.path.basename( args.model_file ) )[0]
    ts_path = os.path.join( output_dir, base_name + '.torchscript.pt' )
    json_path = os.path.join( output_dir, base_name + '.preprocessing.json' )

    print( 'Loading model from ' + args.model_file )
    model = initialize_scout_model( model_file=args.model_file, map_location='cpu' )
    traced = export_torchscript( model, ts_path )

    metadata = build_preprocessing_metadata( args.model_file )
    with open( json_path, 'w' ) as handle:
        json.dump( metadata, handle, indent=2 )
    print( 'Preprocessing metadata saved to ' + json_path )

    if not args.skip_validation:
        validate_export( model, traced )

    print( 'Export complete' )


if __name__ == '__main__':
    main()
