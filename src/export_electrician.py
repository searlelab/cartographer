
import os, sys, json, argparse
from datetime import datetime

import numpy as np
import torch

from electrician_model import initialize_electrician_model
from electrician_settings import max_peptide_len, charge_dist_len
from tensorize import residues, aa_to_int, mod_regex_keys, nterm_keys


def build_mod_regex_rules():
	"""Build the full list of mod_regex_rules for the preprocessing JSON."""
	rules = []
	for pattern, token in mod_regex_keys.items():
		rules.append( { 'pattern': pattern, 'token': token } )
	return rules


def build_preprocessing_metadata( model_file ):
	"""Build the preprocessing metadata dict for the Electrician model."""
	seq_len = max_peptide_len + 2

	metadata = {
		'model_file'          : os.path.abspath( model_file ),
		'model_family'        : 'electrician',
		'residues'            : list( residues ),
		'aa_to_int'           : { k: v for k, v in sorted( aa_to_int.items() ) },
		'padding_index'       : 0,
		'max_peptide_len'     : max_peptide_len,
		'input_names'         : [ 'tokens' ],
		'input_shapes'        : [ [ 'batch', seq_len ] ],
		'input_dtypes'        : [ 'int64' ],
		'output_name'         : 'charge_state_dist',
		'output_shape'        : [ 'batch', charge_dist_len ],
		'output_dtype'        : 'float32',
		'mod_regex_rules'     : build_mod_regex_rules(),
		'nterm_keys'          : dict( nterm_keys ),
	}
	return metadata


def build_example_inputs( batch_size=2 ):
	"""Build example inputs for tracing the model."""
	seq_len = max_peptide_len + 2

	seq   = torch.zeros( batch_size, seq_len, dtype=torch.int64 )
	seq[ :, :8 ] = torch.tensor( [1, 2, 3, 4, 5, 6, 7, 8] )  # dummy tokens

	return ( seq, )


def export_torchscript( model, output_path ):
	"""Trace the model and save as TorchScript."""
	model.eval()
	example_inputs = build_example_inputs()

	with torch.no_grad():
		traced = torch.jit.trace( model, example_inputs )

	traced.save( output_path )
	print( 'TorchScript model saved to ' + output_path )
	return traced


def validate_export( model, traced_model, n_tests=5 ):
	"""Compare Python model outputs against TorchScript outputs."""
	model.eval()
	max_diff = 0.0

	for i in range( n_tests ):
		seq_len = max_peptide_len + 2
		rand_len = torch.randint( 8, seq_len - 1, (1,) ).item()
		seq = torch.zeros( 4, seq_len, dtype=torch.int64 )
		seq[ :, :rand_len ] = torch.randint( 1, len(residues), (4, rand_len) )

		with torch.no_grad():
			py_out = model( seq )
			ts_out = traced_model( seq )

		diff = torch.abs( py_out - ts_out ).max().item()
		max_diff = max( max_diff, diff )

	print( 'Validation max absolute diff: ' + format( max_diff, '.2e' ) )
	if max_diff > 1e-5:
		print( 'WARNING: TorchScript outputs differ from Python model by more than 1e-5' )
	else:
		print( 'Validation passed' )

	return max_diff


def parse_args( args ):
	parser = argparse.ArgumentParser( description='Export Electrician model to TorchScript' )
	parser.add_argument( '--model_file', type=str, required=True,
	                     help='Path to trained .pt state dict file' )
	parser.add_argument( '--output_dir', type=str, default=None,
	                     help='Output directory (default: same as model_file)' )
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

	# Derive base name from model file (e.g. Electrician_20260218164940)
	base_name = os.path.splitext( os.path.basename( model_file ) )[0]
	ts_path   = os.path.join( output_dir, base_name + '.torchscript.pt' )
	json_path = os.path.join( output_dir, base_name + '.preprocessing.json' )

	# Load model
	print( 'Loading model from ' + model_file )
	model = initialize_electrician_model( model_file=model_file )
	model.eval()

	# Export TorchScript
	traced = export_torchscript( model, ts_path )

	# Write preprocessing metadata
	metadata = build_preprocessing_metadata( model_file )
	with open( json_path, 'w' ) as f:
		json.dump( metadata, f, indent=2 )
	print( 'Preprocessing metadata saved to ' + json_path )

	# Validate
	if not args.skip_validation:
		validate_export( model, traced )

	print( 'Export complete' )


if __name__ == '__main__':
	main()
