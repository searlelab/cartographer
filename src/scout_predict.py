import argparse
import json
import os
import sys

import numpy as np
import torch

from scout_model import initialize_scout_model
from scout_settings import max_peptide_len
from tensorize import codedseq_to_array, return_charge_array, unimod_to_codedseq
from training_loop import resolve_device


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Predict MS2, iRT, and CCS with Scout' )
    parser.add_argument( '--model_file', type=str, required=True,
                         help='Path to trained Scout checkpoint (.pt)' )
    parser.add_argument( '--modified_sequence', type=str, required=True,
                         help='Modified sequence in Cartographer UNIMOD format, e.g. []-PEPTIDE-[]' )
    parser.add_argument( '--precursor_charge', type=int, required=True,
                         help='Precursor charge state' )
    parser.add_argument( '--nce', type=float, required=True,
                         help='Aligned normalized NCE value, e.g. 0.30 for NCE 30' )
    parser.add_argument( '--device', type=str, default='auto',
                         help='Inference device {auto,mps,cuda,cpu}' )
    return parser.parse_args( args )


def load_scalar_stats( model_file ):
    metadata_file = model_file + '.metadata.json'
    if not os.path.isfile( metadata_file ):
        return { 'irt_mean' : 0.0, 'irt_std' : 1.0, 'ccs_mean' : 0.0, 'ccs_std' : 1.0 }
    with open( metadata_file, 'r' ) as handle:
        payload = json.load( handle )
    return payload.get( 'scalar_stats', { 'irt_mean' : 0.0, 'irt_std' : 1.0, 'ccs_mean' : 0.0, 'ccs_std' : 1.0 } )


def main():
    args = parse_args( sys.argv[1:] )
    device = resolve_device( args.device )

    coded = unimod_to_codedseq( args.modified_sequence, max_len=max_peptide_len )
    if coded is None:
        print( 'ERROR: Unsupported modified sequence for Scout tokenizer: ' + args.modified_sequence )
        sys.exit( 1 )

    model = initialize_scout_model( model_file=args.model_file, map_location=device ).to( device )
    model.eval()
    scalar_stats = load_scalar_stats( args.model_file )

    seq_tensor = torch.as_tensor( np.asarray( [ codedseq_to_array( coded, max_size=max_peptide_len + 2 ) ], dtype='int64' ),
                                  dtype=torch.long,
                                  device=device )
    charge_tensor = torch.as_tensor( return_charge_array( args.precursor_charge, 1 ),
                                     dtype=torch.float32,
                                     device=device )
    nce_tensor = torch.as_tensor( np.asarray( [ [ float( args.nce ) ] ], dtype='float32' ),
                                  dtype=torch.float32,
                                  device=device )

    with torch.no_grad():
        outputs = model( seq_tensor, charge_tensor, nce_tensor )

    irt = float( outputs[ 'irt' ][0, 0].cpu().item() * scalar_stats[ 'irt_std' ] + scalar_stats[ 'irt_mean' ] )
    ccs = float( outputs[ 'ccs' ][0, 0].cpu().item() * scalar_stats[ 'ccs_std' ] + scalar_stats[ 'ccs_mean' ] )
    ms2 = outputs[ 'ms2' ][0].cpu().numpy().tolist()

    print( json.dumps( { 'modified_sequence' : args.modified_sequence,
                         'precursor_charge' : int( args.precursor_charge ),
                         'nce' : float( args.nce ),
                         'irt' : irt,
                         'ccs' : ccs,
                         'ms2' : ms2 }, indent=2 ) )


if __name__ == '__main__':
    main()
    sys.exit()
