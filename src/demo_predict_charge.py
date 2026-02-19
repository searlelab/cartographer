import os, sys, argparse
import numpy as np

import torch

from electrician_model import initialize_electrician_model
from electrician_settings import max_peptide_len
from tensorize import unimod_to_codedseq, codedseq_to_array


TEST_PEPTIDES = [ '[]-HC[UNIMOD:4]VDPAVIAAIISR-[]',
                  '[]-TLLISSLSPALPAEHLEDR-[]',
                  '[]-TPIGSFLGSLS-[]',
                  '[]-ANAEKTSGSNVKIVKVKKE-[]', ]


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Predict charge state distributions for test peptides' )
    parser.add_argument( '--model_file', type=str, required=True,
                         help='Path to trained Electrician .pt state dict file' )
    return parser.parse_args( args )


def predict_charge( model_file, peptides=TEST_PEPTIDES ):
    model = initialize_electrician_model( model_file=model_file )
    model.eval()

    seq_size = max_peptide_len + 2
    seq_arrays = []
    for pep in peptides:
        coded = unimod_to_codedseq( pep, max_len=max_peptide_len )
        assert coded is not None, 'Failed to encode: ' + pep
        seq_arrays.append( codedseq_to_array( coded, max_size=seq_size ) )

    seq_t = torch.from_numpy( np.array( seq_arrays, 'int64' ) )

    with torch.no_grad():
        pred = model( seq_t )

    return pred.cpu().numpy()


def print_table( peptides, predictions ):
    header = [ 'Peptide', '+1', '+2', '+3', '+4', '+5', '+6' ]
    rows = []
    for i, pep in enumerate( peptides ):
        row = [ pep ] + [ format( v, '.4f' ) for v in predictions[i] ]
        rows.append( row )

    # Compute column widths
    widths = [ max( len(header[c]), max( len(rows[r][c]) for r in range(len(rows)) ) )
               for c in range( len(header) ) ]

    def format_row( cells ):
        return '  '.join( cells[c].ljust( widths[c] ) for c in range( len(cells) ) )

    print( format_row( header ) )
    print( '  '.join( '-' * w for w in widths ) )
    for row in rows:
        print( format_row( row ) )


def main():
    args = parse_args( sys.argv[1:] )

    if not os.path.isfile( args.model_file ):
        print( 'ERROR: Model file not found: ' + args.model_file )
        sys.exit( 1 )

    print( 'Model: ' + args.model_file )
    print()

    predictions = predict_charge( args.model_file, TEST_PEPTIDES )
    print_table( TEST_PEPTIDES, predictions )


if __name__ == '__main__':
    main()
