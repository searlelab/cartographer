
## Pre-process spectra in DLIB for Cartographer training
## USAGE: python script.py input_file.dlib output_file.pkl --n_threads 4 --frag_type hcd --nce 30

import sys, argparse, re, sqlite3, zlib, pickle, concurrent
import concurrent.futures
import numpy as np, pandas as pd
import constants
from masses import modseq_toModDict, ladder_mz_generator
from local_io import read_table
from tensorize import modseq_to_codedseq, mod_regex_keys
#from .masses import *


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        type=str,
                        help='path to input dlib')
    parser.add_argument('output_file',
                        type=str,
                        help='path to output pkl',
                        default=500)
    parser.add_argument('--n_threads',
                        type=int,
                        help='Number of threads for processing',
                        default=4)
    parser.add_argument('--frag_type',
                        type=str,
                        help='hcd or cid',
                        default='cid')
    parser.add_argument('--nce',
                        type=float,
                        help='NCE value, only applicable for hcd',
                        default=30.0)        
    return parser.parse_args(args)



def modseq_renamer( modseq ):
    for mod_id in mod_regex_keys:
        modseq = re.sub( mod_id, mod_regex_keys[mod_id], modseq )
    return modseq



def pad_array( array, n_ion_types=4, ): #peplen, precursor_z, ):
    padded_array = np.full( ( n_ion_types, constants.max_peptide_len-1 ), -1.0, 'float32' )
    for i, row in enumerate( array ):
        padded_array[ i, :len(row)] = row
    return padded_array 




def ion_array_matcher( search_mzs, mzs, ints, resolution='high',):
    array_shape = search_mzs.shape
    search_ints = []
    for s_mz in search_mzs.flatten():
        if s_mz == 0.0: 
            search_ints.append( 0.0 )
        else:
            idx = np.abs(mzs-s_mz).argmin()
            mass_tolerance = constants.return_tolerance( s_mz, resolution, )
            if np.abs(mzs[idx] - s_mz) <= mass_tolerance:
                ion_int = ints[idx]
            else:
                ion_int = 0.0
            search_ints.append( ion_int )
    norm_ints = np.array(search_ints) / ( np.max( search_ints ) + constants.epsilon )
    return np.reshape( norm_ints, array_shape, )



        
def decompress_spectrum( record ):
    mzs = np.ndarray( shape = (int(record['MassEncodedLength']/8)), 
                      dtype = '>d', 
                      buffer = zlib.decompress(record['MassArray']) ).astype( 'float64' )
    ints = np.ndarray( shape = (int(record['IntensityEncodedLength']/4)), 
                       dtype = '>f', 
                       buffer = zlib.decompress(record['IntensityArray']) ).astype( 'float64' )
    return mzs, ints




def dlib_row_parser( record, frag_type, nce, ):
    peptide_len = len( record['PeptideSeq'] ) 
    if peptide_len > constants.max_peptide_len or peptide_len < constants.min_peptide_len:
        print( modseq + ' outside valid size range' )
        return 0
    
    # Extract and label spectra
    modseq = str( record['PeptideModSeq'] )
    codedseq = modseq_to_codedseq( modseq )
    if not codedseq:
         print( 'Invalid modifications in ' + modseq )
         return 0

    key_name = codedseq+'_'+str(record['PrecursorCharge'])
    mzs, ints = decompress_spectrum( record )
    predict_z = np.min( [ record['PrecursorCharge'], 3 ] )
    ion_mzs = ladder_mz_generator( record['PeptideModSeq'], charge=predict_z, phos_loss=True, )
    
    
    norm_ints = ion_array_matcher( ion_mzs, mzs, ints, )
    if np.all( norm_ints == 0.0 ):
        print( 'No ions identified in ' + modseq )
        return 0
    
    RT = record['RTInSeconds']

    results_dict = { 'peptide_mod_seq' : modseq,
                     'precursor_z' : record['PrecursorCharge'],
                     'ion_array' : pad_array( norm_ints,
                                              peptide_len,
                                              record['PrecursorCharge'], ),
                     'frag_type' : frag_type,
                     'nce' : nce,
                     'RT' : RT,
                    }
    return (key_name,results_dict)


def main( ):
    args = parse_args(sys.argv[1:])
    executor = concurrent.futures.ProcessPoolExecutor( args.n_threads )

    df = read_table( args.input_file, 'entries', )
    dlib_columns = list( df.columns )
    futures = [ executor.submit( dlib_row_parser, 
                                 dict(zip(dlib_columns,row)), 
                                 args.frag_type,
                                 args.nce ) 
                for row in df.itertuples( index=False ) ]
    concurrent.futures.wait( futures )
    parsed_data = dict( [ f.result() for f in futures if type(f.result()) != type(0) ] )
    print( '\t' + args.input_file + ':\tNumber of spectra = ' + str(len(parsed_data)) )
    pickle.dump( parsed_data, 
                 open( args.output_file, 'wb' ) )
    


if __name__ == "__main__":
    main()
    sys.exit()
