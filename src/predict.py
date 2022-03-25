
import time, datetime, itertools, struct, zlib
import concurrent
import concurrent.futures
import numpy as np, pandas as pd
import torch
from pyteomics.parser import cleave, expasy_rules
import constants 
from tensorize import residues, modseq_to_codedseq, codedseq_to_array, return_charge_array
from masses import mod_masses, mod_seq_generator, modseq_to_seq, mass_calc, p_mass, ladder_mz_generator
from cartographer_model import initialize_cartographer_model
from chronologer_model import initialize_chronologer_model
from local_io import read_fasta, create_dlib, append_table_to_dlib
#from constants import default_cartographer_model, default_chronologer_model

def timer( start, ):
    return str( datetime.timedelta( seconds=round( time.time() - start ) ) )

def peptide_generator( seq, protease = 'trypsin', miscleavages = 1, ):
    peptides = cleave( seq.upper(), 
                       expasy_rules[ protease ], 
                       missed_cleavages = miscleavages, )
    # Limit peptides based on min/max length and only canonical amino acids
    peptides = set( [ p for p in peptides 
                      if len(p) >= constants.min_peptide_len and 
                         len(p) <= constants.max_peptide_len and
                         set(residues[:20]) | set(p) == set(residues[:20]) ] )
    return sorted( peptides )


def generate_peptidetoprotein( protein_dict, decoys=False, ):
    dfs = []
    for accession, protein in protein_dict.items():
        peptides = peptide_generator( protein, )
        df = pd.DataFrame( { 'PeptideSeq' : peptides,
                             'isDecoy' : False,
                             'ProteinAccession' : accession, } )
        dfs.append( df )
        if decoys:
            peptides = [ p[0]+p[1:-1][::-1]+p[-1] for p in peptides ]
            df = pd.DataFrame( { 'PeptideSeq' : peptides,
                                 'isDecoy' : True,
                                 'ProteinAccession' : accession, } )
            dfs.append( df )
    return pd.concat( dfs )


def apply_mods( peptides, fixed_mods, var_mods, max_n_var_mods, ):
    modified_peptides = []
    for peptide in peptides:
        fixed_mod_list = [ ( i+1, fixed_mods[r] ) 
                           for i, r in enumerate(peptide) 
                           if r in fixed_mods ]
        var_mod_list = [ ( i+1, var_mods[r] ) 
                         for i, r in enumerate(peptide) 
                         if r in var_mods ]
        var_mod_lists = [ s for n in range(1, max_n_var_mods+1) 
                          for s in itertools.combinations( var_mods.items(), n ) ]
        for var_mod_list in [[]] + var_mod_lists:
            mod_dict = dict( fixed_mod_list + var_mod_list )
            modified_peptides.append( mod_seq_generator( peptide, mod_dict, ) )
    return modified_peptides
    
def scrub_masked_ions( modseqs, predictions, ):
    n_ions = [ len(modseq_to_seq(p))-1 for p in modseqs ]
    scrubbed_predictions = [ predictions[ i ][ : , :n ] for i,n in enumerate(n_ions) ]
    return scrubbed_predictions
    
def compress_mz_intensity( mzs, intensities, ):
    df = pd.DataFrame( { 'mz' : mzs.flatten(), 
                         'intensity' : intensities.flatten(), } )
    df = df[ df.intensity >= 0.001 ]
    ba_mzs = bytearray()
    ba_intensities = bytearray()
    for mz in df.mz: ba_mzs += bytearray(struct.pack('>d', mz))
    for i in df.intensity: ba_intensities += bytearray(struct.pack('>f', i))
    values = { 'MassArray' : zlib.compress(ba_mzs),
               'MassEncodedLength' : len(df) * 8,
               'IntensityArray' : zlib.compress(ba_intensities),
               'IntensityEncodedLength' : len(df) * 4, }
    return values
    


    

def build_library( fasta_file, 
                   output_dlib,
                   fixed_modifications = { 'C' : mod_masses['CAM'] },
                   variable_modifications = {},
                   max_variable_mods = 2,
                   precursor_charges = [ 2, 3, ],
                   nce = 30.0, # Can be either a single float or list of NCEs for each precursor charge
                   cartographer_model_file = constants.default_cartographer_model_file, 
                   chronologer_model_file = constants.default_chronologer_model_file,
                   beam_fragmentation = True,
                   batch_size = 2048,
                   n_threads = 8,
                   device = 'cuda',
                   print_checkpoint = 100000,
                   add_decoys = False, ):
    
    start_time = time.time()
    
    executor = concurrent.futures.ProcessPoolExecutor( n_threads )
    
    
    # Check NCE data type, and if a float, expand to a list to match precursor charges
    if type(nce) == float: 
        nce = [ nce ] * len( precursor_charges )
    else:
        assert len(nce) == len(precursor_charges), 'Mismatch in # of NCEs and Precursor Charges'
    
    # Create dlib
    create_dlib( output_dlib, )
    
    # Load models
    cartographer = initialize_cartographer_model( 'beam' if beam_fragmentation else 'resonance', 
                                                  cartographer_model_file, ).to( device )
    chronologer = initialize_chronologer_model( chronologer_model_file, ).to( device )
    
    print( 'Cartographer and Chronologer models loaded (' + timer(start_time) + ')' )
    
    # Read in protein sequences, extract peptides, apply modifications, and tensorize
    protein_sequences = read_fasta( fasta_file, )
    peptide_to_protein = generate_peptidetoprotein( protein_sequences, add_decoys, )
    
    ## NEED TO GET PEPTIDETOPROTEIN TABLE INTO DLIB ##
    unique_peptides = sorted( set( peptide_to_protein.PeptideSeq ) )
    modified_peptides = apply_mods( unique_peptides, 
                                    fixed_modifications, 
                                    variable_modifications,
                                    max_variable_mods, )
    ###
    ### SHOULD ADD SOME KIND OF CHECK THAT INPUT MODS ARE VALID
    ### It will be easier if I set this earlier rather than filter in coded_peptides
    ###
    peptide_batches = [ modified_peptides[ i : i + batch_size ] 
                        for i in range( 0, len(modified_peptides), batch_size ) ]
    print( 'Peptides generated, n = ' + str(len(modified_peptides)) + ' (' + timer(start_time) + ')' )
    n_peptides_processed = 0
    checkpoint = print_checkpoint
    for peptide_batch in peptide_batches:
        n_peptides = len( peptide_batch )
        entries = pd.DataFrame( { 'PeptideModSeq' : peptide_batch, 'SourceFile' : 'Cartographer_alpha' } )
        entries[ 'PeptideSeq' ] = [ modseq_to_seq( p ) for p in peptide_batch ]
        
        coded_peptides = [ modseq_to_codedseq( p ) for p in peptide_batch ]
        peptide_array = np.array( [ codedseq_to_array( p ) for p in coded_peptides ] )
        peptide_tensor = torch.Tensor( peptide_array ).to( torch.int64 ).to( device )
        
        entries['RTInSeconds'] = chronologer( peptide_tensor ).cpu().T[0].detach().numpy()
        
        peptide_masses = np.array( [ mass_calc( p ) for p in peptide_batch ] )
        
        for i, precursor_charge in enumerate(precursor_charges):
            entries[ 'PrecursorCharge' ] = precursor_charge
            entries[ 'PrecursorMz' ] = ( peptide_masses + precursor_charge*p_mass ) / precursor_charge
            
            charge_tensor = torch.Tensor( return_charge_array( precursor_charge, 
                                                               n_peptides ) ).to( device )
            cartographer_inputs = [ peptide_tensor, charge_tensor ]
            if beam_fragmentation:
                nce_tensor = torch.full( ( n_peptides, 1 ), nce[i] ).to( device )
                cartographer_inputs.append( nce_tensor )

            predictions = cartographer( *cartographer_inputs ).cpu().detach().numpy()
            ladder_intensities = scrub_masked_ions( peptide_batch, predictions )
            
            futures = [ executor.submit( ladder_mz_generator, p, precursor_charge, 2, False, )
                        for p in peptide_batch ]
            concurrent.futures.wait(futures)
            ladder_mzs = [ f.result() for f in futures ]
            
            
            futures = [ executor.submit( compress_mz_intensity, mz, intensities ) 
                        for mz, intensities in zip(ladder_mzs, ladder_intensities) ]
            concurrent.futures.wait(futures)
            compressed_results = [ f.result() for f in futures ]
            
            for key in compressed_results[0]: # Grab new columns from keys of first entry
                values = [ results[key] for results in compressed_results ]
                entries[ key ] = values
                        
    
            append_table_to_dlib( entries, 'entries', output_dlib )
        n_peptides_processed += n_peptides
        if n_peptides_processed >= checkpoint:
            print( '\t' + str(n_peptides_processed) + ' peptides processed (' + timer(start_time) + ')' )
            checkpoint += print_checkpoint
            
    print( 'DLIB generation complete (' + timer(start_time) + ')' )
        

    
    