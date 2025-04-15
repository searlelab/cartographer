
epsilon = 1e-7
train_fdr = 0.01

min_peptide_len = 6
max_peptide_len = 35

min_precursor_charge = 1
max_precursor_charge = 6


chronologer_db_loc = '../data/Chronologer_DB_220308.gz'
cartographer_ptdata_loc = '../data/Cartographer_PT/'

default_cartographer_model_file = '../models/Cartographer_20220319002135.pt'
default_chronologer_model_file = '../models/Chronologer_20220317200246.pt'

seed = 2447

validation_fraction = 0.2


mass_tolerances = { 'high' : 20e-6, #ppm
                    'low' : 0.35, #amu
                  }

def return_tolerance( mass, resolution ):
    assert resolution in mass_tolerances, 'INVALID FRAGMENTATION TYPE'
    if resolution == 'high':
        tol = mass_tolerances['high'] * mass
    else:
        tol = mass_tolerances['low']
    return tol

