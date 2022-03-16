
epsilon = 1e-7
train_fdr = 0.01

min_peptide_len = 6
max_peptide_len = 35


chronologer_db_loc = '../data/Chronologer_DB_220308.gz'

seed = 2447

validation_fraction = 0.2


mass_tolerances = { 'hcd' : 20e-6, #ppm
                    'cid' : 0.35, #amu
                  }

def return_tolerance( mass, frag_type ):
    assert frag_type in mass_tolerances, 'INVALID FRAGMENTATION TYPE'
    if frag_type == 'hcd':
        tol = mass_tolerances['hcd'] * mass
    else:
        tol = mass_tolerances['cid']
    return tol

