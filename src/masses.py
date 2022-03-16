import re, itertools
import numpy as np

p_mass = 1.00727646688
masses = { 'A':71.037113805,  'C':103.009184505, 'D':115.026943065, 'E':129.042593135,
           'F':147.068413945, 'G':57.021463735,  'H':137.058911875, 'I':113.084064015,
           'K':128.094963050, 'L':113.084064015, 'M':131.040484645, 'N':114.042927470,
           'P':97.052763875,  'Q':128.058577540, 'R':156.101111050, 'S':87.032028435,
           'T':101.047678505, 'V':99.068413945,  'W':186.079312980, 'Y':163.063328575 }

mod_masses = { 'H2O' : 18.0105647, 'NH3' : 17.0265491,
               'CAM' : 57.0214635, 'Ox' : 15.994915, 'Phospho' : 79.966331,
               'Ac' : 42.010565, 'Me' : 14.01565, 'Succ' : 100.016044, 
               'Ub' : masses['G']*2, 'TMT0' : 224.152478, 'TMT10' : 229.162932, }

# Modified residues
masses['c'] = masses['C'] + mod_masses['CAM']
masses['m'] = masses['M'] + mod_masses['Ox']
masses['d'] = masses['c'] - mod_masses['NH3']
masses['e'] = masses['E'] - mod_masses['H2O']
masses['s'] = masses['S'] + mod_masses['Phospho']
masses['t'] = masses['T'] + mod_masses['Phospho']
masses['y'] = masses['Y'] + mod_masses['Phospho']
masses['a'] = masses['K'] + mod_masses['Ac']
masses['b'] = masses['K'] + mod_masses['Succ']
masses['u'] = masses['K'] + mod_masses['Ub']
masses['n'] = masses['K'] + mod_masses['Me']
masses['o'] = masses['K'] + mod_masses['Me']*2
masses['p'] = masses['K'] + mod_masses['Me']*3
masses['q'] = masses['R'] + mod_masses['Me']
masses['r'] = masses['R'] + mod_masses['Me']*2
masses['z'] = masses['K'] + mod_masses['TMT0']
masses['x'] = masses['K'] + mod_masses['TMT10']



def mass_calc( seq, initial_mass = 18.0105647, mods={}, mod_offset=0 ):
    mass = initial_mass
    for i in range(len(seq)):
        mass += masses[seq[i]]
        if i+1+mod_offset in mods:
            mass += mods[i+1+mod_offset]
    return mass

def mod_str_formatter(seq, mod_dict, start_site, end_site, mod_offset=0):
    output_str = ''
    for site in mod_dict:
        if start_site <= site <= end_site: # ensure that mods are not outside substring
            mod_str = str(int(site-mod_offset)) + seq[site-1-mod_offset] +\
                      format(int(mod_dict[site]), '+d') + ' '
            output_str += mod_str
    return output_str[:-1]

def mod_seq_generator(seq, mod_dict):
    mod_seq = ''
    start = 0
    for i in mod_dict:
        mod = '['+str(mod_dict[i])+']'
        mod_seq = mod_seq + seq[start:i] + mod
        start = i
    mod_seq += seq[start:]
    return mod_seq

def modseq_toModDict(mod_seq):
    mods = {}
    temp_seq = str(mod_seq)
    for i in range(mod_seq.count('[')):
        mod_start_index = temp_seq.find('[')
        mod_end_index = temp_seq.find(']')
        mod_mass = np.float64( temp_seq[mod_start_index+1:mod_end_index] )
        mods[mod_start_index] = mod_mass
        temp_seq = temp_seq[:mod_start_index] + temp_seq[mod_end_index+1:]
    return mods




def fragment_mass_generator(mod_seq, charge=2):
    seq = re.sub(r'\[.+?\]','',mod_seq)
    ## Identify modifications
    mods = modseq_toModDict(mod_seq)

    residue_masses = [ masses[a] for a in seq ]
    for m in mods: residue_masses[m-1] += mods[m]

    peptide_mass = mass_calc( seq, mods=mods )
    frag_masses = {}
    frag_masses['b'] = np.cumsum( residue_masses[:-1] )
    frag_masses['y'] = peptide_mass - frag_masses['b'][::-1]

    mzs = []
    for ion_type, fragment_z in itertools.product( ['b','y'], range(1,charge+1) ):
        mzs += list( (frag_masses[ion_type]+p_mass*fragment_z) / fragment_z )
        
    return mzs

