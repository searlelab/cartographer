
import re
import numpy as np

import torch
from torch.utils.data import TensorDataset

from constants import max_peptide_len, min_precursor_charge, max_precursor_charge

residues = [ 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',

             'c', 'm', 'd', 'e', 's', 't', 'y', 'a', 'b', 'u',
             'n', 'o', 'p', 'q', 'r', 'x', 'z',
             'w', 'f', 'g', 'h', 'i', 'j', 'k',

             '-', '^', '(', ')', '&', '*', '_', ]

n_states = len(residues)+1

aa_to_int = dict(zip(residues,range(1,len(residues)+1)))


mod_regex_keys = { r'M\[\+15\.99.{,6}\]':'m', r'C\[\+57\.02.{,6}\]':'c',
                   r'C\[\+39\.99.{,6}\]':'d', r'E\[\-18\.01.{,6}\]':'e', r'Q\[\-17\.02.{,6}\]':'e',
                   r'S\[\+79\.96.{,6}\]':'s', r'T\[\+79\.96.{,6}\]':'t', r'Y\[\+79\.96.{,6}\]':'y',
                   r'K\[\+42\.01.{,6}\]':'a', r'K\[\+100\.0.{,6}\]':'b', r'K\[\+114\.0.{,6}\]':'u',
                   r'K\[\+14\.01.{,6}\]':'n', r'K\[\+28\.03.{,6}\]':'o', r'K\[\+42\.04.{,6}\]':'p',
                   r'R\[\+14\.01.{,6}\]':'q', r'R\[\+28\.03.{,6}\]':'r',
                   r'W\[\+15\.99.{,6}\]':'w',
                   r'N\[\+0\.98.{,6}\]':'f',  r'Q\[\+0\.98.{,6}\]':'g',  r'R\[\+0\.98.{,6}\]':'k',
                   r'N\[\+203\.0.{,6}\]':'h', r'S\[\+203\.0.{,6}\]':'i', r'T\[\+203\.0.{,6}\]':'j',
                   r'K\[\+224\.1.{,6}\]':'z', r'K\[\+229\.1.{,6}\]':'x',
                }

nterm_keys = { '+42.01' : '^', '+224.1' : '&', '+229.1' : '*',
               '-17.02' : '(', '-18.01' : ')' }
## pyroglu: ( = pyroglu-Q, ) = pyroglu-E;  cyclocys = )d

# UNIMOD-based mapping tables for Prospect-PTMs parquet data
nterm_unimod_map = {
	'UNIMOD:1'  : '^',   # Acetyl
	'UNIMOD:737': '*',   # TMT6plex
	'UNIMOD:739': '&',   # TMT0
	'UNIMOD:28' : '(',   # Gln->pyro-Glu (pyroglu Q)
	'UNIMOD:27' : ')',   # Glu->pyro-Glu (pyroglu E)
}

residue_unimod_map = {
	('C', 'UNIMOD:4')  : 'c',   # Carbamidomethyl
	('M', 'UNIMOD:35') : 'm',   # Oxidation
	('W', 'UNIMOD:35') : 'w',   # Oxidation
	('S', 'UNIMOD:21') : 's',   # Phospho
	('T', 'UNIMOD:21') : 't',   # Phospho
	('Y', 'UNIMOD:21') : 'y',   # Phospho
	('K', 'UNIMOD:1')  : 'a',   # Acetyl
	('K', 'UNIMOD:64') : 'b',   # Succinyl (mapped from original mass)
	('K', 'UNIMOD:121'): 'u',   # GlyGly (ubiquitin remnant)
	('K', 'UNIMOD:34') : 'n',   # Methyl
	('K', 'UNIMOD:36') : 'o',   # Dimethyl
	('K', 'UNIMOD:37') : 'p',   # Trimethyl
	('R', 'UNIMOD:34') : 'q',   # Methyl
	('R', 'UNIMOD:36') : 'r',   # Dimethyl
	('K', 'UNIMOD:739'): 'z',   # TMT0
	('K', 'UNIMOD:737'): 'x',   # TMT6plex
	('N', 'UNIMOD:7')  : 'f',   # Deamidated
	('Q', 'UNIMOD:7')  : 'g',   # Deamidated
	('R', 'UNIMOD:7')  : 'k',   # Deamidated
	('N', 'UNIMOD:43') : 'h',   # HexNAc
	('S', 'UNIMOD:43') : 'i',   # HexNAc
	('T', 'UNIMOD:43') : 'j',   # HexNAc
}

def modseq_to_codedseq( seq ):
    for mod_id in mod_regex_keys:
        seq = re.sub( mod_id, mod_regex_keys[mod_id], seq )
        
    # N/C mods
    if seq[0] == 'd' : seq = ')' + seq
    elif seq[0] == 'e' : seq = '(' + seq
    elif seq[0] == '[' : seq = nterm_keys[ seq[1:7] ] + seq[ seq.find(']')+1: ]
    else: seq = '-'+seq
    seq = seq+'_'    
    
    # Ensure that there are no additional modifications
    if seq.count('[') > 0:
        return None
    else:
        return seq


def unimod_to_codedseq( modified_sequence, max_len=max_peptide_len, skip_counts=None ):
    """Convert UNIMOD-annotated sequence (e.g. [UNIMOD:1]-PEPTM[UNIMOD:35]IDE-[]) to coded sequence.
    If skip_counts (a Counter or dict) is provided, skip reasons are tallied there instead of printed."""
    parts = modified_sequence.split('-', 2)
    if len(parts) != 3:
        if skip_counts is not None: skip_counts['unexpected_format'] = skip_counts.get('unexpected_format', 0) + 1
        return None

    nterm_part, body, cterm_part = parts

    # N-terminal
    if nterm_part == '[]' or nterm_part == '':
        nterm_char = '-'
    else:
        nterm_tag = nterm_part.strip('[]')
        if nterm_tag in nterm_unimod_map:
            nterm_char = nterm_unimod_map[nterm_tag]
        else:
            if skip_counts is not None: skip_counts['nterm:' + nterm_tag] = skip_counts.get('nterm:' + nterm_tag, 0) + 1
            return None

    # C-terminal (currently always unmodified)
    cterm_char = '_'

    # Body
    coded_body = ''
    i = 0
    while i < len(body):
        aa = body[i]
        i += 1
        if i < len(body) and body[i] == '[':
            end = body.index(']', i)
            unimod_tag = body[i+1:end]
            i = end + 1
            # Check for stacked mods (another bracket immediately follows)
            if i < len(body) and body[i] == '[':
                if skip_counts is not None: skip_counts['stacked_mods'] = skip_counts.get('stacked_mods', 0) + 1
                return None
            # First body residue: fold N-terminal cyclization mods onto nterm
            if len(coded_body) == 0 and nterm_char == '-':
                if aa == 'Q' and unimod_tag == 'UNIMOD:28':
                    nterm_char = '('
                    coded_body += aa
                    continue
                elif aa == 'E' and unimod_tag == 'UNIMOD:27':
                    nterm_char = ')'
                    coded_body += aa
                    continue
            key = (aa, unimod_tag)
            if key in residue_unimod_map:
                coded_body += residue_unimod_map[key]
            else:
                reason = aa + '[' + unimod_tag + ']'
                if skip_counts is not None: skip_counts[reason] = skip_counts.get(reason, 0) + 1
                return None
        else:
            coded_body += aa

    if len(coded_body) > max_len:
        if skip_counts is not None: skip_counts['too_long'] = skip_counts.get('too_long', 0) + 1
        return None

    return nterm_char + coded_body + cterm_char


def codedseq_to_array(seq, max_size=max_peptide_len+2):
    seq_by_int = [aa_to_int[seq[i]] for i in range(len(seq))]
    seq_by_int += [0]*(max_size - len(seq_by_int))
    return np.asarray( seq_by_int, 'int64' )


def hi_db_to_tensors( hi_db ):
    seq_array = np.asarray( [ codedseq_to_array( p ) for p in hi_db.CodedPeptideSeq ],
                            'int64', )
    hi_array = np.asarray( [ [ hi ] for hi in hi_db.HI ], 'float32', )
    sources = sorted( set( hi_db.Source ) )
    source_array = np.asarray( [ [ float( s == sx ) for sx in sources ] for s in hi_db.Source ],
                               'float32', )
    tensors = [ torch.Tensor( x ) for x in [ seq_array, hi_array, source_array] ]
    tensors[0] = tensors[0].to(torch.int64) # Need to ensure seq tensor are ints for embedding layer
    
    return TensorDataset( *tensors )
    
    
def return_charge_array( precursor_charge, batch_size, ):
    charge_ohe = [ precursor_charge == z for z in range( min_precursor_charge, 
                                                         max_precursor_charge+1, ) ]
    return np.array( [ charge_ohe ] * batch_size )
    

