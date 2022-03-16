
import re
import numpy as np

import torch
from torch.utils.data import TensorDataset

from constants import max_peptide_len

residues = [ 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
             
             'c', 'm', 'd', 'e', 's', 't', 'y', 'a', 'b', 'u', 
             'n', 'o', 'p', 'q', 'r', 'x', 'z',
             
             '-', '^', '(', ')', '&', '*', '_', ]

n_states = len(residues)+1

aa_to_int = dict(zip(residues,range(1,len(residues)+1)))


mod_regex_keys = { r'M\[\+15\.99.{,6}\]':'m', r'C\[\+57\.02.{,6}\]':'c', 
                   r'C\[\+39\.99.{,6}\]':'d', r'E\[\-18\.01.{,6}\]':'e', r'Q\[\-17\.02.{,6}\]':'e',
                   r'S\[\+79\.96.{,6}\]':'s', r'T\[\+79\.96.{,6}\]':'t', r'Y\[\+79\.96.{,6}\]':'y',
                   r'K\[\+42\.01.{,6}\]':'a', r'K\[\+100\.0.{,6}\]':'b', r'K\[\+114\.0.{,6}\]':'u', 
                   r'K\[\+14\.01.{,6}\]':'n', r'K\[\+28\.03.{,6}\]':'o', r'K\[\+42\.04.{,6}\]':'p',
                   r'R\[\+14\.01.{,6}\]':'q', r'R\[\+28\.03.{,6}\]':'r', 
                }

nterm_keys = { '+42.01' : '^', '+224.1' : '&', '+229.1' : '*' }
## pyroglu = (e, cyclocys = )d

def modseq_to_coded_seq( seq ):
    for mod_id in mod_regex_keys:
        seq = re.sub( mod_id, mod_regex_keys[mod_id], seq )
        
    # N/C mods
    if seq[0] == 'd' : seq = ')' + seq
    elif seq[0] == 'e' : seq = '(' + seq
    elif seq[0] == '[' : seq = nterm_keys[ seq[1:7] ] + seq[ seq.find(']')+1: ]
    else: seq = '-'+seq
    seq = seq+'_'    
    
    return seq


def char_mapper(seq, max_size=max_peptide_len+2):
    seq_by_int = [aa_to_int[seq[i]] for i in range(len(seq))]
    seq_by_int += [0]*(max_size - len(seq_by_int))
    return seq_by_int


def hi_db_to_tensors( hi_db ):
    seq_array = np.asarray( [ char_mapper( p ) for p in hi_db.CodedPeptideSeq ],
                            'int64', )
    hi_array = np.asarray( [ [ hi ] for hi in hi_db.HI ], 'float32', )
    sources = sorted( set( hi_db.Source ) )
    source_array = np.asarray( [ [ float( s == sx ) for sx in sources ] for s in hi_db.Source ],
                               'float32', )
    tensors = [ torch.Tensor( x ) for x in [ seq_array, hi_array, source_array] ]
    tensors[0] = tensors[0].to(torch.int64) # Need to ensure seq tensor are ints for embedding layer
    
    return TensorDataset( *tensors )
    
    

    

