import numpy as np

from sculptor_settings import max_peptide_len, charge_dist_len


# Sculptor keeps Cartographer-style single-character sequence encoding,
# extended with additional tokens for CCS-relevant PTMs.
residues = [ 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',

             'c', 'm', 'd', 'e', 's', 't', 'y', 'a', 'b', 'u',
             'n', 'o', 'p', 'q', 'r', 'x', 'z',
             'w', 'f', 'g', 'h', 'i', 'j', 'k',

             'l', 'v', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',

             '-', '^', '(', ')', '&', '*', '_', ]

if len( residues ) != len( set( residues ) ):
    raise ValueError( 'Duplicate residue token(s) in Sculptor vocabulary' )

n_states = len( residues ) + 1


aa_to_int = dict( zip( residues, range( 1, len(residues) + 1 ) ) )


nterm_unimod_map = {
    'UNIMOD:1'  : '^',   # Acetyl
    'UNIMOD:58' : '9',   # Propionyl
}


residue_unimod_map = {
    ('C', 'UNIMOD:4')    : 'c',   # Carbamidomethyl
    ('M', 'UNIMOD:35')   : 'm',   # Oxidation
    ('W', 'UNIMOD:35')   : 'w',   # Oxidation

    ('S', 'UNIMOD:21')   : 's',   # Phospho
    ('T', 'UNIMOD:21')   : 't',   # Phospho
    ('Y', 'UNIMOD:21')   : 'y',   # Phospho

    ('K', 'UNIMOD:121')  : 'u',   # GlyGly
    ('K', 'UNIMOD:1')    : 'a',   # Acetyl
    ('K', 'UNIMOD:747')  : 'l',   # Malonyl

    ('N', 'UNIMOD:7')    : 'f',   # Deamidation
    ('Q', 'UNIMOD:7')    : 'g',   # Deamidation
    ('R', 'UNIMOD:7')    : 'k',   # Deamidation

    ('K', 'UNIMOD:34')   : 'n',   # Methyl
    ('R', 'UNIMOD:34')   : 'q',   # Methyl
    ('K', 'UNIMOD:36')   : 'o',   # Dimethyl
    ('R', 'UNIMOD:36')   : 'r',   # Dimethyl
    ('K', 'UNIMOD:37')   : 'p',   # Trimethyl

    ('C', 'UNIMOD:312')  : 'v',   # Cysteinyl
    ('K', 'UNIMOD:1848') : '0',   # Glutarylation
    ('K', 'UNIMOD:64')   : 'b',   # Succinyl
    ('K', 'UNIMOD:1849') : '1',   # Hydroxyisobutyryl
    ('K', 'UNIMOD:1289') : '2',   # Butyryl
    ('K', 'UNIMOD:1363') : '3',   # Crotonyl
    ('K', 'UNIMOD:122')  : '4',   # Formyl
    ('Y', 'UNIMOD:354')  : '5',   # Nitro
    ('K', 'UNIMOD:3')    : '6',   # Biotin
    ('P', 'UNIMOD:408')  : '7',   # Glycosyl hydroxyproline

    ('N', 'UNIMOD:43')   : 'h',   # HexNAc
    ('S', 'UNIMOD:43')   : 'i',   # HexNAc
    ('T', 'UNIMOD:43')   : 'j',   # HexNAc

    ('K', 'UNIMOD:58')   : '8',   # Propionyl
}


def _increment_skip( skip_counts, key ):
    if skip_counts is not None:
        skip_counts[ key ] = skip_counts.get( key, 0 ) + 1


def unimod_to_codedseq( modified_sequence, max_len=max_peptide_len, skip_counts=None ):
    """Convert Cartographer-format UNIMOD sequence to coded token sequence.

    Expected input format:
      [UNIMOD:x]-PEPTM[UNIMOD:y]IDE-[]
    """
    parts = modified_sequence.split( '-', 2 )
    if len( parts ) != 3:
        _increment_skip( skip_counts, 'unexpected_format' )
        return None

    nterm_part, body, cterm_part = parts

    if nterm_part == '[]' or nterm_part == '':
        nterm_char = '-'
    else:
        nterm_tag = nterm_part.strip( '[]' )
        if nterm_tag in nterm_unimod_map:
            nterm_char = nterm_unimod_map[ nterm_tag ]
        else:
            _increment_skip( skip_counts, 'nterm:' + nterm_tag )
            return None

    if cterm_part != '[]' and cterm_part != '':
        cterm_tag = cterm_part.strip( '[]' )
        _increment_skip( skip_counts, 'cterm:' + cterm_tag )
        return None

    coded_body = ''
    i = 0
    while i < len( body ):
        aa = body[ i ]
        if not ( 'A' <= aa <= 'Z' ):
            _increment_skip( skip_counts, 'invalid_residue:' + aa )
            return None
        i += 1

        if i < len( body ) and body[ i ] == '[':
            end = body.find( ']', i )
            if end < 0:
                _increment_skip( skip_counts, 'unterminated_mod' )
                return None

            unimod_tag = body[ i + 1 : end ]
            i = end + 1

            if i < len( body ) and body[ i ] == '[':
                _increment_skip( skip_counts, 'stacked_mods' )
                return None

            key = ( aa, unimod_tag )
            if key in residue_unimod_map:
                coded_body += residue_unimod_map[ key ]
            else:
                _increment_skip( skip_counts, aa + '[' + unimod_tag + ']' )
                return None
        else:
            coded_body += aa

    if len( coded_body ) > max_len:
        _increment_skip( skip_counts, 'too_long' )
        return None

    return nterm_char + coded_body + '_'


def codedseq_to_array( seq, max_size=max_peptide_len + 2 ):
    seq_by_int = [ aa_to_int[ seq[i] ] for i in range( len(seq) ) ]
    seq_by_int += [ 0 ] * ( max_size - len(seq_by_int) )
    return np.asarray( seq_by_int, 'int64' )


def return_charge_onehot( precursor_charge, n_charges=charge_dist_len ):
    return [ float( precursor_charge == z ) for z in range( 1, n_charges + 1 ) ]
