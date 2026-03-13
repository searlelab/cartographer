import numpy as np

from chronologer_unimod_settings import chronologer_max_peptide_len, chronologer_min_peptide_len


# Canonical Chronologer state space:
# 20 AA + 17 modified residues + 7 terminal states + 10 reserved custom states.
residues = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    'c', 'm', 'd', 'e', 's', 't', 'y', 'a', 'b', 'u',
    'n', 'o', 'p', 'q', 'r', 'x', 'z',
    '-', '^', '(', ')', '&', '*', '_',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
]

n_states = len(residues) + 1
aa_to_int = dict(zip(residues, range(1, len(residues) + 1)))


nterm_unimod_map = {
    'UNIMOD:1': '^',    # Acetyl
    'UNIMOD:737': '*',  # TMT6plex
    'UNIMOD:739': '&',  # TMT0
    'UNIMOD:28': '(',   # pyro-Glu (Q)
    'UNIMOD:27': ')',   # pyro-Glu (E)
}


# Canonical residue mappings.
residue_unimod_map = {
    ('C', 'UNIMOD:4'): 'c',
    ('M', 'UNIMOD:35'): 'm',
    ('S', 'UNIMOD:21'): 's',
    ('T', 'UNIMOD:21'): 't',
    ('Y', 'UNIMOD:21'): 'y',
    ('K', 'UNIMOD:1'): 'a',
    ('K', 'UNIMOD:64'): 'b',
    ('K', 'UNIMOD:121'): 'u',
    ('K', 'UNIMOD:34'): 'n',
    ('K', 'UNIMOD:36'): 'o',
    ('K', 'UNIMOD:37'): 'p',
    ('R', 'UNIMOD:34'): 'q',
    ('R', 'UNIMOD:36'): 'r',
    ('K', 'UNIMOD:739'): 'z',
    ('K', 'UNIMOD:737'): 'x',
}


# iRT additions mapped to reserved canonical custom slots.
custom_residue_unimod_map = {
    ('W', 'UNIMOD:35'): '0',  # Oxidation (W)
    ('N', 'UNIMOD:7'): '1',   # Deamidation (N)
    ('Q', 'UNIMOD:7'): '2',   # Deamidation (Q)
    ('R', 'UNIMOD:7'): '3',   # Deamidation (R)
    ('N', 'UNIMOD:43'): '4',  # HexNAc (N)
    ('S', 'UNIMOD:43'): '5',  # HexNAc (S)
    ('T', 'UNIMOD:43'): '6',  # HexNAc (T)
}


def _inc(skip_counts, key):
    if skip_counts is None:
        return
    skip_counts[key] = skip_counts.get(key, 0) + 1


def _extract_single_tag(part):
    if part in ['', '[]']:
        return None
    if not (part.startswith('[') and part.endswith(']')):
        return None
    inner = part[1:-1]
    if '][' in inner:
        return 'STACKED'
    return inner


def _normalize_source_sequence(modified_sequence):
    if isinstance(modified_sequence, str):
        return modified_sequence.strip()
    return None


def unimod_to_chronologer_codedseq(modified_sequence,
                                   max_len=chronologer_max_peptide_len,
                                   skip_counts=None):
    seq_text = _normalize_source_sequence(modified_sequence)
    if seq_text is None or len(seq_text) == 0:
        _inc(skip_counts, 'invalid_sequence')
        return None

    parts = seq_text.split('-', 2)
    if len(parts) != 3:
        _inc(skip_counts, 'unexpected_format')
        return None

    nterm_part, body, cterm_part = parts

    nterm_tag = _extract_single_tag(nterm_part)
    if nterm_tag == 'STACKED':
        _inc(skip_counts, 'stacked_nterm')
        return None
    if nterm_tag is None:
        nterm_char = '-'
    else:
        nterm_char = nterm_unimod_map.get(nterm_tag)
        if nterm_char is None:
            _inc(skip_counts, 'unknown_nterm:' + nterm_tag)
            return None

    cterm_tag = _extract_single_tag(cterm_part)
    if cterm_tag == 'STACKED':
        _inc(skip_counts, 'stacked_cterm')
        return None
    if cterm_tag is None:
        cterm_char = '_'
    else:
        # v1 currently supports only unmodified C-termini.
        _inc(skip_counts, 'unknown_cterm:' + cterm_tag)
        return None

    coded_body = ''
    i = 0
    while i < len(body):
        aa = body[i]
        if aa == '[':
            _inc(skip_counts, 'unexpected_mod_without_residue')
            return None
        i += 1

        tags = []
        while i < len(body) and body[i] == '[':
            end = body.find(']', i)
            if end < 0:
                _inc(skip_counts, 'unclosed_bracket')
                return None
            tags.append(body[i + 1:end])
            i = end + 1

        if len(tags) > 1:
            _inc(skip_counts, 'stacked_mods')
            return None

        if len(tags) == 0:
            if aa not in aa_to_int:
                _inc(skip_counts, 'unknown_residue:' + aa)
                return None
            coded_body += aa
            continue

        tag = tags[0]
        if len(coded_body) == 0 and nterm_char == '-':
            # Support pyro-Glu encoded on the first body residue.
            if aa == 'Q' and tag == 'UNIMOD:28':
                nterm_char = '('
                coded_body += aa
                continue
            if aa == 'E' and tag == 'UNIMOD:27':
                nterm_char = ')'
                coded_body += aa
                continue

        key = (aa, tag)
        if key in residue_unimod_map:
            coded_body += residue_unimod_map[key]
            continue
        if key in custom_residue_unimod_map:
            coded_body += custom_residue_unimod_map[key]
            continue

        _inc(skip_counts, 'unknown_residue_mod:' + aa + '[' + tag + ']')
        return None

    body_len = len(coded_body)
    if body_len < chronologer_min_peptide_len:
        _inc(skip_counts, 'too_short')
        return None
    if body_len > max_len:
        _inc(skip_counts, 'too_long')
        return None

    return nterm_char + coded_body + cterm_char


def codedseq_to_array(seq, max_size=chronologer_max_peptide_len + 2):
    seq_by_int = [aa_to_int[s] for s in seq]
    if len(seq_by_int) > max_size:
        raise ValueError('Coded sequence exceeds max_size')
    seq_by_int += [0] * (max_size - len(seq_by_int))
    return np.asarray(seq_by_int, dtype='int64')


def tokenizer_metadata():
    return {
        'n_states': int(n_states),
        'residues': list(residues),
        'nterm_unimod_map': dict(sorted(nterm_unimod_map.items())),
        'residue_unimod_map': {
            aa + ':' + tag: code
            for (aa, tag), code in sorted(residue_unimod_map.items())
        },
        'custom_residue_unimod_map': {
            aa + ':' + tag: code
            for (aa, tag), code in sorted(custom_residue_unimod_map.items())
        },
    }

