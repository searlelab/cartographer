import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from chronologer_unimod_tokenizer import n_states, residues, unimod_to_chronologer_codedseq


class ChronologerUnimodTokenizerTest(unittest.TestCase):
    def test_canonical_state_space_size(self):
        self.assertEqual(len(residues), 54)
        self.assertEqual(n_states, 55)

    def test_supported_ptm_scope(self):
        cases = [
            ('[]-PEPTM[UNIMOD:35]IDE-[]', 'm'),   # Oxidation (M)
            ('[]-PEPTW[UNIMOD:35]IDE-[]', '0'),   # Oxidation (W)
            ('[]-PEPTC[UNIMOD:4]IDE-[]', 'c'),    # Carbamidomethyl
            ('[]-PEPTS[UNIMOD:21]IDE-[]', 's'),   # Phospho S
            ('[]-PEPTT[UNIMOD:21]IDE-[]', 't'),   # Phospho T
            ('[]-PEPTY[UNIMOD:21]IDE-[]', 'y'),   # Phospho Y
            ('[]-PEPTK[UNIMOD:121]IDE-[]', 'u'),  # GlyGly
            ('[]-PEPTK[UNIMOD:34]IDE-[]', 'n'),   # Methyl K
            ('[]-PEPTR[UNIMOD:34]IDE-[]', 'q'),   # Methyl R
            ('[]-PEPTN[UNIMOD:7]IDE-[]', '1'),    # Deamid N
            ('[]-PEPTQ[UNIMOD:7]IDE-[]', '2'),    # Deamid Q
            ('[]-PEPTR[UNIMOD:7]IDE-[]', '3'),    # Deamid R
            ('[]-PEPTN[UNIMOD:43]IDE-[]', '4'),   # HexNAc N
            ('[]-PEPTS[UNIMOD:43]IDE-[]', '5'),   # HexNAc S
            ('[]-PEPTT[UNIMOD:43]IDE-[]', '6'),   # HexNAc T
            ('[]-PEPTK[UNIMOD:737]IDE-[]', 'x'),  # TMT6 K
            ('[]-PEPTK[UNIMOD:739]IDE-[]', 'z'),  # TMT0 K
            ('[]-PEPTK[UNIMOD:1]IDE-[]', 'a'),    # Acetyl K
        ]

        for mod_seq, expected_token in cases:
            coded = unimod_to_chronologer_codedseq(mod_seq)
            self.assertIsNotNone(coded, msg=mod_seq)
            self.assertIn(expected_token, coded, msg=mod_seq)

    def test_nterm_ptm_scope(self):
        self.assertTrue(unimod_to_chronologer_codedseq('[UNIMOD:1]-PEPTIDE-[]').startswith('^'))
        self.assertTrue(unimod_to_chronologer_codedseq('[UNIMOD:737]-PEPTIDE-[]').startswith('*'))
        self.assertTrue(unimod_to_chronologer_codedseq('[UNIMOD:739]-PEPTIDE-[]').startswith('&'))
        self.assertTrue(unimod_to_chronologer_codedseq('[]-Q[UNIMOD:28]PEPTIDE-[]').startswith('('))
        self.assertTrue(unimod_to_chronologer_codedseq('[]-E[UNIMOD:27]PEPTIDE-[]').startswith(')'))

    def test_stacked_mods_are_skipped(self):
        skip_counts = {}
        coded = unimod_to_chronologer_codedseq(
            '[]-PEPTIDEK[UNIMOD:737][UNIMOD:1]-[]',
            skip_counts=skip_counts,
        )
        self.assertIsNone(coded)
        self.assertEqual(skip_counts.get('stacked_mods', 0), 1)


if __name__ == '__main__':
    unittest.main()

