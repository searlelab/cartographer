import os
import sys
import unittest

import torch


sys.path.insert( 0, os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', 'src' ) ) )

import cartographer_contrast_angle_report


class CartographerContrastAngleReportTest( unittest.TestCase ):
    def test_nce_conversion_helpers( self ):
        self.assertAlmostEqual( cartographer_contrast_angle_report.nce_to_aligned_normed( 30.0 ), 0.30, places=7 )
        self.assertAlmostEqual( cartographer_contrast_angle_report.aligned_normed_to_nce( 0.30 ), 30.0, places=7 )

    def test_classify_ptm_group_ignores_carbamidomethyl( self ):
        seq = '[]-PEPC[UNIMOD:4]TIDE-[]'
        self.assertEqual( cartographer_contrast_angle_report.classify_ptm_group( seq ), 'Unmodified' )

    def test_classify_ptm_group_returns_single_mod( self ):
        seq = '[]-PEPM[UNIMOD:35]TIDE-[]'
        self.assertEqual( cartographer_contrast_angle_report.classify_ptm_group( seq ), 'Oxidation' )

    def test_classify_ptm_group_flags_mixed( self ):
        seq = '[UNIMOD:1]-PEPM[UNIMOD:35]TIDE-[]'
        self.assertEqual( cartographer_contrast_angle_report.classify_ptm_group( seq ), 'mixed' )

    def test_spectral_contrast_angles_identity_and_orthogonal( self ):
        pred = torch.tensor( [ [ 1.0, 2.0, 3.0 ],
                               [ 1.0, 0.0, 0.0 ] ],
                             dtype=torch.float32 )
        true = torch.tensor( [ [ 1.0, 2.0, 3.0 ],
                               [ 0.0, 1.0, 0.0 ] ],
                             dtype=torch.float32 )

        out = cartographer_contrast_angle_report.spectral_contrast_angles( pred, true )
        self.assertGreater( float( out[0] ), 0.999 )
        self.assertAlmostEqual( float( out[1] ), 0.0, places=6 )

    def test_spectral_contrast_angles_respects_invalid_ion_mask( self ):
        pred = torch.tensor( [ [ 2.0, 4.0, 8.0 ] ], dtype=torch.float32 )
        true = torch.tensor( [ [ 2.0, 4.0, -1.0 ] ], dtype=torch.float32 )

        out = cartographer_contrast_angle_report.spectral_contrast_angles( pred, true )
        self.assertGreater( float( out[0] ), 0.999 )

    def test_should_replace_target_nce_match_prefers_closer_delta( self ):
        best = ( 0.12, 0.42, 0, 0, 0 )
        self.assertTrue( cartographer_contrast_angle_report.should_replace_target_nce_match( best,
                                                                                              0.33,
                                                                                              0.30 ) )
        self.assertFalse( cartographer_contrast_angle_report.should_replace_target_nce_match( best,
                                                                                               0.55,
                                                                                               0.30 ) )

    def test_should_replace_target_nce_match_tiebreak_prefers_lower_nce( self ):
        best = ( 0.10, 0.40, 0, 0, 0 )
        self.assertTrue( cartographer_contrast_angle_report.should_replace_target_nce_match( best,
                                                                                              0.20,
                                                                                              0.30 ) )
        self.assertFalse( cartographer_contrast_angle_report.should_replace_target_nce_match( best,
                                                                                               0.50,
                                                                                               0.30 ) )


if __name__ == '__main__':
    unittest.main()
