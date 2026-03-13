import os
import sys
import unittest
from unittest import mock


sys.path.insert( 0, os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', 'src' ) ) )

import ptm_loss_report


class PtmLossReportSplitTest( unittest.TestCase ):
    def test_discover_fit_and_test_files_includes_val( self ):
        split_files = {
            'train' : [ 'train-001.parquet', 'train-002.parquet' ],
            'val' : [ 'val-001.parquet' ],
            'test' : [ 'test-001.parquet' ],
        }

        def fake_discover( _dataset_root, split ):
            return list( split_files.get( split, [] ) )

        with mock.patch.object( ptm_loss_report,
                                'discover_split_files',
                                side_effect=fake_discover ):
            train_files, val_files, fit_files, test_files = ptm_loss_report.discover_fit_and_test_files( '/tmp/dataset' )

        self.assertEqual( train_files, split_files[ 'train' ] )
        self.assertEqual( val_files, split_files[ 'val' ] )
        self.assertEqual( fit_files, split_files[ 'train' ] + split_files[ 'val' ] )
        self.assertEqual( test_files, split_files[ 'test' ] )

    def test_discover_fit_and_test_files_without_val( self ):
        split_files = {
            'train' : [ 'train-001.parquet' ],
            'test' : [ 'test-001.parquet', 'test-002.parquet' ],
        }

        def fake_discover( _dataset_root, split ):
            return list( split_files.get( split, [] ) )

        with mock.patch.object( ptm_loss_report,
                                'discover_split_files',
                                side_effect=fake_discover ):
            train_files, val_files, fit_files, test_files = ptm_loss_report.discover_fit_and_test_files( '/tmp/dataset' )

        self.assertEqual( train_files, split_files[ 'train' ] )
        self.assertEqual( val_files, [] )
        self.assertEqual( fit_files, split_files[ 'train' ] )
        self.assertEqual( test_files, split_files[ 'test' ] )


if __name__ == '__main__':
    unittest.main()
