import glob
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert( 0, os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', 'src' ) ) )

import cartographer_multistart


class CartographerMultistartTest( unittest.TestCase ):
    def setUp( self ):
        self.temp_dir = tempfile.mkdtemp( prefix='cartographer_multistart_' )

    def tearDown( self ):
        shutil.rmtree( self.temp_dir )

    def test_multistart_forwards_patience_and_keeps_best_checkpoint( self ):
        run_losses = [ 0.9, 0.4, 0.7 ]
        calls = []

        def fake_train( dataset_root,
                        output_file_name,
                        device='auto',
                        num_workers=0,
                        prtc_report=None,
                        patience=None,
                        model_file=None,
                        start_epoch=1,
                        n_epochs=None ):
            call_idx = len( calls )
            calls.append( { 'dataset_root' : dataset_root,
                            'output_file_name' : output_file_name,
                            'device' : device,
                            'num_workers' : num_workers,
                            'prtc_report' : prtc_report,
                            'patience' : patience,
                            'model_file' : model_file,
                            'start_epoch' : start_epoch,
                            'n_epochs' : n_epochs, } )

            with open( output_file_name, 'w' ) as f:
                f.write( 'run=' + str(call_idx + 1) )

            return float( run_losses[ call_idx ] )

        argv = [ 'cartographer_multistart.py',
                 '--dataset_root', '/tmp/prospect-ptms-ms2',
                 '--output_dir', self.temp_dir,
                 '--output_file', 'Cartographer_best.pt',
                 '--device', 'cpu',
                 '--num_workers', '0',
                 '--n_starts', '3',
                 '--patience', '7',
                 '--n_epochs', '12',
                 '--prtc_report', os.path.join( self.temp_dir, 'prtc.tsv' ) ]

        with mock.patch.object( cartographer_multistart,
                                'train_cartographer',
                                side_effect=fake_train ), \
             mock.patch.object( sys, 'argv', argv ):
            cartographer_multistart.main()

        self.assertEqual( len( calls ), 3 )
        self.assertTrue( all( c[ 'patience' ] == 7 for c in calls ) )
        self.assertTrue( all( c[ 'n_epochs' ] == 12 for c in calls ) )
        self.assertTrue( all( c[ 'start_epoch' ] == 1 for c in calls ) )
        self.assertTrue( all( c[ 'model_file' ] is None for c in calls ) )
        self.assertEqual( calls[0][ 'prtc_report' ],
                          os.path.join( self.temp_dir, 'prtc_run01.tsv' ) )
        self.assertEqual( calls[1][ 'prtc_report' ],
                          os.path.join( self.temp_dir, 'prtc_run02.tsv' ) )
        self.assertEqual( calls[2][ 'prtc_report' ],
                          os.path.join( self.temp_dir, 'prtc_run03.tsv' ) )

        final_model = os.path.join( self.temp_dir, 'Cartographer_best.pt' )
        self.assertTrue( os.path.isfile( final_model ) )
        with open( final_model, 'r' ) as f:
            self.assertEqual( f.read(), 'run=2' )

        self.assertEqual( glob.glob( os.path.join( self.temp_dir, '*_run*.pt' ) ), [] )


if __name__ == '__main__':
    unittest.main()
