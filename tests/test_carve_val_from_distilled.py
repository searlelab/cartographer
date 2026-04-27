import json
import os
import shutil
import sys
import unittest
import uuid
from unittest import mock


sys.path.insert( 0, os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', 'src' ) ) )

import carve_val_from_distilled


class CarveValFromDistilledTest( unittest.TestCase ):
    def setUp( self ):
        self.temp_dir = os.path.join( os.getcwd(), 'tests_artifacts_carve_val_' + uuid.uuid4().hex )
        self.data_dir = os.path.join( self.temp_dir, 'data' )
        os.makedirs( self.data_dir, exist_ok=True )

        for idx in range( 3 ):
            open( os.path.join( self.data_dir, 'train-' + format( idx, '05d' ) + '.parquet' ), 'wb' ).close()
        for idx in range( 6 ):
            open( os.path.join( self.data_dir, 'test-' + format( idx, '05d' ) + '.parquet' ), 'wb' ).close()

        with open( os.path.join( self.temp_dir, 'distilled_dataset_metadata.json' ), 'w' ) as handle:
            json.dump( { 'train_shards_written' : 3, 'test_shards_written' : 6 }, handle )

    def tearDown( self ):
        if os.path.isdir( self.temp_dir ):
            try:
                shutil.rmtree( self.temp_dir )
            except PermissionError:
                pass

    def test_choose_and_rename_test_shards_produces_expected_val_targets( self ):
        test_files = carve_val_from_distilled.discover_split_files( self.temp_dir, 'test' )
        chosen = carve_val_from_distilled.choose_val_shards( test_files, 0.5, seed=7 )
        self.assertEqual( len( chosen ), 3 )

        with mock.patch.object( carve_val_from_distilled.os, 'replace', return_value=None ) as replace_mock:
            renamed = carve_val_from_distilled.rename_test_shards( self.temp_dir, chosen )

        self.assertEqual( len( renamed ), 3 )
        self.assertEqual( os.path.basename( renamed[0] ), 'val-00000.parquet' )
        self.assertEqual( os.path.basename( renamed[1] ), 'val-00001.parquet' )
        self.assertEqual( os.path.basename( renamed[2] ), 'val-00002.parquet' )
        self.assertEqual( replace_mock.call_count, 3 )


if __name__ == '__main__':
    unittest.main()
