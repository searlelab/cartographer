import os
import sys
import unittest
from unittest import mock

import torch


sys.path.insert( 0, os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', 'src' ) ) )

import cartographer_trainer
import electrician_trainer
import sculptor_trainer


class DummyModel( torch.nn.Module ):
    def __init__( self ):
        super().__init__()
        self.weight = torch.nn.Parameter( torch.tensor( 0.0 ) )

    def forward( self, *args, **kwargs ):
        return self.weight


class DummyOptimizer( object ):
    def __init__( self, params, lr ):
        self.params = list( params )
        self.lr = float( lr )

    def step( self ):
        return None

    def zero_grad( self ):
        return None


class FakeDataset( object ):
    def __init__( self, files, **kwargs ):
        self.files = list( files )
        self.kwargs = dict( kwargs )


class TrainerValInclusionTest( unittest.TestCase ):
    def test_cartographer_uses_train_plus_val_for_fit( self ):
        split_files = {
            'train' : [ 'train-000.parquet' ],
            'val' : [ 'val-000.parquet' ],
            'test' : [ 'test-000.parquet' ],
        }

        def fake_discover( _dataset_root, split ):
            return split_files.get( split, [] )

        def fake_train_model( _model, datasets, *args, **kwargs ):
            self.assertEqual( datasets[ 'train' ].files,
                              [ 'train-000.parquet', 'val-000.parquet' ] )
            self.assertEqual( datasets[ 'test' ].files, [ 'test-000.parquet' ] )
            return 0.123

        with mock.patch.object( cartographer_trainer,
                                'discover_split_files',
                                side_effect=fake_discover ), \
             mock.patch.object( cartographer_trainer,
                                'ProspectMS2Dataset',
                                side_effect=lambda files, shuffle_files=False: FakeDataset( files, shuffle_files=shuffle_files ) ), \
             mock.patch.object( cartographer_trainer,
                                'initialize_cartographer_model',
                                return_value=DummyModel() ), \
             mock.patch.object( cartographer_trainer,
                                'Spectrum_masked_negLogit',
                                return_value=object() ), \
             mock.patch.object( cartographer_trainer,
                                'train_model',
                                side_effect=fake_train_model ), \
             mock.patch.dict( cartographer_trainer.training_parameters,
                              {
                                  'optimizer' : DummyOptimizer,
                                  'learning_rate' : 0.001,
                                  'n_epochs' : 1,
                                  'initial_batch_size' : 2,
                                  'max_batch_size' : 2,
                                  'epochs_to_2x_batch' : 1,
                              },
                              clear=True ):
            loss = cartographer_trainer.train_cartographer( '/tmp/dataset',
                                                            '/tmp/out.pt',
                                                            device='cpu',
                                                            num_workers=0,
                                                            n_epochs=1 )

        self.assertAlmostEqual( loss, 0.123, places=6 )

    def test_electrician_uses_train_plus_val_for_fit( self ):
        split_files = {
            'train' : [ 'train-000.parquet' ],
            'val' : [ 'val-000.parquet' ],
            'test' : [ 'test-000.parquet' ],
        }

        def fake_discover( _dataset_root, split ):
            return split_files.get( split, [] )

        def fake_train_model( _model, datasets, *args, **kwargs ):
            self.assertEqual( datasets[ 'train' ].files,
                              [ 'train-000.parquet', 'val-000.parquet' ] )
            self.assertEqual( datasets[ 'test' ].files, [ 'test-000.parquet' ] )
            return 0.456

        with mock.patch.object( electrician_trainer,
                                'discover_split_files',
                                side_effect=fake_discover ), \
             mock.patch.object( electrician_trainer,
                                'ProspectChargeDataset',
                                side_effect=lambda files, shuffle_files=False: FakeDataset( files, shuffle_files=shuffle_files ) ), \
             mock.patch.object( electrician_trainer,
                                'initialize_electrician_model',
                                return_value=DummyModel() ), \
             mock.patch.object( electrician_trainer,
                                'ChargeDistribution_CrossEntropy',
                                return_value=object() ), \
             mock.patch.object( electrician_trainer,
                                'train_model',
                                side_effect=fake_train_model ), \
             mock.patch.dict( electrician_trainer.training_parameters,
                              {
                                  'optimizer' : DummyOptimizer,
                                  'learning_rate' : 0.001,
                                  'n_epochs' : 1,
                                  'initial_batch_size' : 2,
                                  'max_batch_size' : 2,
                                  'epochs_to_2x_batch' : 1,
                              },
                              clear=True ):
            loss = electrician_trainer.train_electrician( '/tmp/dataset',
                                                          '/tmp/out.pt',
                                                          device='cpu',
                                                          num_workers=0,
                                                          n_epochs=1 )

        self.assertAlmostEqual( loss, 0.456, places=6 )

    def test_sculptor_uses_train_plus_val_for_fit( self ):
        split_files = {
            'train' : [ 'train-000.parquet' ],
            'val' : [ 'val-000.parquet' ],
            'test' : [ 'test-000.parquet' ],
        }

        def fake_discover( _dataset_root, split ):
            return split_files.get( split, [] )

        def fake_train_model( _model, datasets, *args, **kwargs ):
            self.assertEqual( datasets[ 'train' ].files,
                              [ 'train-000.parquet', 'val-000.parquet' ] )
            self.assertEqual( datasets[ 'test' ].files, [ 'test-000.parquet' ] )
            return 0.789

        with mock.patch.object( sculptor_trainer,
                                'load_metadata',
                                return_value=( { 'train_ccs_mean' : 100.0, 'train_ccs_std' : 10.0 },
                                               '/tmp/meta.json' ) ), \
             mock.patch.object( sculptor_trainer,
                                'discover_split_files',
                                side_effect=fake_discover ), \
             mock.patch.object( sculptor_trainer,
                                'SculptorCCSDataset',
                                side_effect=lambda files, ccs_mean, ccs_std, shuffle_files=False: FakeDataset( files,
                                                                                                                 ccs_mean=ccs_mean,
                                                                                                                 ccs_std=ccs_std,
                                                                                                                 shuffle_files=shuffle_files ) ), \
             mock.patch.object( sculptor_trainer,
                                'initialize_sculptor_model',
                                return_value=DummyModel() ), \
             mock.patch.object( sculptor_trainer,
                                'CCS_HuberLoss',
                                return_value=object() ), \
             mock.patch.object( sculptor_trainer,
                                'evaluate_metrics',
                                return_value={ 'mae' : 1.0, 'rmse' : 2.0, 'n_test' : 3 } ), \
             mock.patch.object( sculptor_trainer,
                                'train_model',
                                side_effect=fake_train_model ), \
             mock.patch.dict( sculptor_trainer.training_parameters,
                              {
                                  'optimizer' : DummyOptimizer,
                                  'learning_rate' : 0.001,
                                  'n_epochs' : 1,
                                  'initial_batch_size' : 2,
                                  'max_batch_size' : 2,
                                  'epochs_to_2x_batch' : 1,
                              },
                              clear=True ):
            metrics = sculptor_trainer.train_sculptor( '/tmp/dataset',
                                                       '/tmp/out.pt',
                                                       device='cpu',
                                                       num_workers=0,
                                                       n_epochs=1 )

        self.assertAlmostEqual( metrics[ 'best_test_loss' ], 0.789, places=6 )
        self.assertEqual( metrics[ 'n_test' ], 3 )


if __name__ == '__main__':
    unittest.main()
