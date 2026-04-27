import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import torch


sys.path.insert( 0, os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', 'src' ) ) )

import scout_trainer
import training_loop
from electrician_settings import charge_dist_len
from loss_functions import ChargeDistribution_CrossEntropy
from scout_loss import ScoutMultiTaskLoss
from scout_loader import SCOUT_CHARGE_DIST_OFFSET, SCOUT_TARGET_LEN
from scout_settings import ms2_vector_len


class _SimpleTrainingLoss( torch.nn.Module ):
    def forward( self, pred, true, weight ):
        return torch.mean( ( pred - true ) ** 2 * weight )


class _SimpleEvalModel( torch.nn.Module ):
    def __init__( self ):
        super().__init__()
        self.anchor = torch.nn.Parameter( torch.tensor( 0.0 ) )

    def forward( self, seq, charge, nce ):
        batch_size = int( seq.shape[0] )
        ms2 = torch.ones( ( batch_size, ms2_vector_len ), device=seq.device )
        irt = torch.zeros( ( batch_size, 1 ), device=seq.device )
        ccs = torch.zeros( ( batch_size, 1 ), device=seq.device )
        charge_dist = torch.full( ( batch_size, charge_dist_len ), 1.0 / charge_dist_len, device=seq.device )
        return { 'ms2' : ms2, 'irt' : irt, 'ccs' : ccs, 'charge_dist' : charge_dist }


class _FakeLoader( object ):
    def __init__( self, batch_size ):
        self.batch_size = int( batch_size )


class TrainingLoopSkipPhaseTest( unittest.TestCase ):
    def test_skip_batch_phases_bypasses_callback_owned_val_loader( self ):
        model = torch.nn.Linear( 1, 1, bias=False )
        optimizer = torch.optim.SGD( model.parameters(), lr=0.1 )
        loss_fx = _SimpleTrainingLoss()

        train_dataset = object()
        val_dataset = object()
        callback_calls = []

        train_batch = (
            torch.tensor( [ [ 1.0 ], [ 2.0 ] ] ),
            torch.tensor( [ [ 1.0 ], [ 2.0 ] ] ),
            torch.ones( 2, 1 ),
        )

        def fake_dataloader( dataset, batch_size, shuffle=False, num_workers=0 ):
            if dataset is train_dataset:
                return [ train_batch ]
            if dataset is val_dataset:
                raise AssertionError( 'val DataLoader should not be built when phase is callback-owned' )
            raise AssertionError( 'Unexpected dataset' )

        def fake_checkpoint_metric( **kwargs ):
            callback_calls.append( kwargs )
            return 0.25, 'score'

        with mock.patch.object( training_loop, 'DataLoader', side_effect=fake_dataloader ), \
             mock.patch.object( torch, 'save', return_value=None ):
            best_loss = training_loop.train_model( model,
                                                   { 'train' : train_dataset, 'val' : val_dataset },
                                                   initial_batch_size=8,
                                                   max_batch_size=8,
                                                   epochs_to_double_batch=1,
                                                   loss_fx=loss_fx,
                                                   optimizer=optimizer,
                                                   num_epochs=1,
                                                   train_device='cpu',
                                                   other_device='cpu',
                                                   file_name='dummy.pt',
                                                   checkpoint_metric_callback=fake_checkpoint_metric,
                                                   checkpoint_phase='val',
                                                   skip_batch_phases={ 'val' },
                                                   report_epoch_loss=False )

        self.assertEqual( len( callback_calls ), 1 )
        self.assertEqual( callback_calls[0][ 'phase' ], 'val' )
        self.assertAlmostEqual( best_loss, 0.25, places=6 )


class ScoutEvaluationHelpersTest( unittest.TestCase ):
    def test_eval_backoff_halves_batch_size_after_cuda_oom( self ):
        metrics = { 'test_ms2_cosine' : 0.8,
                    'test_ms2_count' : 5,
                    'test_irt_mae' : 0.2,
                    'test_irt_rmse' : 0.3,
                    'test_irt_count' : 5,
                    'test_ccs_mae' : 1.1,
                    'test_ccs_rmse' : 1.3,
                    'test_ccs_count' : 5,
                    'test_charge_ce' : 0.9,
                    'test_charge_count' : 5, }
        batch_sizes_seen = []

        def fake_loader( dataset, batch_size, shuffle=False, num_workers=0 ):
            return _FakeLoader( batch_size )

        def fake_eval_loader( model,
                              loader,
                              scalar_stats,
                              device='cpu',
                              label=None,
                              expected_rows=None,
                              progress_tick_rows=0,
                              effective_batch_size=None,
                              start_time=None ):
            batch_sizes_seen.append( int( effective_batch_size ) )
            if int( effective_batch_size ) == 8:
                raise RuntimeError( 'CUDA out of memory while evaluating Scout' )
            return dict( metrics )

        with mock.patch.object( scout_trainer, 'DataLoader', side_effect=fake_loader ), \
             mock.patch.object( scout_trainer, '_evaluate_scout_loader', side_effect=fake_eval_loader ), \
             mock.patch.object( torch.cuda, 'empty_cache', return_value=None ) as empty_cache_mock:
            result_metrics, effective_batch_size = scout_trainer.evaluate_scout_dataset( object(),
                                                                                         object(),
                                                                                         { 'irt_mean' : 0.0,
                                                                                           'irt_std' : 1.0,
                                                                                           'ccs_mean' : 0.0,
                                                                                           'ccs_std' : 1.0, },
                                                                                         batch_size=8,
                                                                                         num_workers=0,
                                                                                         device='cuda',
                                                                                         label='Val checkpoint',
                                                                                         return_effective_batch_size=True )

        self.assertEqual( batch_sizes_seen, [ 8, 4 ] )
        self.assertEqual( effective_batch_size, 4 )
        self.assertEqual( result_metrics[ 'effective_eval_batch_size' ], 4 )
        self.assertEqual( result_metrics[ 'test_ms2_count' ], 5 )
        empty_cache_mock.assert_called_once()

    def test_eval_progress_reporting_prints_start_progress_and_complete( self ):
        model = _SimpleEvalModel()
        scalar_stats = { 'irt_mean' : 0.0,
                         'irt_std' : 1.0,
                         'ccs_mean' : 0.0,
                         'ccs_std' : 1.0, }

        def make_batch():
            seq = torch.zeros( ( 2, 5 ), dtype=torch.int64 )
            charge = torch.zeros( ( 2, 6 ), dtype=torch.float32 )
            nce = torch.zeros( ( 2, 1 ), dtype=torch.float32 )
            target = torch.zeros( ( 2, SCOUT_TARGET_LEN ), dtype=torch.float32 )
            target[ :, :ms2_vector_len ] = 1.0
            target[ :, SCOUT_CHARGE_DIST_OFFSET : SCOUT_CHARGE_DIST_OFFSET + charge_dist_len ] = 1.0 / charge_dist_len
            mask = torch.ones( ( 2, 4 ), dtype=torch.float32 )
            return seq, charge, nce, target, mask

        loader = [ make_batch(), make_batch() ]

        output = io.StringIO()
        with contextlib.redirect_stdout( output ):
            metrics = scout_trainer._evaluate_scout_loader( model,
                                                            loader,
                                                            scalar_stats,
                                                            device='cpu',
                                                            label='Val checkpoint',
                                                            expected_rows=4,
                                                            progress_tick_rows=2,
                                                            effective_batch_size=2,
                                                            start_time=0.0 )

        text = output.getvalue()
        self.assertIn( 'Val checkpoint start:', text )
        self.assertIn( 'Val checkpoint progress:', text )
        self.assertIn( 'Val checkpoint complete:', text )
        self.assertEqual( metrics[ 'test_ms2_count' ], 4 )
        self.assertEqual( metrics[ 'test_charge_count' ], 4 )

    def test_load_cached_dataset_stats_uses_matching_split_signatures( self ):
        cache_dir = os.path.join( os.getcwd(), 'tests_artifacts_scout_cache_' + uuid.uuid4().hex )
        os.makedirs( cache_dir, exist_ok=True )
        try:
            payload = {
                'split_files' : {
                    'train' : [ 'train-000.parquet' ],
                    'val' : [ 'val-000.parquet' ],
                    'test' : [ 'test-000.parquet' ],
                },
                'coverage' : {
                    'train' : { 'rows_total' : 10, 'rows_tokenized' : 10, 'ms2_rows' : 10, 'irt_rows' : 9, 'ccs_rows' : 8, 'charge_dist_rows' : 10, 'skip_counts' : {} },
                    'val' : { 'rows_total' : 4, 'rows_tokenized' : 4, 'ms2_rows' : 4, 'irt_rows' : 4, 'ccs_rows' : 3, 'charge_dist_rows' : 4, 'skip_counts' : {} },
                    'test' : { 'rows_total' : 5, 'rows_tokenized' : 5, 'ms2_rows' : 5, 'irt_rows' : 5, 'ccs_rows' : 5, 'charge_dist_rows' : 5, 'skip_counts' : {} },
                },
                'scalar_stats' : {
                    'irt_mean' : 11.5,
                    'irt_std' : 5.6,
                    'ccs_mean' : 472.6,
                    'ccs_std' : 113.8,
                },
            }
            with open( os.path.join( cache_dir, scout_trainer.DATASET_STATS_CACHE_NAME ), 'w' ) as handle:
                json.dump( payload, handle )

            cached = scout_trainer._load_cached_dataset_stats( cache_dir,
                                                               [ os.path.join( cache_dir, 'data', 'train-000.parquet' ) ],
                                                               [ os.path.join( cache_dir, 'data', 'val-000.parquet' ) ],
                                                               [ os.path.join( cache_dir, 'data', 'test-000.parquet' ) ] )
            self.assertIsNotNone( cached )
            self.assertEqual( cached[ 'train_scan' ][ 'rows_tokenized' ], 10 )
            self.assertAlmostEqual( cached[ 'scalar_stats' ][ 'irt_mean' ], 11.5, places=6 )
        finally:
            if os.path.isdir( cache_dir ):
                try:
                    shutil.rmtree( cache_dir )
                except PermissionError:
                    pass

    def test_charge_loss_matches_electrician_cross_entropy_for_active_rows( self ):
        pred = { 'ms2' : torch.zeros( ( 2, ms2_vector_len ), dtype=torch.float32 ),
                 'irt' : torch.zeros( ( 2, 1 ), dtype=torch.float32 ),
                 'ccs' : torch.zeros( ( 2, 1 ), dtype=torch.float32 ),
                 'charge_dist' : torch.tensor( [ [ 0.70, 0.20, 0.05, 0.03, 0.01, 0.01 ],
                                                 [ 0.10, 0.10, 0.10, 0.10, 0.10, 0.50 ] ], dtype=torch.float32 ), }
        target = torch.zeros( ( 2, SCOUT_TARGET_LEN ), dtype=torch.float32 )
        target[ 0, SCOUT_CHARGE_DIST_OFFSET : SCOUT_CHARGE_DIST_OFFSET + charge_dist_len ] = torch.tensor( [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] )
        target[ 1, SCOUT_CHARGE_DIST_OFFSET : SCOUT_CHARGE_DIST_OFFSET + charge_dist_len ] = torch.tensor( [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] )
        mask = torch.zeros( ( 2, 4 ), dtype=torch.float32 )
        mask[:, 3] = 1.0

        loss_fx = ScoutMultiTaskLoss()
        scout_loss = loss_fx( pred, target, mask )
        expected_loss = ChargeDistribution_CrossEntropy()( pred[ 'charge_dist' ],
                                                           target[ :, SCOUT_CHARGE_DIST_OFFSET : SCOUT_CHARGE_DIST_OFFSET + charge_dist_len ],
                                                           torch.ones( 2, 1, dtype=torch.float32 ) )
        self.assertAlmostEqual( float( scout_loss.item() ), float( expected_loss.item() ), places=6 )

    def test_checkpoint_score_uses_weighted_rms_with_charge_ce_baseline( self ):
        metrics = { 'test_ms2_cosine' : 0.90,
                    'test_irt_mae' : 0.50,
                    'test_ccs_mae' : 5.0,
                    'test_charge_ce' : scout_trainer.CHARGE_CE_BASELINE, }
        score = scout_trainer._compute_balanced_checkpoint_score( metrics )
        expected = ( ( 2.0 / 5.0 ) * ( 1.0 ** 2 ) +
                     ( 1.0 / 5.0 ) * ( 0.5 ** 2 ) +
                     ( 1.0 / 5.0 ) * ( 0.5 ** 2 ) +
                     ( 1.0 / 5.0 ) * ( 1.0 ** 2 ) ) ** 0.5
        self.assertAlmostEqual( score, expected, places=6 )


class ScoutTrainerSmokeTest( unittest.TestCase ):
    def setUp( self ):
        self.temp_dir = os.path.join( os.getcwd(), 'tests_artifacts_scout_smoke_' + uuid.uuid4().hex )
        self.data_dir = os.path.join( self.temp_dir, 'data' )
        os.makedirs( self.data_dir, exist_ok=True )
        self._write_split( 'train-00000-of-00001.parquet',
                           [ self._row( 10.0, 110.0 ), self._row( 11.0, 111.0 ), self._row( 12.0, 112.0 ) ] )
        self._write_split( 'val-00000-of-00001.parquet',
                           [ self._row( 13.0, 113.0 ), self._row( 14.0, 114.0 ) ] )
        self._write_split( 'test-00000-of-00001.parquet',
                           [ self._row( 15.0, 115.0 ), self._row( 16.0, 116.0 ) ] )

    def tearDown( self ):
        if os.path.isdir( self.temp_dir ):
            try:
                shutil.rmtree( self.temp_dir )
            except PermissionError:
                pass

    def _row( self, irt, ccs ):
        return {
            'modified_sequence' : '[]-PEPTIDE-[]',
            'precursor_charge_onehot' : [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ],
            'charge_state_dist' : [ 0.05, 0.85, 0.10, 0.0, 0.0, 0.0 ],
            'collision_energy_aligned_normed' : 0.33,
            'indexed_retention_time' : float( irt ),
            'ccs' : float( ccs ),
            'intensities_raw' : [ 0.5 ] * ms2_vector_len,
        }

    def _write_split( self, filename, rows ):
        table = pa.table( { key : [ row[ key ] for row in rows ] for key in rows[0].keys() } )
        pq.write_table( table, os.path.join( self.data_dir, filename ) )

    def test_train_scout_one_epoch_reports_progress_and_writes_outputs( self ):
        output_file = os.path.join( self.temp_dir, 'Scout_smoke.pt' )
        output = io.StringIO()

        with contextlib.redirect_stdout( output ):
            result = scout_trainer.train_scout( self.temp_dir,
                                                output_file,
                                                device='cpu',
                                                num_workers=0,
                                                n_epochs=1,
                                                eval_batch_size=4 )

        text = output.getvalue()
        self.assertIn( 'Val checkpoint sample metrics:', text )
        self.assertIn( 'Final test start:', text )
        self.assertIn( 'Final test complete:', text )
        self.assertTrue( os.path.isfile( output_file ) )
        self.assertTrue( os.path.isfile( result[ 'metadata_path' ] ) )
        self.assertIn( 'metrics', result )
        self.assertIn( 'test_ms2_cosine', result[ 'metrics' ] )
        self.assertIn( 'test_charge_ce', result[ 'metrics' ] )


if __name__ == '__main__':
    unittest.main()
