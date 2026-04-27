import glob
import math
import os
import random

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from electrician_settings import charge_dist_len
from scout_settings import max_peptide_len, ms2_vector_len
from tensorize import codedseq_to_array, unimod_to_codedseq


SCOUT_IRT_OFFSET = ms2_vector_len
SCOUT_CCS_OFFSET = SCOUT_IRT_OFFSET + 1
SCOUT_CHARGE_DIST_OFFSET = SCOUT_CCS_OFFSET + 1
SCOUT_TARGET_LEN = SCOUT_CHARGE_DIST_OFFSET + charge_dist_len
SCOUT_MASK_LEN = 4


def discover_split_files( dataset_root, prefix ):
    pattern = os.path.join( dataset_root, 'data', prefix + '-*.parquet' )
    return sorted( glob.glob( pattern ) )


def _to_float_or_none( value ):
    if value is None:
        return None
    try:
        x = float( value )
    except ( TypeError, ValueError ):
        return None
    if not math.isfinite( x ):
        return None
    return x


def scan_distilled_dataset( parquet_files ):
    stats = { 'rows_total' : 0,
              'rows_tokenized' : 0,
              'ms2_rows' : 0,
              'irt_rows' : 0,
              'ccs_rows' : 0,
              'charge_dist_rows' : 0,
              'irt_sum' : 0.0,
              'irt_sum_sq' : 0.0,
              'ccs_sum' : 0.0,
              'ccs_sum_sq' : 0.0,
              'skip_counts' : {}, }

    for filepath in parquet_files:
        pf = pq.ParquetFile( filepath )
        columns = [ 'modified_sequence',
                    'charge_state_dist',
                    'indexed_retention_time',
                    'ccs',
                    'intensities_raw' ]
        for rg_idx in range( pf.metadata.num_row_groups ):
            table = pf.read_row_group( rg_idx, columns=columns )
            mod_seqs = table.column( 'modified_sequence' ).to_pylist()
            charge_dists = table.column( 'charge_state_dist' ).to_pylist()
            irts = table.column( 'indexed_retention_time' ).to_pylist()
            ccss = table.column( 'ccs' ).to_pylist()
            ms2s = table.column( 'intensities_raw' ).to_pylist()

            for mod_seq, charge_dist_value, irt_value, ccs_value, ms2_value in zip( mod_seqs, charge_dists, irts, ccss, ms2s ):
                stats[ 'rows_total' ] += 1
                coded = unimod_to_codedseq( mod_seq, max_len=max_peptide_len, skip_counts=stats[ 'skip_counts' ] )
                if coded is None:
                    continue
                stats[ 'rows_tokenized' ] += 1

                if ms2_value is not None:
                    stats[ 'ms2_rows' ] += 1

                if charge_dist_value is not None:
                    charge_dist_arr = np.asarray( charge_dist_value, dtype='float32' )
                    if charge_dist_arr.shape[0] != charge_dist_len:
                        raise ValueError( 'Unexpected charge distribution length in ' + filepath +
                                          ': got ' + str(charge_dist_arr.shape[0]) +
                                          ', expected ' + str(charge_dist_len) )
                    stats[ 'charge_dist_rows' ] += 1

                irt_float = _to_float_or_none( irt_value )
                if irt_float is not None:
                    stats[ 'irt_rows' ] += 1
                    stats[ 'irt_sum' ] += irt_float
                    stats[ 'irt_sum_sq' ] += irt_float * irt_float

                ccs_float = _to_float_or_none( ccs_value )
                if ccs_float is not None:
                    stats[ 'ccs_rows' ] += 1
                    stats[ 'ccs_sum' ] += ccs_float
                    stats[ 'ccs_sum_sq' ] += ccs_float * ccs_float

    return stats


def _mean_and_std( total, total_sq, count ):
    if count <= 0:
        return 0.0, 1.0
    mean = float( total ) / float( count )
    variance = max( 0.0, float( total_sq ) / float( count ) - mean * mean )
    std = variance ** 0.5
    if std <= 0.0:
        std = 1.0
    return mean, std


def build_scalar_stats( fit_stats, test_stats ):
    irt_mean, irt_std = _mean_and_std( fit_stats[ 'irt_sum' ], fit_stats[ 'irt_sum_sq' ], fit_stats[ 'irt_rows' ] )
    ccs_mean, ccs_std = _mean_and_std( fit_stats[ 'ccs_sum' ], fit_stats[ 'ccs_sum_sq' ], fit_stats[ 'ccs_rows' ] )
    return { 'irt_mean' : float( irt_mean ),
             'irt_std' : float( irt_std ),
             'ccs_mean' : float( ccs_mean ),
             'ccs_std' : float( ccs_std ),
             'fit' : fit_stats,
             'test' : test_stats, }


class ScoutDistilledDataset( IterableDataset ):
    def __init__( self,
                  parquet_files,
                  irt_mean,
                  irt_std,
                  ccs_mean,
                  ccs_std,
                  max_pep_len=max_peptide_len,
                  shuffle_files=False ):
        super().__init__()
        self.parquet_files = sorted( parquet_files )
        self.irt_mean = float( irt_mean )
        self.irt_std = float( irt_std ) if float( irt_std ) > 0 else 1.0
        self.ccs_mean = float( ccs_mean )
        self.ccs_std = float( ccs_std ) if float( ccs_std ) > 0 else 1.0
        self.max_pep_len = int( max_pep_len )
        self.shuffle_files = bool( shuffle_files )
        self.epoch = 0

    def set_epoch( self, epoch ):
        self.epoch = int( epoch )

    def __iter__( self ):
        files = list( self.parquet_files )
        worker_info = get_worker_info()
        if worker_info is not None:
            files = [ f for i, f in enumerate( files ) if i % worker_info.num_workers == worker_info.id ]
        if self.shuffle_files:
            rng = random.Random( self.epoch )
            rng.shuffle( files )

        seq_size = self.max_pep_len + 2
        skip_counts = {}

        for filepath in files:
            pf = pq.ParquetFile( filepath )
            columns = [ 'modified_sequence',
                        'precursor_charge_onehot',
                        'charge_state_dist',
                        'collision_energy_aligned_normed',
                        'indexed_retention_time',
                        'ccs',
                        'intensities_raw' ]
            for rg_idx in range( pf.metadata.num_row_groups ):
                table = pf.read_row_group( rg_idx, columns=columns )
                mod_seqs = table.column( 'modified_sequence' ).to_pylist()
                charges = table.column( 'precursor_charge_onehot' ).to_pylist()
                charge_dists = table.column( 'charge_state_dist' ).to_pylist()
                nces = table.column( 'collision_energy_aligned_normed' ).to_pylist()
                irts = table.column( 'indexed_retention_time' ).to_pylist()
                ccss = table.column( 'ccs' ).to_pylist()
                ms2s = table.column( 'intensities_raw' ).to_pylist()

                for mod_seq, charge_value, charge_dist_value, nce_value, irt_value, ccs_value, ms2_value in zip( mod_seqs, charges, charge_dists, nces, irts, ccss, ms2s ):
                    coded = unimod_to_codedseq( mod_seq, max_len=self.max_pep_len, skip_counts=skip_counts )
                    if coded is None:
                        continue

                    seq_arr = codedseq_to_array( coded, max_size=seq_size )
                    charge_arr = np.asarray( charge_value, dtype='float32' )
                    nce_arr = np.asarray( [ float( nce_value ) ], dtype='float32' )

                    target_arr = np.zeros( SCOUT_TARGET_LEN, dtype='float32' )
                    mask_arr = np.zeros( SCOUT_MASK_LEN, dtype='float32' )

                    if ms2_value is not None:
                        ms2_arr = np.asarray( ms2_value, dtype='float32' )
                        if ms2_arr.shape[0] != ms2_vector_len:
                            raise ValueError( 'Unexpected MS2 vector length in ' + filepath +
                                              ': got ' + str(ms2_arr.shape[0]) +
                                              ', expected ' + str(ms2_vector_len) )
                        target_arr[ :ms2_vector_len ] = ms2_arr
                        mask_arr[ 0 ] = 1.0

                    if charge_dist_value is not None:
                        charge_dist_arr = np.asarray( charge_dist_value, dtype='float32' )
                        if charge_dist_arr.shape[0] != charge_dist_len:
                            raise ValueError( 'Unexpected charge distribution length in ' + filepath +
                                              ': got ' + str(charge_dist_arr.shape[0]) +
                                              ', expected ' + str(charge_dist_len) )
                        target_arr[ SCOUT_CHARGE_DIST_OFFSET : SCOUT_TARGET_LEN ] = charge_dist_arr
                        mask_arr[ 3 ] = 1.0

                    irt_float = _to_float_or_none( irt_value )
                    if irt_float is not None:
                        target_arr[ SCOUT_IRT_OFFSET ] = ( irt_float - self.irt_mean ) / self.irt_std
                        mask_arr[ 1 ] = 1.0

                    ccs_float = _to_float_or_none( ccs_value )
                    if ccs_float is not None:
                        target_arr[ SCOUT_CCS_OFFSET ] = ( ccs_float - self.ccs_mean ) / self.ccs_std
                        mask_arr[ 2 ] = 1.0

                    yield ( torch.from_numpy( seq_arr ),
                            torch.from_numpy( charge_arr ),
                            torch.from_numpy( nce_arr ),
                            torch.from_numpy( target_arr ),
                            torch.from_numpy( mask_arr ), )

        total_skipped = sum( skip_counts.values() )
        if total_skipped > 0:
            parts = [ key + '=' + str(value) for key, value in sorted( skip_counts.items(), key=lambda item: -item[1] ) ]
            print( 'Skipped ' + str(total_skipped) + ' scout sequences: ' + ', '.join( parts ) )
