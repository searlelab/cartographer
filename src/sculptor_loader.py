import glob
import os
import random
import numpy as np
import pyarrow.parquet as pq

import torch
from torch.utils.data import IterableDataset, get_worker_info

from sculptor_settings import max_peptide_len


class SculptorCCSDataset( IterableDataset ):
    """Iterable dataset for Sculptor CCS parquet shards."""

    def __init__( self,
                  parquet_files,
                  ccs_mean,
                  ccs_std,
                  max_pep_len=max_peptide_len,
                  shuffle_files=False, ):
        super().__init__()
        self.parquet_files = sorted( parquet_files )
        self.max_pep_len = max_pep_len
        self.shuffle_files = shuffle_files
        self.epoch = 0
        self.ccs_mean = float( ccs_mean )
        self.ccs_std = float( ccs_std ) if float( ccs_std ) > 0 else 1.0

    def set_epoch( self, epoch ):
        self.epoch = epoch

    def __iter__( self ):
        files = list( self.parquet_files )

        worker_info = get_worker_info()
        if worker_info is not None:
            files = [ f for i, f in enumerate(files) if i % worker_info.num_workers == worker_info.id ]

        if self.shuffle_files:
            rng = random.Random( self.epoch )
            rng.shuffle( files )

        expected_size = self.max_pep_len + 2

        for filepath in files:
            pf = pq.ParquetFile( filepath )

            for rg_idx in range( pf.metadata.num_row_groups ):
                table = pf.read_row_group( rg_idx )

                seq_tokens = table.column( 'seq_tokens' ).to_pylist()
                charge_onehot = table.column( 'charge_onehot' ).to_pylist()
                ccs_values = table.column( 'ccs' ).to_pylist()

                if 'weight' in table.column_names:
                    weights = table.column( 'weight' ).to_pylist()
                else:
                    weights = [ 1.0 ] * len( seq_tokens )

                for i in range( len(seq_tokens) ):
                    seq_arr = np.asarray( seq_tokens[i], 'int64' )
                    if seq_arr.shape[0] != expected_size:
                        raise ValueError( 'Unexpected tokenized sequence length in ' + filepath +
                                          ': got ' + str(seq_arr.shape[0]) +
                                          ', expected ' + str(expected_size) )

                    charge_arr = np.asarray( charge_onehot[i], 'float32' )
                    ccs_norm = ( float( ccs_values[i] ) - self.ccs_mean ) / self.ccs_std
                    target_arr = np.asarray( [ ccs_norm ], 'float32' )
                    weight_arr = np.asarray( [ float(weights[i]) ], 'float32' )

                    yield ( torch.from_numpy( seq_arr ),
                            torch.from_numpy( charge_arr ),
                            torch.from_numpy( target_arr ),
                            torch.from_numpy( weight_arr ), )


def discover_split_files( dataset_root, prefix ):
    pattern = os.path.join( dataset_root, 'data', prefix + '-*.parquet' )
    return sorted( glob.glob( pattern ) )
