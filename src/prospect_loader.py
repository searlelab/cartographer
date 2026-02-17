
import glob, os, random
import numpy as np
import pyarrow.parquet as pq

import torch
from torch.utils.data import IterableDataset, get_worker_info

from tensorize import unimod_to_codedseq, codedseq_to_array
from cartographer_settings import max_peptide_len


class ProspectMS2Dataset( IterableDataset ):
	"""Iterable dataset for Prospect-PTMs MS2 parquet shards."""

	def __init__( self, parquet_files, max_pep_len=max_peptide_len, shuffle_files=False, ):
		super().__init__()
		self.parquet_files = sorted( parquet_files )
		self.max_pep_len = max_pep_len
		self.shuffle_files = shuffle_files
		self.epoch = 0

	def set_epoch( self, epoch ):
		self.epoch = epoch

	def __iter__( self ):
		files = list( self.parquet_files )

		# Partition files across DataLoader workers
		worker_info = get_worker_info()
		if worker_info is not None:
			files = [ f for i, f in enumerate(files) if i % worker_info.num_workers == worker_info.id ]

		# Shuffle file order deterministically per epoch
		if self.shuffle_files:
			rng = random.Random( self.epoch )
			rng.shuffle( files )

		seq_size = self.max_pep_len + 2
		skip_counts = {}
		total_rows = 0

		for filepath in files:
			pf = pq.ParquetFile( filepath )

			for rg_idx in range( pf.metadata.num_row_groups ):
				table = pf.read_row_group( rg_idx )

				mod_seqs = table.column( 'modified_sequence' ).to_pylist()
				charges = table.column( 'precursor_charge_onehot' ).to_pylist()
				nces = table.column( 'collision_energy_aligned_normed' ).to_pylist()
				intensities = table.column( 'intensities_raw' ).to_pylist()

				for j in range( len(mod_seqs) ):
					total_rows += 1
					coded = unimod_to_codedseq( mod_seqs[j], max_len=self.max_pep_len, skip_counts=skip_counts )
					if coded is None:
						continue

					seq_arr = codedseq_to_array( coded, max_size=seq_size )
					charge_arr = np.asarray( charges[j], 'float32' )
					nce_arr = np.asarray( [ nces[j] ], 'float32' )
					intensity_arr = np.asarray( intensities[j], 'float32' )
					intensity_arr = np.clip( intensity_arr, -1.0, None )
					weight_arr = np.asarray( [ 1.0 ], 'float32' )

					yield ( torch.from_numpy( seq_arr ),
					        torch.from_numpy( charge_arr ),
					        torch.from_numpy( nce_arr ),
					        torch.from_numpy( intensity_arr ),
					        torch.from_numpy( weight_arr ), )

		# Report skip summary
		total_skipped = sum( skip_counts.values() )
		if total_skipped > 0:
			parts = [ k + '=' + str(v) for k, v in sorted( skip_counts.items(), key=lambda x: -x[1] ) ]
			print( 'Skipped ' + str(total_skipped) + ' of ' + str(total_rows) +
			       ' sequences: ' + ', '.join(parts) )


def discover_split_files( dataset_root, prefix ):
	"""Discover parquet shards matching dataset_root/data/{prefix}-*.parquet."""
	pattern = os.path.join( dataset_root, 'data', prefix + '-*.parquet' )
	files = sorted( glob.glob( pattern ) )
	return files
