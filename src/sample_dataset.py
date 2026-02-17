"""Sample every 100th row from prospect-ptms-ms2 into a small test dataset."""

import os
import pyarrow.parquet as pq
import pyarrow as pa
from prospect_loader import discover_split_files

SRC = '/Users/searle.brian/Documents/huggingface/data/prospect-ptms-ms2'
DST = '/Users/searle.brian/Documents/huggingface/data/sampled-prospect-ptms-ms2'

KEEP_EVERY = 100


def sample_split( split ):
	files = discover_split_files( SRC, split )
	print( split + ': ' + str(len(files)) + ' shards' )

	out_dir = os.path.join( DST, 'data' )
	os.makedirs( out_dir, exist_ok=True )

	total_in = 0
	total_out = 0
	kept_batches = []

	for filepath in files:
		pf = pq.ParquetFile( filepath )

		for rg_idx in range( pf.metadata.num_row_groups ):
			table = pf.read_row_group( rg_idx )
			n = len( table )

			# Figure out which global indices fall on every-100th boundary
			start = total_in
			total_in += n

			# Indices within this batch to keep
			first = (KEEP_EVERY - (start % KEEP_EVERY)) % KEEP_EVERY
			local_indices = list( range( first, n, KEEP_EVERY ) )

			if local_indices:
				kept_batches.append( table.take( local_indices ) )
				total_out += len( local_indices )

	if kept_batches:
		combined = pa.concat_tables( kept_batches )
		out_name = split + '-00000-of-00001.parquet'
		pq.write_table( combined, os.path.join( out_dir, out_name ) )

	print( '  ' + str(total_out) + ' / ' + str(total_in) + ' rows kept' )


if __name__ == '__main__':
	for split in ['train', 'val', 'test']:
		sample_split( split )
	print( 'Done. Output in ' + DST )
