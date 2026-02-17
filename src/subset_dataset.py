"""Extract unmodified +2H peptides from prospect-ptms-ms2 into a subset dataset."""

import os, re
import pyarrow.parquet as pq
import pyarrow as pa
from prospect_loader import discover_split_files

SRC = '/Users/searle.brian/Documents/huggingface/data/prospect-ptms-ms2'
DST = '/Users/searle.brian/Documents/huggingface/data/subset-prospect-ptms-ms2'

UNMOD_PATTERN = re.compile( r'^\[\]-[A-Z]+-\[\]$' )


def keep_row( seq, charge ):
	"""Unmodified sequence with precursor charge +2."""
	return UNMOD_PATTERN.match( seq ) is not None and charge[1] == 1


def filter_split( split ):
	files = discover_split_files( SRC, split )
	print( split + ': ' + str(len(files)) + ' shards' )

	out_dir = os.path.join( DST, 'data' )
	os.makedirs( out_dir, exist_ok=True )

	total_in = 0
	total_out = 0

	for i, filepath in enumerate( files ):
		pf = pq.ParquetFile( filepath )
		kept_batches = []

		for rg_idx in range( pf.metadata.num_row_groups ):
			table = pf.read_row_group( rg_idx )
			total_in += len( table )

			mod_seqs = table.column( 'modified_sequence' ).to_pylist()
			charges = table.column( 'precursor_charge_onehot' ).to_pylist()
			mask = [ keep_row( mod_seqs[j], charges[j] ) for j in range( len(mod_seqs) ) ]
			filtered = table.filter( mask )

			if len( filtered ) > 0:
				kept_batches.append( filtered )

		if kept_batches:
			combined = pa.concat_tables( kept_batches )
			total_out += len( combined )
			out_name = split + '-' + format(i, '05d') + '-of-' + format(len(files), '05d') + '.parquet'
			pq.write_table( combined, os.path.join( out_dir, out_name ) )

	print( '  ' + str(total_out) + ' / ' + str(total_in) + ' rows kept' )


if __name__ == '__main__':
	for split in ['train', 'test']:
		filter_split( split )
	print( 'Done. Output in ' + DST )
