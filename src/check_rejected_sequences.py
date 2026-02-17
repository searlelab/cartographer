
"""Scan Prospect-PTMs MS2 parquet files for rejected sequences and check
whether they match known recoverable patterns."""

import glob, os, re, sys
from collections import Counter

import pyarrow.parquet as pq

from tensorize import unimod_to_codedseq
from cartographer_settings import max_peptide_len


def classify_rejection( modified_sequence ):
	"""Return a classification string if the sequence matches a known
	recoverable pattern, otherwise None."""
	parts = modified_sequence.split( '-', 2 )
	if len(parts) != 3:
		return None

	nterm_part, body, cterm_part = parts

	# Pattern 1: Q[UNIMOD:28] on first body residue (pyro-glu Q on residue, not nterm)
	if body.startswith( 'Q[UNIMOD:28]' ):
		return 'Q[UNIMOD:28]_at_nterm_residue'

	# Pattern 2: E[UNIMOD:27] on first body residue (pyro-glu E on residue, not nterm)
	if body.startswith( 'E[UNIMOD:27]' ):
		return 'E[UNIMOD:27]_at_nterm_residue'

	# Pattern 3: Stacked mods involving C[UNIMOD:4][UNIMOD:28] = pyro-carbamidomethyl
	# Could be on the first residue or anywhere. Check if body has C with stacked mods
	# that include UNIMOD:4 + UNIMOD:28 (or UNIMOD:28 + UNIMOD:4)
	if 'C[UNIMOD:4][UNIMOD:28]' in body or 'C[UNIMOD:28][UNIMOD:4]' in body:
		return 'C_pyro_carbamidomethyl_stacked'

	# Also check for C[UNIMOD:26] which is the single-tag form (Pyro-carbamidomethyl)
	if 'C[UNIMOD:26]' in body:
		return 'C[UNIMOD:26]_pyro_cam'

	# Pattern 4: R[UNIMOD:7] (deamidation on R)
	if 'R[UNIMOD:7]' in body:
		return 'R[UNIMOD:7]_deamidation'

	return None


def scan_dataset( dataset_root ):
	"""Scan all parquet files and classify rejected sequences."""
	pattern = os.path.join( dataset_root, 'data', '*.parquet' )
	files = sorted( glob.glob( pattern ) )
	if not files:
		print( 'No parquet files found at ' + pattern )
		sys.exit( 1 )
	print( 'Scanning ' + str(len(files)) + ' parquet files...' )

	total_rows = 0
	total_accepted = 0
	skip_counts = Counter()
	recoverable_counts = Counter()
	# Keep a few examples per recoverable pattern
	examples = {}

	for fi, filepath in enumerate( files ):
		basename = os.path.basename( filepath )
		pf = pq.ParquetFile( filepath )

		for rg_idx in range( pf.metadata.num_row_groups ):
			table = pf.read_row_group( rg_idx, columns=['modified_sequence'] )
			mod_seqs = table.column( 'modified_sequence' ).to_pylist()

			for seq in mod_seqs:
				total_rows += 1
				coded = unimod_to_codedseq( seq, max_len=max_peptide_len, skip_counts=skip_counts )
				if coded is not None:
					total_accepted += 1
				else:
					classification = classify_rejection( seq )
					if classification is not None:
						recoverable_counts[classification] += 1
						if classification not in examples or len(examples[classification]) < 3:
							examples.setdefault( classification, [] ).append( seq )

		if (fi + 1) % 10 == 0 or fi + 1 == len(files):
			print( '  ... processed ' + str(fi + 1) + ' / ' + str(len(files)) +
			       ' files (' + str(total_rows) + ' rows)' )

	total_rejected = total_rows - total_accepted
	print( '\n' + '=' * 60 )
	print( 'SUMMARY' )
	print( '=' * 60 )
	print( 'Total rows:     ' + str(total_rows) )
	print( 'Accepted:       ' + str(total_accepted) )
	print( 'Rejected:       ' + str(total_rejected) )
	print()

	print( 'Rejection reasons (from unimod_to_codedseq):' )
	for reason, count in skip_counts.most_common():
		print( '  ' + str(count).rjust(8) + '  ' + reason )
	print()

	print( 'Recoverable patterns found in rejected sequences:' )
	for pattern, count in recoverable_counts.most_common():
		print( '  ' + str(count).rjust(8) + '  ' + pattern )
		for ex in examples.get( pattern, [] ):
			print( '             example: ' + ex )
	print()

	unrecoverable = total_rejected - sum( recoverable_counts.values() )
	print( 'Recoverable:    ' + str(sum( recoverable_counts.values() )) )
	print( 'Unrecoverable:  ' + str(unrecoverable) )


if __name__ == '__main__':
	dataset_root = sys.argv[1] if len(sys.argv) > 1 else \
		'/Users/searle.brian/Documents/huggingface/data/prospect-ptms-ms2'
	scan_dataset( dataset_root )
