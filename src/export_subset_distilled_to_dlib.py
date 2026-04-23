"""Convert the subset distilled parquet dataset into a single DLIB file."""

import argparse
import glob
import os
import struct
import zlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from local_io import append_table_to_dlib, create_dlib
from masses import mass_calc, modseq_to_seq, p_mass


DEFAULT_DATASET_ROOT = '/Users/searle.brian/Documents/huggingface/data/subset_chronologer_suite_distilled'
DEFAULT_OUTPUT_DLIB = '/Users/searle.brian/Documents/huggingface/data/subset_chronologer_suite_distilled/subset_chronologer_suite_distilled.dlib'
DISTILLED_FRAGMENT_CLEAVAGES = 29
DISTILLED_FRAGMENT_VECTOR_LEN = DISTILLED_FRAGMENT_CLEAVAGES * 4

UNMOD_PATTERN = '[]-'

RESIDUE_MASSES = {
    'A' : 71.037113805,
    'C' : 103.009184505,
    'D' : 115.026943065,
    'E' : 129.042593135,
    'F' : 147.068413945,
    'G' : 57.021463735,
    'H' : 137.058911875,
    'I' : 113.084064015,
    'K' : 128.094963050,
    'L' : 113.084064015,
    'M' : 131.040484645,
    'N' : 114.042927470,
    'P' : 97.052763875,
    'Q' : 128.058577540,
    'R' : 156.101111050,
    'S' : 87.032028435,
    'T' : 101.047678505,
    'V' : 99.068413945,
    'W' : 186.079312980,
    'Y' : 163.063328575,
}


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Export subset_chronologer_suite_distilled parquet shards to a DLIB file.' )
    parser.add_argument( '--dataset_root', type=str, default=DEFAULT_DATASET_ROOT,
                         help='Subset distilled dataset root containing data/train-*.parquet and data/test-*.parquet.' )
    parser.add_argument( '--output_dlib', type=str, default=DEFAULT_OUTPUT_DLIB,
                         help='Output DLIB path.' )
    parser.add_argument( '--max_rows', type=int, default=None,
                         help='Optional cap on rows exported for smoke testing.' )
    parser.add_argument( '--min_intensity', type=float, default=0.001,
                         help='Minimum intensity retained in the compressed mass/intensity arrays.' )
    parser.add_argument( '--overwrite', action='store_true',
                         help='Overwrite an existing output DLIB.' )
    return parser.parse_args( args )


def discover_split_files( dataset_root, split ):
    return sorted( glob.glob( os.path.join( dataset_root, 'data', split + '-*.parquet' ) ) )


def ensure_output_path( output_dlib, overwrite ):
    output_dir = os.path.dirname( os.path.abspath( output_dlib ) )
    if output_dir != '':
        os.makedirs( output_dir, exist_ok=True )
    if os.path.isfile( output_dlib ) and not overwrite:
        raise RuntimeError( 'Output DLIB already exists. Use --overwrite to replace it: ' + output_dlib )


def modified_sequence_to_modseq( modified_sequence ):
    text = str( modified_sequence ).strip()
    if not ( text.startswith( '[]-' ) and text.endswith( '-[]' ) ):
        raise ValueError( 'Expected unmodified Prospect sequence, got: ' + text )
    body = text[ 3 : -3 ]
    if '[' in body or ']' in body:
        raise ValueError( 'Expected unmodified body, got: ' + text )
    return body


def compute_y_masses( residue_masses ):
    total_mass = float( np.sum( residue_masses ) ) + 18.0105647
    b_masses = np.cumsum( residue_masses[:-1] )
    return total_mass - b_masses[::-1]


def distilled_fragment_mzs( modseq, precursor_charge ):
    seq = modseq_to_seq( modseq )
    if len( seq ) < 2:
        return np.asarray( [], dtype='float64' )

    residue_masses = np.asarray( [ RESIDUE_MASSES[ aa ] for aa in seq ], dtype='float64' )
    b_masses = np.cumsum( residue_masses[:-1] )[:DISTILLED_FRAGMENT_CLEAVAGES]
    y_masses = compute_y_masses( residue_masses )[:DISTILLED_FRAGMENT_CLEAVAGES]

    y1 = ( y_masses + p_mass ) / 1.0
    y2 = ( y_masses + 2.0 * p_mass ) / 2.0
    b1 = ( b_masses + p_mass ) / 1.0
    b2 = ( b_masses + 2.0 * p_mass ) / 2.0

    if precursor_charge < 2:
        y2 = np.zeros_like( y2 )
        b2 = np.zeros_like( b2 )

    return np.stack( [ y1, y2, b1, b2 ], axis=1 ).reshape( -1 )


def compress_mz_intensity( mzs, intensities, min_intensity ):
    mzs = np.asarray( mzs, dtype='float64' )
    intensities = np.asarray( intensities, dtype='float32' )
    if mzs.shape[0] != intensities.shape[0]:
        raise ValueError( 'MZ/intensity length mismatch: ' + str(mzs.shape[0]) + ' vs ' + str(intensities.shape[0]) )

    mask = intensities >= float( min_intensity )
    mzs = mzs[ mask ]
    intensities = intensities[ mask ]

    ba_mzs = bytearray()
    ba_intensities = bytearray()
    for mz in mzs:
        ba_mzs += bytearray( struct.pack( '>d', float( mz ) ) )
    for intensity in intensities:
        ba_intensities += bytearray( struct.pack( '>f', float( intensity ) ) )

    return {
        'MassArray' : zlib.compress( ba_mzs ),
        'MassEncodedLength' : len( mzs ) * 8,
        'IntensityArray' : zlib.compress( ba_intensities ),
        'IntensityEncodedLength' : len( intensities ) * 4,
    }


def row_to_entry( modified_sequence, charge_onehot, nce_norm, irt_value, intensities_raw, split_name, min_intensity ):
    if intensities_raw is None:
        return None

    precursor_charge = int( np.argmax( np.asarray( charge_onehot ) ) + 1 )
    modseq = modified_sequence_to_modseq( modified_sequence )
    peptide_seq = modseq_to_seq( modseq )

    mzs = distilled_fragment_mzs( modseq, precursor_charge )
    expected_len = mzs.shape[0]
    intensities = np.asarray( intensities_raw, dtype='float32' )
    if intensities.shape[0] < expected_len:
        raise ValueError( 'Unexpected distilled intensity length: ' + str(intensities.shape[0]) +
                          ' < ' + str(expected_len) + ' for ' + modified_sequence )
    intensities = intensities[:expected_len]
    compressed = compress_mz_intensity( mzs, intensities, min_intensity )

    nce_text = format( float( nce_norm ) * 100.0, '.2f' )
    precursor_mz = ( mass_calc( modseq ) + precursor_charge * p_mass ) / precursor_charge

    entry = {
        'PrecursorMz' : float( precursor_mz ),
        'PrecursorCharge' : precursor_charge,
        'PeptideModSeq' : modseq,
        'PeptideSeq' : peptide_seq,
        'Copies' : 1,
        'RTInSeconds' : None if irt_value is None else float( irt_value ),
        'Score' : 0.0,
        'SourceFile' : 'subset_' + split_name + '_nce_' + nce_text,
    }
    entry.update( compressed )
    return entry


def export_split( split_name, parquet_files, output_dlib, max_rows, min_intensity ):
    rows_seen = 0
    rows_written = 0

    for filepath in parquet_files:
        pf = pq.ParquetFile( filepath )
        for rg_idx in range( pf.metadata.num_row_groups ):
            table = pf.read_row_group( rg_idx,
                                       columns=[ 'modified_sequence',
                                                 'precursor_charge_onehot',
                                                 'collision_energy_aligned_normed',
                                                 'indexed_retention_time',
                                                 'intensities_raw' ] )
            mod_seqs = table.column( 'modified_sequence' ).to_pylist()
            charges = table.column( 'precursor_charge_onehot' ).to_pylist()
            nces = table.column( 'collision_energy_aligned_normed' ).to_pylist()
            irts = table.column( 'indexed_retention_time' ).to_pylist()
            intensities = table.column( 'intensities_raw' ).to_pylist()

            entries = []
            for mod_seq, charge, nce_value, irt_value, intensity_value in zip( mod_seqs, charges, nces, irts, intensities ):
                if max_rows is not None and rows_seen >= int( max_rows ):
                    break
                rows_seen += 1
                entry = row_to_entry( mod_seq, charge, nce_value, irt_value, intensity_value, split_name, min_intensity )
                if entry is not None:
                    entries.append( entry )

            if len( entries ) > 0:
                append_table_to_dlib( pd.DataFrame( entries ), 'entries', output_dlib )
                rows_written += len( entries )

            if max_rows is not None and rows_seen >= int( max_rows ):
                return rows_seen, rows_written

    return rows_seen, rows_written


def main():
    args = parse_args( os.sys.argv[1:] )
    ensure_output_path( args.output_dlib, args.overwrite )
    create_dlib( args.output_dlib, overwrite=args.overwrite )

    train_files = discover_split_files( args.dataset_root, 'train' )
    test_files = discover_split_files( args.dataset_root, 'test' )
    if len( train_files ) == 0 and len( test_files ) == 0:
        raise RuntimeError( 'No parquet shards found under ' + args.dataset_root )

    total_seen = 0
    total_written = 0
    remaining = args.max_rows

    for split_name, files in [ ( 'train', train_files ), ( 'test', test_files ) ]:
        if len( files ) == 0:
            continue
        split_seen, split_written = export_split( split_name,
                                                  files,
                                                  args.output_dlib,
                                                  remaining,
                                                  args.min_intensity )
        total_seen += split_seen
        total_written += split_written
        print( split_name + ': seen=' + str(split_seen) + ' written=' + str(split_written) )
        if remaining is not None:
            remaining -= split_seen
            if remaining <= 0:
                break

    print( 'DLIB export complete' )
    print( '  rows_seen=' + str( total_seen ) )
    print( '  rows_written=' + str( total_written ) )
    print( '  output=' + args.output_dlib )


if __name__ == '__main__':
    main()
