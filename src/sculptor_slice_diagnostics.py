import argparse
import csv
import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from sculptor_settings import max_peptide_len
from sculptor_tensorize import unimod_to_codedseq


UNIMOD_PATTERN = re.compile( r'UNIMOD:\\d+' )
DEFAULT_INPUT_CSV = '/Users/searle.brian/Documents/testing/trainingdata/union_ccs.csv'


class RunningStats( object ):
    def __init__( self ):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update( self, value ):
        x = float( value )
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std( self ):
        if self.n < 2:
            return 0.0
        return math.sqrt( self.m2 / ( self.n - 1 ) )


def choose_split( modified_sequence, test_fraction ):
    digest = hashlib.md5( modified_sequence.encode( 'utf-8' ) ).hexdigest()
    bucket = int( digest, 16 ) % 10000000
    threshold = int( test_fraction * 10000000 )
    return 'test' if bucket < threshold else 'train'


def parse_args( args ):
    parser = argparse.ArgumentParser(
        description='Summarize Sculptor training-data slice diagnostics'
    )
    parser.add_argument( '--input_csv',
                         type=str,
                         default=DEFAULT_INPUT_CSV,
                         help='Path to union_ccs.csv' )
    parser.add_argument( '--split',
                         type=str,
                         default='train',
                         choices=[ 'train', 'test', 'all' ],
                         help='Which split to summarize' )
    parser.add_argument( '--test_fraction',
                         type=float,
                         default=0.2,
                         help='Split fraction used during dataset prep' )
    parser.add_argument( '--max_rows',
                         type=int,
                         default=None,
                         help='Optional row cap for fast diagnostics' )
    parser.add_argument( '--output_file',
                         type=str,
                         default=None,
                         help='Output markdown path (default next to input csv)' )
    return parser.parse_args( args )


def length_bin( length ):
    if length <= 10:
        return '07-10'
    if length <= 15:
        return '11-15'
    if length <= 20:
        return '16-20'
    if length <= 25:
        return '21-25'
    if length <= 30:
        return '26-30'
    if length <= 40:
        return '31-40'
    return '41-50'


def should_keep_split( split_name, selected ):
    if selected == 'all':
        return True
    return split_name == selected


def write_markdown( output_file,
                    input_csv,
                    selected_split,
                    test_fraction,
                    total_rows,
                    split_selected_rows,
                    accepted_rows,
                    skip_counts,
                    charge_stats,
                    len_counter,
                    len_bin_counter,
                    ptm_counter ):
    with open( output_file, 'w' ) as out:
        out.write( '# Sculptor Slice Diagnostics\n\n' )
        out.write( '- generated_at_utc: ' + datetime.now( timezone.utc ).strftime( '%Y-%m-%dT%H:%M:%SZ' ) + '\n' )
        out.write( '- input_csv: `' + os.path.abspath( input_csv ) + '`\n' )
        out.write( '- selected_split: `' + selected_split + '`\n' )
        out.write( '- test_fraction: `' + str(test_fraction) + '`\n' )
        out.write( '- max_peptide_len: `' + str(max_peptide_len) + '`\n\n' )

        out.write( '## Overview\n\n' )
        out.write( '- rows_scanned: `' + str(total_rows) + '`\n' )
        out.write( '- rows_in_selected_split: `' + str(split_selected_rows) + '`\n' )
        out.write( '- accepted_rows: `' + str(accepted_rows) + '`\n' )
        out.write( '- rejected_rows: `' + str(split_selected_rows - accepted_rows) + '`\n\n' )

        out.write( '## Rejection Reasons\n\n' )
        out.write( '| reason | count |\n' )
        out.write( '| --- | ---: |\n' )
        if len( skip_counts ) == 0:
            out.write( '| (none) | 0 |\n' )
        else:
            for reason, count in sorted( skip_counts.items(), key=lambda kv: (-kv[1], kv[0]) ):
                out.write( '| ' + reason + ' | ' + str(count) + ' |\n' )
        out.write( '\n' )

        out.write( '## Charge Slices\n\n' )
        out.write( '| charge | count | mean_ccs | std_ccs |\n' )
        out.write( '| ---: | ---: | ---: | ---: |\n' )
        for z in sorted( charge_stats ):
            stats = charge_stats[ z ]
            out.write( '| ' + str(z) +
                       ' | ' + str(stats.n) +
                       ' | ' + format( stats.mean, '.6f' ) +
                       ' | ' + format( stats.std(), '.6f' ) +
                       ' |\n' )
        out.write( '\n' )

        out.write( '## Length Bins\n\n' )
        out.write( '| bin | count |\n' )
        out.write( '| --- | ---: |\n' )
        ordered_bins = [ '07-10', '11-15', '16-20', '21-25', '26-30', '31-40', '41-50' ]
        for b in ordered_bins:
            out.write( '| ' + b + ' | ' + str(len_bin_counter.get( b, 0 )) + ' |\n' )
        out.write( '\n' )

        out.write( '## Exact Length Counts\n\n' )
        out.write( '| length | count |\n' )
        out.write( '| ---: | ---: |\n' )
        for length in sorted( len_counter ):
            out.write( '| ' + str(length) + ' | ' + str(len_counter[length]) + ' |\n' )
        out.write( '\n' )

        out.write( '## PTM Presence\n\n' )
        out.write( '| PTM tag | count |\n' )
        out.write( '| --- | ---: |\n' )
        for tag, count in sorted( ptm_counter.items(), key=lambda kv: (-kv[1], kv[0]) ):
            out.write( '| ' + tag + ' | ' + str(count) + ' |\n' )


def main():
    args = parse_args( os.sys.argv[1:] )
    if not os.path.isfile( args.input_csv ):
        raise FileNotFoundError( 'Input CSV not found: ' + args.input_csv )
    if not ( 0.0 < args.test_fraction < 1.0 ):
        raise ValueError( '--test_fraction must be in (0,1)' )

    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join( os.path.dirname( os.path.abspath( args.input_csv ) ),
                                    'sculptor_slice_diagnostics_' + args.split + '.md' )

    total_rows = 0
    split_selected_rows = 0
    accepted_rows = 0
    skip_counts = Counter()
    charge_stats = defaultdict( RunningStats )
    len_counter = Counter()
    len_bin_counter = Counter()
    ptm_counter = Counter()

    with open( args.input_csv, 'r', newline='' ) as handle:
        reader = csv.DictReader( handle )
        required = { 'modified_sequence', 'charge', 'ccs' }
        if not required.issubset( set( reader.fieldnames or [] ) ):
            raise ValueError( 'Input CSV missing one of required columns: ' + str( sorted( required ) ) )

        for row in reader:
            if args.max_rows is not None and total_rows >= args.max_rows:
                break

            total_rows += 1
            modified_sequence = row[ 'modified_sequence' ].strip()
            split_name = choose_split( modified_sequence, args.test_fraction )
            if not should_keep_split( split_name, args.split ):
                continue
            split_selected_rows += 1

            try:
                ccs = float( row[ 'ccs' ] )
                charge = int( row[ 'charge' ] )
            except Exception:
                skip_counts[ 'bad_numeric' ] += 1
                continue

            coded = unimod_to_codedseq( modified_sequence,
                                        max_len=max_peptide_len,
                                        skip_counts=skip_counts )
            if coded is None:
                continue

            accepted_rows += 1
            peptide_len = len( coded ) - 2

            charge_stats[ charge ].update( ccs )
            len_counter[ peptide_len ] += 1
            len_bin_counter[ length_bin( peptide_len ) ] += 1

            tags = set( UNIMOD_PATTERN.findall( modified_sequence ) )
            if len( tags ) == 0:
                ptm_counter[ 'UNMODIFIED' ] += 1
            else:
                for tag in tags:
                    ptm_counter[ tag ] += 1

    write_markdown( output_file,
                    args.input_csv,
                    args.split,
                    args.test_fraction,
                    total_rows,
                    split_selected_rows,
                    accepted_rows,
                    skip_counts,
                    charge_stats,
                    len_counter,
                    len_bin_counter,
                    ptm_counter )

    print( 'Wrote slice diagnostics: ' + output_file )
    print( 'Rows scanned: ' + str(total_rows) +
           ' | selected split rows: ' + str(split_selected_rows) +
           ' | accepted rows: ' + str(accepted_rows) )


if __name__ == '__main__':
    main()
