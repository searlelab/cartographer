import json
import os
import subprocess
from datetime import datetime, timezone

from sculptor_settings import metadata_filename


VALID_SPLITS = [ 'train', 'test', 'all' ]


def parse_slice_splits( raw_value ):
    parts = [ p.strip() for p in raw_value.split( ',' ) if p.strip() != '' ]
    if len( parts ) == 0:
        raise ValueError( '--slice_splits must include at least one of: ' + ', '.join( VALID_SPLITS ) )

    seen = set()
    result = []
    for split in parts:
        if split not in VALID_SPLITS:
            raise ValueError( 'Invalid split in --slice_splits: ' + split )
        if split not in seen:
            seen.add( split )
            result.append( split )
    return result


def resolve_slice_input_csv( dataset_root, explicit_csv=None ):
    if explicit_csv is not None:
        if not os.path.isfile( explicit_csv ):
            raise FileNotFoundError( 'slice input CSV not found: ' + explicit_csv )
        return os.path.abspath( explicit_csv )

    metadata_path = os.path.join( dataset_root, metadata_filename )
    if not os.path.isfile( metadata_path ):
        raise FileNotFoundError( 'Could not find metadata JSON: ' + metadata_path )

    with open( metadata_path, 'r' ) as f:
        metadata = json.load( f )

    input_csv = metadata.get( 'input_csv', None )
    if input_csv is None or str(input_csv).strip() == '':
        raise ValueError( 'Metadata missing input_csv field: ' + metadata_path )
    if not os.path.isfile( input_csv ):
        raise FileNotFoundError( 'Metadata input_csv does not exist on this machine: ' + str(input_csv) )
    return os.path.abspath( input_csv )


def run_slice_diagnostics( python_executable,
                           src_dir,
                           dataset_root,
                           output_dir,
                           final_model_paths,
                           slice_input_csv=None,
                           slice_splits='train,test',
                           slice_max_rows=None, ):
    splits = parse_slice_splits( slice_splits )
    input_csv = resolve_slice_input_csv( dataset_root, explicit_csv=slice_input_csv )

    diagnostics_script = os.path.join( src_dir, 'sculptor_slice_diagnostics.py' )
    if not os.path.isfile( diagnostics_script ):
        raise FileNotFoundError( 'Diagnostics script not found: ' + diagnostics_script )

    log_path = os.path.join( output_dir, 'slice_diagnostics.log' )
    markdown_paths = []

    with open( log_path, 'w' ) as log:
        log.write( 'Sculptor slice diagnostics log\n' )
        log.write( 'generated_at_utc=' + datetime.now( timezone.utc ).strftime( '%Y-%m-%dT%H:%M:%SZ' ) + '\n' )
        log.write( 'dataset_root=' + os.path.abspath( dataset_root ) + '\n' )
        log.write( 'input_csv=' + input_csv + '\n' )
        log.write( 'splits=' + ','.join( splits ) + '\n' )
        if slice_max_rows is not None:
            log.write( 'slice_max_rows=' + str(slice_max_rows) + '\n' )
        for path in final_model_paths:
            log.write( 'final_model_path=' + os.path.abspath( path ) + '\n' )
        log.write( '\n' )

        for split in splits:
            output_md = os.path.join( output_dir, 'slice_diagnostics_' + split + '.md' )
            cmd = [ python_executable,
                    diagnostics_script,
                    '--input_csv', input_csv,
                    '--split', split,
                    '--output_file', output_md ]
            if slice_max_rows is not None:
                cmd += [ '--max_rows', str(slice_max_rows) ]

            log.write( '[' + split + '] command=' + ' '.join( cmd ) + '\n' )
            result = subprocess.run( cmd, text=True, capture_output=True )
            log.write( '[' + split + '] return_code=' + str(result.returncode) + '\n' )
            if result.stdout is not None and result.stdout != '':
                log.write( '[' + split + '] stdout_begin\n' )
                log.write( result.stdout )
                if not result.stdout.endswith( '\n' ):
                    log.write( '\n' )
                log.write( '[' + split + '] stdout_end\n' )
            if result.stderr is not None and result.stderr != '':
                log.write( '[' + split + '] stderr_begin\n' )
                log.write( result.stderr )
                if not result.stderr.endswith( '\n' ):
                    log.write( '\n' )
                log.write( '[' + split + '] stderr_end\n' )
            log.write( '\n' )

            if result.returncode != 0:
                raise RuntimeError( 'Slice diagnostics failed for split=' + split +
                                    '. See log: ' + log_path )

            markdown_paths.append( output_md )

    return log_path, markdown_paths
