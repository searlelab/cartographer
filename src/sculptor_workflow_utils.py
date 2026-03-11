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


def _looks_like_windows_abs_path( path ):
    raw = str( path )
    return len( raw ) >= 3 and raw[1] == ':' and raw[2] in [ '\\', '/' ]


def _candidate_csv_paths( dataset_root, metadata_input_csv ):
    parent_dir = os.path.dirname( os.path.abspath( dataset_root ) )
    default_name = 'union_ccs.csv'

    candidates = []
    if metadata_input_csv is not None and str( metadata_input_csv ).strip() != '':
        raw = str( metadata_input_csv ).strip()
        candidates.append( raw )
        if not os.path.isabs( raw ) and not _looks_like_windows_abs_path( raw ):
            candidates.append( os.path.join( dataset_root, raw ) )

        # Handle metadata generated on another OS path style.
        base = raw.replace( '\\', '/' ).split( '/' )[-1]
        if base != '':
            candidates.append( os.path.join( dataset_root, base ) )
            candidates.append( os.path.join( parent_dir, base ) )

    candidates.append( os.path.join( dataset_root, default_name ) )
    candidates.append( os.path.join( parent_dir, default_name ) )

    deduped = []
    seen = set()
    for path in candidates:
        norm = os.path.abspath( path ) if not os.path.isabs( path ) else os.path.normpath( path )
        if norm in seen:
            continue
        seen.add( norm )
        deduped.append( path )
    return deduped


def resolve_slice_input_csv( dataset_root, explicit_csv=None ):
    dataset_root = os.path.abspath( dataset_root )
    resolution_notes = []

    if explicit_csv is not None:
        if not os.path.isfile( explicit_csv ):
            raise FileNotFoundError( 'slice input CSV not found: ' + explicit_csv )
        return os.path.abspath( explicit_csv ), resolution_notes

    metadata_path = os.path.join( dataset_root, metadata_filename )
    metadata_input_csv = None
    if os.path.isfile( metadata_path ):
        with open( metadata_path, 'r' ) as f:
            metadata = json.load( f )
        metadata_input_csv = metadata.get( 'input_csv', None )
        if metadata_input_csv is not None and str( metadata_input_csv ).strip() != '':
            resolution_notes.append( 'metadata_input_csv=' + str( metadata_input_csv ) )
        else:
            resolution_notes.append( 'metadata_input_csv missing in metadata file' )
    else:
        resolution_notes.append( 'metadata file not found: ' + metadata_path )

    candidates = _candidate_csv_paths( dataset_root, metadata_input_csv )
    for candidate in candidates:
        if os.path.isfile( candidate ):
            resolved = os.path.abspath( candidate )
            if metadata_input_csv is not None and str( metadata_input_csv ).strip() != '':
                metadata_abs = os.path.abspath( str( metadata_input_csv ) )
                if metadata_abs != resolved:
                    resolution_notes.append( 'using fallback input_csv=' + resolved )
            return resolved, resolution_notes

    searched = []
    for candidate in candidates:
        if os.path.isabs( candidate ) or _looks_like_windows_abs_path( candidate ):
            searched.append( os.path.normpath( candidate ) )
        else:
            searched.append( os.path.abspath( candidate ) )
    raise FileNotFoundError( 'Could not resolve slice diagnostics input CSV. '
                             'Pass --slice_input_csv with a local union_ccs.csv path. '
                             'Searched: ' + '; '.join( searched ) )


def run_slice_diagnostics( python_executable,
                           src_dir,
                           dataset_root,
                           output_dir,
                           final_model_paths,
                           slice_input_csv=None,
                           slice_splits='train,test',
                           slice_max_rows=None, ):
    splits = parse_slice_splits( slice_splits )

    log_path = os.path.join( output_dir, 'slice_diagnostics.log' )
    markdown_paths = []

    with open( log_path, 'w' ) as log:
        log.write( 'Sculptor slice diagnostics log\n' )
        log.write( 'generated_at_utc=' + datetime.now( timezone.utc ).strftime( '%Y-%m-%dT%H:%M:%SZ' ) + '\n' )
        log.write( 'dataset_root=' + os.path.abspath( dataset_root ) + '\n' )
        log.write( 'splits=' + ','.join( splits ) + '\n' )
        if slice_max_rows is not None:
            log.write( 'slice_max_rows=' + str(slice_max_rows) + '\n' )
        for path in final_model_paths:
            log.write( 'final_model_path=' + os.path.abspath( path ) + '\n' )
        log.write( '\n' )

        try:
            input_csv, resolution_notes = resolve_slice_input_csv( dataset_root, explicit_csv=slice_input_csv )
        except Exception as exc:
            log.write( 'status=SKIPPED\n' )
            log.write( 'reason=Failed to resolve input CSV: ' + str( exc ) + '\n' )
            return log_path, markdown_paths

        log.write( 'input_csv=' + input_csv + '\n' )
        for note in resolution_notes:
            log.write( 'resolution_note=' + note + '\n' )
        log.write( '\n' )

        diagnostics_script = os.path.join( src_dir, 'sculptor_slice_diagnostics.py' )
        if not os.path.isfile( diagnostics_script ):
            log.write( 'status=SKIPPED\n' )
            log.write( 'reason=Diagnostics script not found: ' + diagnostics_script + '\n' )
            return log_path, markdown_paths

        failed_splits = []
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
                failed_splits.append( split )
                continue

            markdown_paths.append( output_md )

        if len( failed_splits ) > 0:
            log.write( 'status=PARTIAL_FAILURE\n' )
            log.write( 'failed_splits=' + ','.join( failed_splits ) + '\n' )
        else:
            log.write( 'status=OK\n' )

    return log_path, markdown_paths
