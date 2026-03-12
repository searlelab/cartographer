import argparse
import csv
import glob
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from chronologer_model import initialize_chronologer_model
from constants import max_peptide_len as chronologer_max_len
from constants import seed as chronologer_seed
from constants import validation_fraction as chronologer_validation_fraction
from electrician_model import initialize_electrician_model
from electrician_settings import max_peptide_len as electrician_max_len
from cartographer_model import initialize_cartographer_model
from cartographer_settings import max_peptide_len as cartographer_max_len
from sculptor_model import initialize_sculptor_model
from sculptor_settings import max_peptide_len as sculptor_max_len
from sculptor_settings import metadata_filename as sculptor_metadata_filename
from tensorize import codedseq_to_array as codedseq_to_array_common
from tensorize import modseq_to_codedseq, unimod_to_codedseq as unimod_to_codedseq_common
from sculptor_tensorize import residues as sculptor_default_residues


UNIMOD_PATTERN = re.compile( r'UNIMOD:\d+' )
RESIDUE_UNIMOD_PATTERN = re.compile( r'([A-Z])\[(UNIMOD:\d+)\]' )
RESIDUE_MASS_PATTERN = re.compile( r'([A-Z])\[([^\]]+)\]' )


MOD_ROWS = [
    { 'name' : 'Unmodified', 'sites' : '-', 'unimod' : '-' },
    { 'name' : 'Acetyl', 'sites' : 'K, N-term', 'unimod' : '1' },
    { 'name' : 'Carbamidomethyl', 'sites' : 'C', 'unimod' : '4' },
    { 'name' : 'Deamidation', 'sites' : 'N, Q, R', 'unimod' : '7' },
    { 'name' : 'Dimethyl', 'sites' : 'K, R', 'unimod' : '36' },
    { 'name' : 'GlyGly (Ub)', 'sites' : 'K', 'unimod' : '121' },
    { 'name' : 'HexNAc', 'sites' : 'N, S, T', 'unimod' : '43' },
    { 'name' : 'Methyl', 'sites' : 'K, R', 'unimod' : '34' },
    { 'name' : 'Oxidation', 'sites' : 'M, W', 'unimod' : '35' },
    { 'name' : 'Phospho', 'sites' : 'S, T, Y', 'unimod' : '21' },
    { 'name' : 'Succinyl', 'sites' : 'K', 'unimod' : '64' },
    { 'name' : 'Trimethyl', 'sites' : 'K', 'unimod' : '37' },
    { 'name' : 'Pyro-Glu', 'sites' : 'Q (N-term), E (N-term)', 'unimod' : '28, 27' },
    { 'name' : 'TMT0', 'sites' : 'K, N-term', 'unimod' : '739' },
    { 'name' : 'TMT6plex', 'sites' : 'K, N-term', 'unimod' : '737' },
    { 'name' : 'Biotin', 'sites' : 'K', 'unimod' : '3' },
    { 'name' : 'Butyryl', 'sites' : 'K', 'unimod' : '1289' },
    { 'name' : 'Crotonyl', 'sites' : 'K', 'unimod' : '1363' },
    { 'name' : 'Cysteinyl', 'sites' : 'C', 'unimod' : '312' },
    { 'name' : 'Formyl', 'sites' : 'K', 'unimod' : '122' },
    { 'name' : 'Glutarylation', 'sites' : 'K', 'unimod' : '1848' },
    { 'name' : 'Glycosyl hydroxyproline', 'sites' : 'P', 'unimod' : '408' },
    { 'name' : 'Hydroxyisobutyryl', 'sites' : 'K', 'unimod' : '1849' },
    { 'name' : 'Malonyl', 'sites' : 'K', 'unimod' : '747' },
    { 'name' : 'Nitro', 'sites' : 'Y', 'unimod' : '354' },
    { 'name' : 'Propionyl', 'sites' : 'K, N-term', 'unimod' : '58' },
]

MODEL_ORDER = [ 'Chronologer', 'Cartographer', 'Electrician', 'Sculptor' ]


LEGACY_SCULPTOR_DATASET_ROOT = '/Users/searle.brian/Documents/huggingface/data/IM2Deep_CCS'
LEGACY_SCULPTOR_INPUT_CSV = '/Users/searle.brian/Documents/testing/trainingdata/union_ccs.csv'
LEGACY_ELECTRICIAN_DATASET_ROOT = '/Users/searle.brian/Documents/huggingface/data/prospect-ptms-charge'
LEGACY_CARTOGRAPHER_DATASET_ROOT = '/Users/searle.brian/Documents/huggingface/data/prospect-ptms-ms2'
LEGACY_CHRONOLOGER_DB = '/Users/searle.brian/Documents/projects/chronologer/data/Chronologer_DB_220308.txt'


UNIMOD_TO_MOD = {
    'UNIMOD:1' : 'Acetyl',
    'UNIMOD:3' : 'Biotin',
    'UNIMOD:4' : 'Carbamidomethyl',
    'UNIMOD:7' : 'Deamidation',
    'UNIMOD:21' : 'Phospho',
    'UNIMOD:27' : 'Pyro-Glu',
    'UNIMOD:28' : 'Pyro-Glu',
    'UNIMOD:34' : 'Methyl',
    'UNIMOD:35' : 'Oxidation',
    'UNIMOD:36' : 'Dimethyl',
    'UNIMOD:37' : 'Trimethyl',
    'UNIMOD:43' : 'HexNAc',
    'UNIMOD:58' : 'Propionyl',
    'UNIMOD:64' : 'Succinyl',
    'UNIMOD:121' : 'GlyGly (Ub)',
    'UNIMOD:122' : 'Formyl',
    'UNIMOD:312' : 'Cysteinyl',
    'UNIMOD:354' : 'Nitro',
    'UNIMOD:408' : 'Glycosyl hydroxyproline',
    'UNIMOD:737' : 'TMT6plex',
    'UNIMOD:739' : 'TMT0',
    'UNIMOD:747' : 'Malonyl',
    'UNIMOD:1289' : 'Butyryl',
    'UNIMOD:1363' : 'Crotonyl',
    'UNIMOD:1848' : 'Glutarylation',
    'UNIMOD:1849' : 'Hydroxyisobutyryl',
}

CHRONOLOGER_SUPPORTED = set( [
    'Unmodified',
    'Acetyl',
    'Carbamidomethyl',
    'Deamidation',
    'Dimethyl',
    'GlyGly (Ub)',
    'HexNAc',
    'Methyl',
    'Oxidation',
    'Phospho',
    'Succinyl',
    'Trimethyl',
    'Pyro-Glu',
    'TMT0',
    'TMT6plex',
] )

SCULPTOR_SUPPORTED = set( [ 'Unmodified' ] + sorted( set( UNIMOD_TO_MOD.values() ) - set( [ 'Pyro-Glu', 'TMT0', 'TMT6plex' ] ) ) )


class MSEStats( object ):
    def __init__( self ):
        self.sum_mse = 0.0
        self.count = 0

    def update( self, mse_value ):
        self.sum_mse += float( mse_value )
        self.count += 1

    def rmse( self ):
        if self.count == 0:
            return None
        return math.sqrt( self.sum_mse / self.count )


def parse_args( args ):
    parser = argparse.ArgumentParser( description='Compute per-PTM %RMSE (vs model average RMSE) across models' )
    parser.add_argument( '--sculptor_model',
                         type=str,
                         default='models/Sculptor_20260311095327.pt',
                         help='Sculptor checkpoint path' )
    parser.add_argument( '--electrician_model',
                         type=str,
                         default='models/Electrician_20260225110528.pt',
                         help='Electrician checkpoint path' )
    parser.add_argument( '--cartographer_model',
                         type=str,
                         default='models/Cartographer_20260217162213.pt',
                         help='Cartographer checkpoint path' )
    parser.add_argument( '--chronologer_model',
                         type=str,
                         default='models/Chronologer_20220317200246.pt',
                         help='Chronologer checkpoint path' )
    parser.add_argument( '--hf_data_root',
                         type=str,
                         default=None,
                         help=( 'Optional common data root containing IM2Deep_CCS, prospect-ptms-charge, '
                                'and prospect-ptms-ms2 subfolders' ) )
    parser.add_argument( '--sculptor_dataset_root',
                         type=str,
                         default=None,
                         help='Sculptor dataset root (metadata lookup). Env: SCULPTOR_DATASET_ROOT' )
    parser.add_argument( '--sculptor_input_csv',
                         type=str,
                         default=None,
                         help='Optional Sculptor source CSV (legacy/debug only). Env: SCULPTOR_INPUT_CSV' )
    parser.add_argument( '--electrician_dataset_root',
                         type=str,
                         default=None,
                         help='Electrician parquet dataset root. Env: ELECTRICIAN_DATASET_ROOT' )
    parser.add_argument( '--cartographer_dataset_root',
                         type=str,
                         default=None,
                         help='Cartographer parquet dataset root. Env: CARTOGRAPHER_DATASET_ROOT' )
    parser.add_argument( '--chronologer_db',
                         type=str,
                         default=None,
                         help='Chronologer training database TSV. Env: CHRONOLOGER_DB' )
    parser.add_argument( '--device',
                         type=str,
                         default='auto',
                         help='Inference device {auto,cuda,mps,cpu}' )
    parser.add_argument( '--strict_device',
                         action='store_true',
                         help='Fail if requested non-auto device is unavailable (default: fallback to best available)' )
    parser.add_argument( '--batch_size',
                         type=int,
                         default=2048,
                         help='Inference batch size for vectorized models' )
    parser.add_argument( '--log_every',
                         type=int,
                         default=250000,
                         help='Log per-model progress every N evaluated samples' )
    parser.add_argument( '--output_json',
                         type=str,
                         default='models/ptm_loss_report.json',
                         help='Output JSON path' )
    parser.add_argument( '--output_markdown',
                         type=str,
                         default='models/ptm_loss_report.md',
                         help='Output markdown summary path' )
    return parser.parse_args( args )


def log( message ):
    stamp = datetime.now().strftime( '%Y-%m-%d %H:%M:%S' )
    print( '[' + stamp + '] ' + str( message ), flush=True )


def best_available_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr( torch.backends, 'mps' ) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def resolve_device( raw, strict_device=False ):
    device = str( raw ).strip().lower()
    if device == 'auto':
        return best_available_device()
    if device not in [ 'cpu', 'cuda', 'mps' ]:
        raise ValueError( '--device must be one of {auto,cuda,mps,cpu}' )
    if device == 'cuda' and not torch.cuda.is_available():
        if strict_device:
            raise RuntimeError( 'CUDA requested but not available' )
        fallback = best_available_device()
        log( 'CUDA requested but unavailable; falling back to ' + fallback )
        return fallback
    if device == 'mps':
        if not hasattr( torch.backends, 'mps' ) or not torch.backends.mps.is_available():
            if strict_device:
                raise RuntimeError( 'MPS requested but not available' )
            fallback = best_available_device()
            log( 'MPS requested but unavailable; falling back to ' + fallback )
            return fallback
    return device


def _first_present( values ):
    for value in values:
        if value is None:
            continue
        text = str( value ).strip()
        if text != '':
            return text
    return None


def _resolve_path_setting( explicit_value, env_name=None, derived_value=None, legacy_value=None, required_label='path' ):
    candidate = _first_present( [
        explicit_value,
        os.environ.get( env_name, None ) if env_name is not None else None,
        derived_value,
    ] )
    if candidate is not None:
        return candidate
    if legacy_value is not None and os.path.exists( legacy_value ):
        return legacy_value
    env_hint = '' if env_name is None else ( ' or set env ' + env_name )
    raise ValueError( 'Missing ' + required_label + '. Set CLI flag' + env_hint + '.' )


def resolve_input_paths( args ):
    hf_root = _first_present( [ args.hf_data_root ] )
    sculptor_from_hf = None if hf_root is None else os.path.join( hf_root, 'IM2Deep_CCS' )
    electrician_from_hf = None if hf_root is None else os.path.join( hf_root, 'prospect-ptms-charge' )
    cartographer_from_hf = None if hf_root is None else os.path.join( hf_root, 'prospect-ptms-ms2' )

    args.sculptor_dataset_root = _resolve_path_setting(
        explicit_value=args.sculptor_dataset_root,
        env_name='SCULPTOR_DATASET_ROOT',
        derived_value=sculptor_from_hf,
        legacy_value=LEGACY_SCULPTOR_DATASET_ROOT,
        required_label='Sculptor dataset root (--sculptor_dataset_root)',
    )
    args.sculptor_input_csv = _first_present( [
        args.sculptor_input_csv,
        os.environ.get( 'SCULPTOR_INPUT_CSV', None ),
        LEGACY_SCULPTOR_INPUT_CSV if os.path.exists( LEGACY_SCULPTOR_INPUT_CSV ) else None,
    ] )
    args.electrician_dataset_root = _resolve_path_setting(
        explicit_value=args.electrician_dataset_root,
        env_name='ELECTRICIAN_DATASET_ROOT',
        derived_value=electrician_from_hf,
        legacy_value=LEGACY_ELECTRICIAN_DATASET_ROOT,
        required_label='Electrician dataset root (--electrician_dataset_root)',
    )
    args.cartographer_dataset_root = _resolve_path_setting(
        explicit_value=args.cartographer_dataset_root,
        env_name='CARTOGRAPHER_DATASET_ROOT',
        derived_value=cartographer_from_hf,
        legacy_value=LEGACY_CARTOGRAPHER_DATASET_ROOT,
        required_label='Cartographer dataset root (--cartographer_dataset_root)',
    )
    args.chronologer_db = _resolve_path_setting(
        explicit_value=args.chronologer_db,
        env_name='CHRONOLOGER_DB',
        derived_value=None,
        legacy_value=LEGACY_CHRONOLOGER_DB,
        required_label='Chronologer DB (--chronologer_db)',
    )

    return args


def safe_abs_path( path ):
    return os.path.abspath( path )


def to_mod_occurrences_unimod( modified_sequence ):
    parts = modified_sequence.split( '-', 2 )
    tags = []
    if len( parts ) == 3:
        nterm_part, body, _ = parts
        if nterm_part not in [ '[]', '' ]:
            nterm_tag = nterm_part.strip( '[]' )
            if nterm_tag != '':
                tags.append( nterm_tag )
        tags.extend( UNIMOD_PATTERN.findall( body ) )
    else:
        tags.extend( UNIMOD_PATTERN.findall( modified_sequence ) )

    out = []
    for tag in tags:
        if tag in UNIMOD_TO_MOD:
            out.append( UNIMOD_TO_MOD[ tag ] )
    return out


def build_sculptor_token_to_mod( metadata ):
    tokenizer = metadata.get( 'tokenizer', {} ) if isinstance( metadata, dict ) else {}
    token_to_mod = {}

    residue_entries = tokenizer.get( 'residue_unimod_map', [] )
    for entry in residue_entries:
        if not isinstance( entry, dict ):
            continue
        token = str( entry.get( 'token', '' ) )
        unimod = str( entry.get( 'unimod', '' ) )
        if token == '':
            continue
        mod_name = UNIMOD_TO_MOD.get( unimod, None )
        if mod_name is not None:
            token_to_mod[ token ] = mod_name

    nterm_entries = tokenizer.get( 'nterm_unimod_map', {} )
    if isinstance( nterm_entries, dict ):
        for unimod, token in nterm_entries.items():
            token_text = str( token )
            mod_name = UNIMOD_TO_MOD.get( str(unimod), None )
            if token_text != '' and mod_name is not None:
                token_to_mod[ token_text ] = mod_name

    residues = tokenizer.get( 'residues', None )
    if not isinstance( residues, list ) or len( residues ) == 0:
        residues = list( sculptor_default_residues )

    return token_to_mod, residues


def to_mod_occurrences_sculptor_tokens( seq_tokens, token_to_mod, residues ):
    out = []
    max_token = len( residues )
    for raw_value in seq_tokens:
        try:
            token_index = int( raw_value )
        except Exception:
            continue
        if token_index == 0:
            break
        if token_index < 1 or token_index > max_token:
            continue
        token_char = residues[ token_index - 1 ]
        mod_name = token_to_mod.get( token_char, None )
        if mod_name is not None:
            out.append( mod_name )
    return out


def _starts_with_any( value, prefixes ):
    for prefix in prefixes:
        if value.startswith( prefix ):
            return True
    return False


def classify_chronologer_nterm( mass_text ):
    text = str( mass_text )
    if _starts_with_any( text, [ '+42.01' ] ):
        return 'Acetyl'
    if _starts_with_any( text, [ '+224.1' ] ):
        return 'TMT0'
    if _starts_with_any( text, [ '+229.1' ] ):
        return 'TMT6plex'
    if _starts_with_any( text, [ '-17.02', '-18.01' ] ):
        return 'Pyro-Glu'
    return None


def classify_chronologer_residue( aa, mass_text ):
    text = str( mass_text )
    if aa == 'C' and _starts_with_any( text, [ '+57.02' ] ):
        return 'Carbamidomethyl'
    if aa in [ 'M', 'W' ] and _starts_with_any( text, [ '+15.99' ] ):
        return 'Oxidation'
    if aa in [ 'S', 'T', 'Y' ] and _starts_with_any( text, [ '+79.96' ] ):
        return 'Phospho'
    if aa == 'K' and _starts_with_any( text, [ '+42.01' ] ):
        return 'Acetyl'
    if aa == 'K' and _starts_with_any( text, [ '+100.0', '+100.01' ] ):
        return 'Succinyl'
    if aa == 'K' and _starts_with_any( text, [ '+114.0', '+114.04' ] ):
        return 'GlyGly (Ub)'
    if aa in [ 'K', 'R' ] and _starts_with_any( text, [ '+14.01' ] ):
        return 'Methyl'
    if aa in [ 'K', 'R' ] and _starts_with_any( text, [ '+28.03' ] ):
        return 'Dimethyl'
    if aa == 'K' and _starts_with_any( text, [ '+42.04' ] ):
        return 'Trimethyl'
    if aa in [ 'N', 'Q', 'R' ] and _starts_with_any( text, [ '+0.98' ] ):
        return 'Deamidation'
    if aa in [ 'N', 'S', 'T' ] and _starts_with_any( text, [ '+203.0', '+203.07' ] ):
        return 'HexNAc'
    if aa == 'K' and _starts_with_any( text, [ '+224.1' ] ):
        return 'TMT0'
    if aa == 'K' and _starts_with_any( text, [ '+229.1' ] ):
        return 'TMT6plex'
    if aa == 'Q' and _starts_with_any( text, [ '-17.02' ] ):
        return 'Pyro-Glu'
    if aa == 'E' and _starts_with_any( text, [ '-18.01' ] ):
        return 'Pyro-Glu'
    return None


def to_mod_occurrences_chronologer( modified_sequence ):
    seq = str( modified_sequence )
    out = []

    if seq.startswith( '[' ):
        end = seq.find( ']' )
        if end > 1:
            nterm_mass = seq[ 1 : end ]
            mod = classify_chronologer_nterm( nterm_mass )
            if mod is not None:
                out.append( mod )

    for aa, mass_text in RESIDUE_MASS_PATTERN.findall( seq ):
        mod = classify_chronologer_residue( aa, mass_text )
        if mod is not None:
            out.append( mod )

    return out


def update_occurrence_counter( mod_occurrences, counter ):
    noncanonical = [ m for m in mod_occurrences if m != 'Carbamidomethyl' ]
    if len( noncanonical ) == 0:
        counter[ 'Unmodified' ] += 1
    for mod in mod_occurrences:
        counter[ mod ] += 1


def update_mod_mse( mod_occurrences, sample_mse, by_mod_stats ):
    noncanonical = [ m for m in mod_occurrences if m != 'Carbamidomethyl' ]
    if len( noncanonical ) == 0:
        by_mod_stats[ 'Unmodified' ].update( sample_mse )
    for mod in mod_occurrences:
        by_mod_stats[ mod ].update( sample_mse )


def discover_split_files( dataset_root, split ):
    pattern = os.path.join( dataset_root, 'data', split + '-*.parquet' )
    return sorted( glob.glob( pattern ) )


def load_electrician_arch_overrides( model_path ):
    json_path = os.path.splitext( model_path )[0] + '.preprocessing.json'
    if not os.path.isfile( json_path ):
        return None
    try:
        with open( json_path, 'r' ) as f:
            payload = json.load( f )
        arch = payload.get( 'architecture_overrides', None )
        if isinstance( arch, dict ):
            return arch
    except Exception:
        return None
    return None


def evaluate_sculptor( args, device ):
    start_time = time.time()
    log( '[Sculptor] loading model and metadata' )
    metadata_path = os.path.join( args.sculptor_dataset_root, sculptor_metadata_filename )
    if not os.path.isfile( metadata_path ):
        raise FileNotFoundError( 'Sculptor metadata not found: ' + metadata_path )
    with open( metadata_path, 'r' ) as f:
        metadata = json.load( f )
    ccs_mean = float( metadata.get( 'train_ccs_mean', 0.0 ) )
    ccs_std = float( metadata.get( 'train_ccs_std', 1.0 ) )
    if ccs_std <= 0.0:
        ccs_std = 1.0
    token_to_mod, residues = build_sculptor_token_to_mod( metadata )

    model = initialize_sculptor_model( model_file=args.sculptor_model, map_location='cpu' )
    model = model.to( device )
    model.eval()

    train_counts = Counter()
    eval_stats = MSEStats()
    by_mod_stats = defaultdict( MSEStats )
    skipped_invalid_rows = 0
    processed_eval = 0
    next_log = max( int( args.log_every ), 1 )

    train_files = discover_split_files( args.sculptor_dataset_root, 'train' )
    test_files = discover_split_files( args.sculptor_dataset_root, 'test' )
    if len( train_files ) == 0 or len( test_files ) == 0:
        raise RuntimeError( 'Missing Sculptor train/test parquet files under ' + args.sculptor_dataset_root )

    seq_batch = []
    charge_batch = []
    true_ccs_batch = []
    mods_batch = []

    def flush():
        nonlocal processed_eval
        nonlocal next_log
        if len( seq_batch ) == 0:
            return
        batch_size = len( seq_batch )
        seq_t = torch.as_tensor( np.asarray( seq_batch, dtype='int64' ), dtype=torch.long, device=device )
        charge_t = torch.as_tensor( np.asarray( charge_batch, dtype='float32' ), dtype=torch.float32, device=device )
        true_t = torch.as_tensor( np.asarray( true_ccs_batch, dtype='float32' ), dtype=torch.float32, device=device )
        with torch.no_grad():
            pred_norm = model( seq_t, charge_t ).squeeze( -1 )
            pred_ccs = pred_norm * ccs_std + ccs_mean
            mse = ( pred_ccs - true_t ) ** 2
            mse_np = mse.detach().cpu().numpy()
        for i, mse_value in enumerate( mse_np ):
            eval_stats.update( mse_value )
            update_mod_mse( mods_batch[i], mse_value, by_mod_stats )
        processed_eval += batch_size
        while processed_eval >= next_log:
            current_rmse = eval_stats.rmse()
            rmse_text = 'n/a' if current_rmse is None else format( current_rmse, '.6f' )
            log( '[Sculptor] evaluated=' + str( processed_eval ) +
                 ' rmse=' + rmse_text +
                 ' skipped_invalid_rows=' + str( skipped_invalid_rows ) )
            next_log += max( int( args.log_every ), 1 )
        seq_batch.clear()
        charge_batch.clear()
        true_ccs_batch.clear()
        mods_batch.clear()

    for i, path in enumerate( train_files ):
        log( '[Sculptor] counting train PTMs file ' + str(i + 1) + '/' + str( len(train_files) ) +
             ': ' + os.path.basename( path ) )
        pf = pq.ParquetFile( path )
        for rg_idx in range( pf.metadata.num_row_groups ):
            table = pf.read_row_group( rg_idx, columns=[ 'seq_tokens' ] )
            seq_rows = table.column( 'seq_tokens' ).to_pylist()
            for seq_tokens in seq_rows:
                mod_occ = to_mod_occurrences_sculptor_tokens( seq_tokens, token_to_mod, residues )
                update_occurrence_counter( mod_occ, train_counts )

    for i, path in enumerate( test_files ):
        log( '[Sculptor] evaluating test file ' + str(i + 1) + '/' + str( len(test_files) ) +
             ': ' + os.path.basename( path ) )
        pf = pq.ParquetFile( path )
        for rg_idx in range( pf.metadata.num_row_groups ):
            table = pf.read_row_group( rg_idx, columns=[ 'seq_tokens', 'charge_onehot', 'ccs' ] )
            seq_rows = table.column( 'seq_tokens' ).to_pylist()
            charge_rows = table.column( 'charge_onehot' ).to_pylist()
            ccs_rows = table.column( 'ccs' ).to_pylist()
            for row_idx in range( len( seq_rows ) ):
                seq_tokens = seq_rows[ row_idx ]
                charge_onehot = charge_rows[ row_idx ]
                ccs = ccs_rows[ row_idx ]

                if seq_tokens is None or charge_onehot is None or ccs is None:
                    skipped_invalid_rows += 1
                    continue

                mod_occ = to_mod_occurrences_sculptor_tokens( seq_tokens, token_to_mod, residues )

                seq_batch.append( np.asarray( seq_tokens, dtype='int64' ) )
                charge_batch.append( np.asarray( charge_onehot, dtype='float32' ) )
                true_ccs_batch.append( float( ccs ) )
                mods_batch.append( mod_occ )

                if len( seq_batch ) >= args.batch_size:
                    flush()

    flush()

    elapsed = time.time() - start_time
    log( '[Sculptor] complete in ' + format( elapsed, '.1f' ) + 's; evaluated=' + str( processed_eval ) +
         ' skipped_invalid_rows=' + str( skipped_invalid_rows ) )

    return { 'model_name' : 'Sculptor',
             'average_rmse' : eval_stats.rmse(),
             'train_counts' : dict( train_counts ),
             'ptm_rmse' : { k : v.rmse() for k, v in by_mod_stats.items() if v.count > 0 },
             'n_eval_samples' : int( eval_stats.count ),
             'skipped_invalid_rows' : int( skipped_invalid_rows ),
             'eval_metric' : 'RMSE(CCS)' }


def evaluate_electrician( args, device ):
    start_time = time.time()
    log( '[Electrician] loading model' )
    arch_overrides = load_electrician_arch_overrides( args.electrician_model )
    model = initialize_electrician_model( model_file=args.electrician_model,
                                          arch_overrides=arch_overrides,
                                          map_location='cpu' )
    model = model.to( device )
    model.eval()

    train_counts = Counter()
    eval_stats = MSEStats()
    by_mod_stats = defaultdict( MSEStats )
    skipped_tokenization = 0
    processed_eval = 0
    next_log = max( int( args.log_every ), 1 )

    train_files = discover_split_files( args.electrician_dataset_root, 'train' )
    test_files = discover_split_files( args.electrician_dataset_root, 'test' )
    if len( train_files ) == 0 or len( test_files ) == 0:
        raise RuntimeError( 'Missing Electrician train/test parquet files under ' + args.electrician_dataset_root )

    for i, path in enumerate( train_files ):
        log( '[Electrician] counting train PTMs file ' + str(i + 1) + '/' + str( len(train_files) ) +
             ': ' + os.path.basename( path ) )
        pf = pq.ParquetFile( path )
        for rg_idx in range( pf.metadata.num_row_groups ):
            table = pf.read_row_group( rg_idx, columns=[ 'modified_sequence' ] )
            mod_seqs = table.column( 'modified_sequence' ).to_pylist()
            for mod_seq in mod_seqs:
                mod_occ = to_mod_occurrences_unimod( mod_seq )
                update_occurrence_counter( mod_occ, train_counts )

    seq_batch = []
    true_batch = []
    mods_batch = []

    def flush():
        nonlocal processed_eval
        nonlocal next_log
        if len( seq_batch ) == 0:
            return
        batch_size = len( seq_batch )
        seq_t = torch.as_tensor( np.asarray( seq_batch, dtype='int64' ), dtype=torch.long, device=device )
        true_t = torch.as_tensor( np.asarray( true_batch, dtype='float32' ), dtype=torch.float32, device=device )
        with torch.no_grad():
            pred = model( seq_t )
            mse = torch.mean( ( pred - true_t ) ** 2, dim=1 )
            mse_np = mse.detach().cpu().numpy()
        for i, mse_value in enumerate( mse_np ):
            eval_stats.update( mse_value )
            update_mod_mse( mods_batch[ i ], mse_value, by_mod_stats )
        processed_eval += batch_size
        while processed_eval >= next_log:
            current_rmse = eval_stats.rmse()
            rmse_text = 'n/a' if current_rmse is None else format( current_rmse, '.6f' )
            log( '[Electrician] evaluated=' + str( processed_eval ) +
                 ' rmse=' + rmse_text +
                 ' skipped_tokenization=' + str( skipped_tokenization ) )
            next_log += max( int( args.log_every ), 1 )
        seq_batch.clear()
        true_batch.clear()
        mods_batch.clear()

    for i, path in enumerate( test_files ):
        log( '[Electrician] evaluating test file ' + str(i + 1) + '/' + str( len(test_files) ) +
             ': ' + os.path.basename( path ) )
        pf = pq.ParquetFile( path )
        for rg_idx in range( pf.metadata.num_row_groups ):
            table = pf.read_row_group( rg_idx, columns=[ 'modified_sequence', 'charge_state_dist' ] )
            mod_seqs = table.column( 'modified_sequence' ).to_pylist()
            true_dists = table.column( 'charge_state_dist' ).to_pylist()
            for i in range( len(mod_seqs) ):
                mod_seq = mod_seqs[ i ]
                coded = unimod_to_codedseq_common( mod_seq, max_len=electrician_max_len, skip_counts=None )
                if coded is None:
                    skipped_tokenization += 1
                    continue
                seq_tokens = codedseq_to_array_common( coded, max_size=electrician_max_len + 2 )
                mod_occ = to_mod_occurrences_unimod( mod_seq )

                seq_batch.append( seq_tokens )
                true_batch.append( np.asarray( true_dists[ i ], dtype='float32' ) )
                mods_batch.append( mod_occ )

                if len( seq_batch ) >= args.batch_size:
                    flush()

    flush()

    elapsed = time.time() - start_time
    log( '[Electrician] complete in ' + format( elapsed, '.1f' ) + 's; evaluated=' + str( processed_eval ) +
         ' skipped_tokenization=' + str( skipped_tokenization ) )

    return { 'model_name' : 'Electrician',
             'average_rmse' : eval_stats.rmse(),
             'train_counts' : dict( train_counts ),
             'ptm_rmse' : { k : v.rmse() for k, v in by_mod_stats.items() if v.count > 0 },
             'n_eval_samples' : int( eval_stats.count ),
             'skipped_tokenization' : int( skipped_tokenization ),
             'arch_overrides' : arch_overrides,
             'eval_metric' : 'RMSE(charge_state_dist)' }


def evaluate_cartographer( args, device ):
    start_time = time.time()
    log( '[Cartographer] loading model' )
    model = initialize_cartographer_model( frag_type='beam', model_file=None )
    state = torch.load( args.cartographer_model, map_location='cpu' )
    model.load_state_dict( state, strict=True )
    model = model.to( device )
    model.eval()

    train_counts = Counter()
    eval_stats = MSEStats()
    by_mod_stats = defaultdict( MSEStats )
    skipped_tokenization = 0
    processed_eval = 0
    next_log = max( int( args.log_every ), 1 )

    train_files = discover_split_files( args.cartographer_dataset_root, 'train' )
    test_files = discover_split_files( args.cartographer_dataset_root, 'test' )
    if len( train_files ) == 0 or len( test_files ) == 0:
        raise RuntimeError( 'Missing Cartographer train/test parquet files under ' + args.cartographer_dataset_root )

    for i, path in enumerate( train_files ):
        log( '[Cartographer] counting train PTMs file ' + str(i + 1) + '/' + str( len(train_files) ) +
             ': ' + os.path.basename( path ) )
        pf = pq.ParquetFile( path )
        for rg_idx in range( pf.metadata.num_row_groups ):
            table = pf.read_row_group( rg_idx, columns=[ 'modified_sequence' ] )
            mod_seqs = table.column( 'modified_sequence' ).to_pylist()
            for mod_seq in mod_seqs:
                mod_occ = to_mod_occurrences_unimod( mod_seq )
                update_occurrence_counter( mod_occ, train_counts )

    seq_batch = []
    charge_batch = []
    nce_batch = []
    true_batch = []
    mods_batch = []

    def flush():
        nonlocal processed_eval
        nonlocal next_log
        if len( seq_batch ) == 0:
            return
        batch_size = len( seq_batch )
        seq_t = torch.as_tensor( np.asarray( seq_batch, dtype='int64' ), dtype=torch.long, device=device )
        charge_t = torch.as_tensor( np.asarray( charge_batch, dtype='float32' ), dtype=torch.float32, device=device )
        nce_t = torch.as_tensor( np.asarray( nce_batch, dtype='float32' ), dtype=torch.float32, device=device )
        true_t = torch.as_tensor( np.asarray( true_batch, dtype='float32' ), dtype=torch.float32, device=device )

        with torch.no_grad():
            pred = model( seq_t, charge_t, nce_t )
            valid = true_t >= 0.0
            sq = ( pred - true_t ) ** 2
            valid_counts = valid.sum( dim=1 ).clamp( min=1 )
            mse = torch.sum( sq * valid, dim=1 ) / valid_counts
            mse_np = mse.detach().cpu().numpy()

        for i, mse_value in enumerate( mse_np ):
            eval_stats.update( mse_value )
            update_mod_mse( mods_batch[ i ], mse_value, by_mod_stats )
        processed_eval += batch_size
        while processed_eval >= next_log:
            current_rmse = eval_stats.rmse()
            rmse_text = 'n/a' if current_rmse is None else format( current_rmse, '.6f' )
            log( '[Cartographer] evaluated=' + str( processed_eval ) +
                 ' rmse=' + rmse_text +
                 ' skipped_tokenization=' + str( skipped_tokenization ) )
            next_log += max( int( args.log_every ), 1 )

        seq_batch.clear()
        charge_batch.clear()
        nce_batch.clear()
        true_batch.clear()
        mods_batch.clear()

    for i, path in enumerate( test_files ):
        log( '[Cartographer] evaluating test file ' + str(i + 1) + '/' + str( len(test_files) ) +
             ': ' + os.path.basename( path ) )
        pf = pq.ParquetFile( path )
        for rg_idx in range( pf.metadata.num_row_groups ):
            table = pf.read_row_group( rg_idx,
                                       columns=[ 'modified_sequence',
                                                 'precursor_charge_onehot',
                                                 'collision_energy_aligned_normed',
                                                 'intensities_raw' ] )
            mod_seqs = table.column( 'modified_sequence' ).to_pylist()
            charges = table.column( 'precursor_charge_onehot' ).to_pylist()
            nces = table.column( 'collision_energy_aligned_normed' ).to_pylist()
            intensities = table.column( 'intensities_raw' ).to_pylist()

            for i in range( len(mod_seqs) ):
                mod_seq = mod_seqs[ i ]
                coded = unimod_to_codedseq_common( mod_seq, max_len=cartographer_max_len, skip_counts=None )
                if coded is None:
                    skipped_tokenization += 1
                    continue
                seq_tokens = codedseq_to_array_common( coded, max_size=cartographer_max_len + 2 )
                mod_occ = to_mod_occurrences_unimod( mod_seq )

                seq_batch.append( seq_tokens )
                charge_batch.append( np.asarray( charges[ i ], dtype='float32' ) )
                nce_batch.append( np.asarray( [ float(nces[ i ]) ], dtype='float32' ) )
                true_batch.append( np.asarray( intensities[ i ], dtype='float32' ) )
                mods_batch.append( mod_occ )

                if len( seq_batch ) >= args.batch_size:
                    flush()

    flush()

    elapsed = time.time() - start_time
    log( '[Cartographer] complete in ' + format( elapsed, '.1f' ) + 's; evaluated=' + str( processed_eval ) +
         ' skipped_tokenization=' + str( skipped_tokenization ) )

    return { 'model_name' : 'Cartographer',
             'average_rmse' : eval_stats.rmse(),
             'train_counts' : dict( train_counts ),
             'ptm_rmse' : { k : v.rmse() for k, v in by_mod_stats.items() if v.count > 0 },
             'n_eval_samples' : int( eval_stats.count ),
             'skipped_tokenization' : int( skipped_tokenization ),
             'eval_metric' : 'RMSE(fragment_intensity)' }


def split_chronologer_train_test( db ):
    shuffled = db.sample( frac=1.0, random_state=chronologer_seed ).reset_index( drop=True )
    split_idx = int( np.round( chronologer_validation_fraction * len(shuffled) ) )
    return shuffled.iloc[ split_idx: ].copy(), shuffled.iloc[ :split_idx ].copy()


def evaluate_chronologer( args, device ):
    start_time = time.time()
    log( '[Chronologer] loading model and database' )
    model = initialize_chronologer_model( model_file=None )
    state = torch.load( args.chronologer_model, map_location='cpu' )
    model.load_state_dict( state, strict=True )
    model = model.to( device )
    model.eval()

    db = pd.read_csv( args.chronologer_db, sep='\t', usecols=[ 'PeptideModSeq', 'HI' ] )
    train_db, test_db = split_chronologer_train_test( db )

    train_counts = Counter()
    eval_stats = MSEStats()
    by_mod_stats = defaultdict( MSEStats )
    skipped_tokenization = 0
    processed_eval = 0
    next_log = max( int( args.log_every ), 1 )

    for seq in train_db[ 'PeptideModSeq' ].astype( str ):
        mod_occ = to_mod_occurrences_chronologer( seq )
        update_occurrence_counter( mod_occ, train_counts )

    seq_batch = []
    true_batch = []
    mods_batch = []

    def flush():
        nonlocal processed_eval
        nonlocal next_log
        if len( seq_batch ) == 0:
            return
        batch_size = len( seq_batch )
        seq_t = torch.as_tensor( np.asarray( seq_batch, dtype='int64' ), dtype=torch.long, device=device )
        true_t = torch.as_tensor( np.asarray( true_batch, dtype='float32' ), dtype=torch.float32, device=device )
        with torch.no_grad():
            pred = model( seq_t ).squeeze( -1 )
            mse = ( pred - true_t ) ** 2
            mse_np = mse.detach().cpu().numpy()
        for i, mse_value in enumerate( mse_np ):
            eval_stats.update( mse_value )
            update_mod_mse( mods_batch[ i ], mse_value, by_mod_stats )
        processed_eval += batch_size
        while processed_eval >= next_log:
            current_rmse = eval_stats.rmse()
            rmse_text = 'n/a' if current_rmse is None else format( current_rmse, '.6f' )
            log( '[Chronologer] evaluated=' + str( processed_eval ) +
                 ' rmse=' + rmse_text +
                 ' skipped_tokenization=' + str( skipped_tokenization ) )
            next_log += max( int( args.log_every ), 1 )
        seq_batch.clear()
        true_batch.clear()
        mods_batch.clear()

    for row in test_db.itertuples( index=False ):
        seq = str( row.PeptideModSeq )
        coded = modseq_to_codedseq( seq )
        if coded is None:
            skipped_tokenization += 1
            continue
        seq_tokens = codedseq_to_array_common( coded, max_size=chronologer_max_len + 2 )
        mod_occ = to_mod_occurrences_chronologer( seq )

        seq_batch.append( seq_tokens )
        true_batch.append( float( row.HI ) )
        mods_batch.append( mod_occ )

        if len( seq_batch ) >= args.batch_size:
            flush()

    flush()

    elapsed = time.time() - start_time
    log( '[Chronologer] complete in ' + format( elapsed, '.1f' ) + 's; evaluated=' + str( processed_eval ) +
         ' skipped_tokenization=' + str( skipped_tokenization ) )

    return { 'model_name' : 'Chronologer',
             'average_rmse' : eval_stats.rmse(),
             'train_counts' : dict( train_counts ),
             'ptm_rmse' : { k : v.rmse() for k, v in by_mod_stats.items() if v.count > 0 },
             'n_eval_samples' : int( eval_stats.count ),
             'skipped_tokenization' : int( skipped_tokenization ),
             'holdout_policy' : ( 'Random split following training code: '
                                  'shuffle random_state=' + str(chronologer_seed) +
                                  ', test_fraction=' + str(chronologer_validation_fraction) ),
             'eval_metric' : 'RMSE(HI)' }


def supported_mods_for_model( model_name ):
    if model_name in [ 'Chronologer', 'Cartographer', 'Electrician' ]:
        return CHRONOLOGER_SUPPORTED
    if model_name == 'Sculptor':
        return SCULPTOR_SUPPORTED
    return set( [ 'Unmodified' ] )


def build_table(results_by_model):
    table_rows = []
    for row in MOD_ROWS:
        mod_name = row[ 'name' ]
        out_row = { 'Modification' : mod_name,
                    'Sites' : row[ 'sites' ],
                    'UNIMOD' : row[ 'unimod' ], }
        for model_name in MODEL_ORDER:
            model_result = results_by_model[ model_name ]
            supported = mod_name in supported_mods_for_model( model_name )
            if not supported:
                out_row[ model_name ] = 'N/A'
                continue

            avg_rmse = model_result.get( 'average_rmse', None )
            ptm_rmse = model_result.get( 'ptm_rmse', {} ).get( mod_name, None )
            train_count = int( model_result.get( 'train_counts', {} ).get( mod_name, 0 ) )
            if avg_rmse is None or avg_rmse <= 0.0 or ptm_rmse is None:
                out_row[ model_name ] = 'N/A'
                continue

            pct = 100.0 * float( ptm_rmse ) / float( avg_rmse )
            symbol = '✅' if ( pct < 120.0 and train_count > 1000 ) else '⚠️'
            out_row[ model_name ] = symbol + ' ' + format( pct, '.1f' ) + '%'
        table_rows.append( out_row )
    return table_rows


def build_markdown(results_by_model, table_rows):
    lines = []
    lines.append( '## Supported Modifications' )
    lines.append( '' )
    lines.append( 'Values are `%RMSE vs model average RMSE` on each model holdout/test set.' )
    lines.append( "`✅` indicates `<120%` and `>1000` PTM occurrences in that model's training split; otherwise `⚠️`." )
    lines.append( '' )

    headers = [ 'Modification', 'Sites', 'UNIMOD', 'Chronologer', 'Cartographer', 'Electrician', 'Sculptor' ]
    lines.append( '| ' + ' | '.join( headers ) + ' |' )
    lines.append( '| ' + ' | '.join( [ '---' ] * len(headers) ) + ' |' )
    for row in table_rows:
        values = [ row.get( h, '' ) for h in headers ]
        lines.append( '| ' + ' | '.join( values ) + ' |' )

    lines.append( '' )
    lines.append( '### Average Model Loss (RMSE Baseline)' )
    lines.append( '' )
    for model_name in MODEL_ORDER:
        result = results_by_model[ model_name ]
        avg_rmse = result.get( 'average_rmse', None )
        metric = result.get( 'eval_metric', 'RMSE' )
        if avg_rmse is None:
            lines.append( '- `' + model_name + '` : N/A' )
        else:
            lines.append( '- `' + model_name + '` average `' + metric + '` = `' +
                          format( float(avg_rmse), '.6f' ) + '`' )
    lines.append( '' )
    return '\n'.join( lines )


def write_json(output_path, payload):
    os.makedirs( os.path.dirname( output_path ), exist_ok=True )
    with open( output_path, 'w' ) as f:
        json.dump( payload, f, indent=2 )


def write_text(output_path, text):
    os.makedirs( os.path.dirname( output_path ), exist_ok=True )
    with open( output_path, 'w' ) as f:
        f.write( text )


def main():
    total_start_time = time.time()
    args = parse_args( os.sys.argv[1:] )
    args = resolve_input_paths( args )
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr( torch.backends, 'mps' ) and torch.backends.mps.is_available()
    mps_built = hasattr( torch.backends, 'mps' ) and torch.backends.mps.is_built()
    log( 'Backend availability: cuda=' + str(cuda_available) +
         ', mps_available=' + str(mps_available) +
         ', mps_built=' + str(mps_built) )
    device = resolve_device( args.device, strict_device=args.strict_device )
    log( 'Using device: ' + device + ' (requested=' + str(args.device) + ')' )
    log( 'Input paths:' )
    log( '  sculptor_dataset_root=' + args.sculptor_dataset_root )
    log( '  sculptor_input_csv=' + args.sculptor_input_csv )
    log( '  electrician_dataset_root=' + args.electrician_dataset_root )
    log( '  cartographer_dataset_root=' + args.cartographer_dataset_root )
    log( '  chronologer_db=' + args.chronologer_db )

    required_files = [ args.sculptor_model, args.electrician_model, args.cartographer_model, args.chronologer_model ]
    for path in required_files:
        if not os.path.isfile( path ):
            raise FileNotFoundError( 'Model file not found: ' + path )

    if not os.path.isfile( args.chronologer_db ):
        raise FileNotFoundError( 'Chronologer DB not found: ' + args.chronologer_db )

    results = {}
    results[ 'Sculptor' ] = evaluate_sculptor( args, device )
    results[ 'Electrician' ] = evaluate_electrician( args, device )
    results[ 'Cartographer' ] = evaluate_cartographer( args, device )
    results[ 'Chronologer' ] = evaluate_chronologer( args, device )

    table_rows = build_table( results )
    markdown = build_markdown( results, table_rows )

    payload = { 'generated_at_utc' : datetime.now( timezone.utc ).strftime( '%Y-%m-%dT%H:%M:%SZ' ),
                'device' : device,
                'models' : results,
                'table_rows' : table_rows,
                'markdown' : markdown, }

    write_json( args.output_json, payload )
    write_text( args.output_markdown, markdown )

    elapsed_total = time.time() - total_start_time
    log( 'Wrote JSON: ' + safe_abs_path( args.output_json ) )
    log( 'Wrote Markdown: ' + safe_abs_path( args.output_markdown ) )
    log( 'Total runtime: ' + format( elapsed_total, '.1f' ) + 's' )
    print( '\n' + markdown )


if __name__ == '__main__':
    main()
