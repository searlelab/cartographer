import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from sculptor_workflow_utils import run_slice_diagnostics


def parse_dilations( raw ):
    parts = [ p.strip() for p in raw.split( ',' ) if p.strip() != '' ]
    if len( parts ) == 0:
        raise ValueError( 'dilation_schedule must be a comma-separated list of integers' )
    dilations = [ int(p) for p in parts ]
    if min( dilations ) <= 0:
        raise ValueError( 'dilation_schedule values must be >= 1' )
    return dilations


def parse_args( args ):
    src_dir = os.path.dirname( os.path.abspath( __file__ ) )
    repo_dir = os.path.dirname( src_dir )
    timestamp = datetime.now().strftime( '%Y%m%d_%H%M%S' )

    parser = argparse.ArgumentParser(
        description='Run deep multistart for a single locked Sculptor architecture'
    )
    parser.add_argument( '--dataset_root',
                         type=str,
                         required=True,
                         help='Path to Sculptor dataset root' )
    parser.add_argument( '--output_dir',
                         type=str,
                         default=os.path.join( repo_dir, 'models', 'sculptor_locked_sweeps', timestamp ),
                         help='Output directory for this deep run' )
    parser.add_argument( '--output_file',
                         type=str,
                         default='Sculptor_locked_best.pt',
                         help='Output filename for global best checkpoint copy' )
    parser.add_argument( '--design_name',
                         type=str,
                         default=None,
                         help='Optional explicit design name' )
    parser.add_argument( '--embed_dim',
                         type=int,
                         required=True,
                         help='Embedding dimension (for example 48)' )
    parser.add_argument( '--kernel',
                         type=int,
                         required=True,
                         help='Kernel size (for example 7)' )
    parser.add_argument( '--dilation_schedule',
                         type=str,
                         required=True,
                         help='Comma-separated dilation schedule (for example 1,4,8)' )
    parser.add_argument( '--n_starts',
                         type=int,
                         default=30,
                         help='Starts per design (recommended 20-40, default 30)' )
    parser.add_argument( '--parallel_jobs',
                         type=int,
                         default=3,
                         help='Concurrent jobs per design' )
    parser.add_argument( '--device',
                         type=str,
                         default='auto',
                         help='Training device {auto, mps, cuda, cpu}' )
    parser.add_argument( '--num_workers',
                         type=int,
                         default=4,
                         help='DataLoader workers per training job' )
    parser.add_argument( '--patience',
                         type=int,
                         default=None,
                         help='Early stopping patience' )
    parser.add_argument( '--n_epochs',
                         type=int,
                         default=None,
                         help='Epoch count override' )
    parser.add_argument( '--metadata_file',
                         type=str,
                         default=None,
                         help='Optional metadata JSON override' )
    parser.add_argument( '--eval_batch_size',
                         type=int,
                         default=4096,
                         help='Evaluation batch size for MAE/RMSE' )
    parser.add_argument( '--dry_run',
                         action='store_true',
                         help='Print and run multistart in dry-run mode' )
    parser.add_argument( '--skip_slice_diagnostics',
                         action='store_true',
                         help='Skip post-run slice diagnostics logging' )
    parser.add_argument( '--slice_input_csv',
                         type=str,
                         default=None,
                         help='Optional input CSV override for slice diagnostics' )
    parser.add_argument( '--slice_splits',
                         type=str,
                         default='train,test',
                         help='Comma-separated splits for diagnostics {train,test,all}' )
    parser.add_argument( '--slice_max_rows',
                         type=int,
                         default=None,
                         help='Optional max rows for faster diagnostics runs' )
    return parser.parse_args( args )


def build_design( args, dilations ):
    if args.design_name is None:
        d_label = ''.join( [ str(d) for d in dilations ] )
        design_name = 'full_b3_e' + str(args.embed_dim) + '_k' + str(args.kernel) + '_d' + d_label
    else:
        design_name = args.design_name

    return { 'name' : design_name,
             'arch' : { 'embed_dim' : int(args.embed_dim),
                        'n_blocks' : len( dilations ),
                        'kernel' : int(args.kernel),
                        'dilation_schedule' : list( dilations ),
                        'block_variant' : 'full',
                        'bottleneck_ratio' : 0.5, } }


def write_designs_file( output_dir, design ):
    path = os.path.join( output_dir, 'locked_design.json' )
    payload = { 'designs' : [ design ] }
    with open( path, 'w' ) as f:
        json.dump( payload, f, indent=2 )
    return path


def build_command( args, designs_file ):
    src_dir = os.path.dirname( os.path.abspath( __file__ ) )
    multistart = os.path.join( src_dir, 'sculptor_multistart.py' )

    cmd = [ sys.executable,
            multistart,
            '--dataset_root', args.dataset_root,
            '--output_dir', args.output_dir,
            '--output_file', args.output_file,
            '--designs_file', designs_file,
            '--n_starts', str(args.n_starts),
            '--parallel_jobs', str(args.parallel_jobs),
            '--device', args.device,
            '--num_workers', str(args.num_workers),
            '--eval_batch_size', str(args.eval_batch_size) ]

    if args.patience is not None:
        cmd += [ '--patience', str(args.patience) ]
    if args.n_epochs is not None:
        cmd += [ '--n_epochs', str(args.n_epochs) ]
    if args.metadata_file is not None:
        cmd += [ '--metadata_file', args.metadata_file ]
    if args.dry_run:
        cmd += [ '--dry_run' ]

    return cmd


def main():
    args = parse_args( sys.argv[1:] )
    os.makedirs( args.output_dir, exist_ok=True )

    dilations = parse_dilations( args.dilation_schedule )
    design = build_design( args, dilations )
    designs_file = write_designs_file( args.output_dir, design )
    cmd = build_command( args, designs_file )

    print( 'Running locked-architecture Sculptor multistart' )
    print( 'Locked design file: ' + designs_file )
    print( 'Command:' )
    print( '  ' + ' '.join( cmd ) )

    subprocess.run( cmd, check=True )

    if args.dry_run or args.skip_slice_diagnostics:
        return

    src_dir = os.path.dirname( os.path.abspath( __file__ ) )
    final_models = [ os.path.join( args.output_dir, 'global_best.pt' ),
                     os.path.join( args.output_dir, args.output_file ) ]

    print( 'Running post-run slice diagnostics...' )
    log_path, markdown_paths = run_slice_diagnostics( sys.executable,
                                                      src_dir,
                                                      args.dataset_root,
                                                      args.output_dir,
                                                      final_models,
                                                      slice_input_csv=args.slice_input_csv,
                                                      slice_splits=args.slice_splits,
                                                      slice_max_rows=args.slice_max_rows, )
    print( 'Slice diagnostics log: ' + log_path )
    if len( markdown_paths ) == 0:
        print( 'Slice diagnostics did not produce markdown output. See log for details.' )
    else:
        for path in markdown_paths:
            print( 'Slice diagnostics markdown: ' + path )


if __name__ == '__main__':
    main()
