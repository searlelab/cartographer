import argparse
import json
import os
import sys
from datetime import datetime, timezone

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from chronologer_unimod_loader import (
    ChronologerIrtParquetDataset,
    build_source_vocabulary,
    discover_split_files,
    scan_parquet_files,
)
from chronologer_unimod_loss import LogL_Loss
from chronologer_unimod_model import initialize_chronologer_unimod_model
from chronologer_unimod_settings import (
    chronologer_max_peptide_len,
    default_modified_sequence_column,
    default_source_column,
    default_target_column,
    hyperparameters,
    progress_tick_rows,
    source_min_rows_per_split,
    train_fdr,
    training_parameters,
)
from chronologer_unimod_tokenizer import tokenizer_metadata
from training_loop import train_model


def parse_args(args):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    default_out_filename = 'Chronologer_UNIMOD_' + timestamp + '.pt'

    parser = argparse.ArgumentParser(
        description='Train Chronologer v1 on UNIMOD parquet iRT data (prospect-ptms-irt).'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='Path to iRT dataset root containing data/train-*.parquet, data/test-*.parquet, and optional data/val-*.parquet',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=default_out_filename,
        help='Model filename',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(src_dir, '..', 'models'),
        help='Directory to save model',
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'mps', 'cuda', 'cpu'],
        default='auto',
        help='Train/eval device {auto, mps, cuda, cpu}',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='DataLoader workers for parquet streaming',
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=None,
        help='Total epoch count, overrides settings (default: use training_parameters)',
    )
    parser.add_argument(
        '--start_model',
        type=str,
        default=None,
        help='Optional checkpoint (.pt) to resume from',
    )
    return parser.parse_args(args)


def _top_counts(counter_dict, n=10):
    items = sorted(counter_dict.items(), key=lambda x: (-x[1], x[0]))
    return items[:n]


def _print_scan_summary(label, scan):
    print(label + ' rows total: ' + str(scan['rows_total']) + ', valid: ' + str(scan['rows_valid']))
    if len(scan['skip_counts']) > 0:
        parts = [k + '=' + str(v) for k, v in _top_counts(scan['skip_counts'], n=12)]
        print(label + ' top skip reasons: ' + ', '.join(parts))


def _serialize_training_parameters():
    serialized = dict(training_parameters)
    optimizer_obj = training_parameters.get('optimizer')
    serialized['optimizer'] = optimizer_obj.__name__ if optimizer_obj is not None else 'None'
    return serialized


def _merge_scans(scans):
    merged_source_counts = {}
    merged_skip_counts = {}
    rows_total = 0
    rows_valid = 0
    for scan in scans:
        rows_total += int(scan.get('rows_total', 0))
        rows_valid += int(scan.get('rows_valid', 0))

        for key, value in scan.get('source_counts', {}).items():
            merged_source_counts[key] = merged_source_counts.get(key, 0) + int(value)

        for key, value in scan.get('skip_counts', {}).items():
            merged_skip_counts[key] = merged_skip_counts.get(key, 0) + int(value)

    return {
        'rows_total': int(rows_total),
        'rows_valid': int(rows_valid),
        'source_counts': dict(sorted(merged_source_counts.items())),
        'skip_counts': dict(sorted(merged_skip_counts.items())),
    }


def _write_metadata(metadata_path,
                    output_file_name,
                    dataset_root,
                    train_files,
                    val_files,
                    fit_files,
                    test_files,
                    train_scan,
                    val_scan,
                    fit_scan,
                    test_scan,
                    source_vocab,
                    best_loss):
    fit_source_counts = fit_scan['source_counts']
    test_source_counts = test_scan['source_counts']

    kept_fit_rows = int(sum(fit_source_counts.get(s, 0) for s in source_vocab))
    kept_test_rows = int(sum(test_source_counts.get(s, 0) for s in source_vocab))

    all_sources = sorted(set(fit_source_counts.keys()).union(set(test_source_counts.keys())))
    dropped_sources = [s for s in all_sources if s not in set(source_vocab)]

    metadata = {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'model_checkpoint': os.path.abspath(output_file_name),
        'dataset_root': os.path.abspath(dataset_root),
        'splits': {
            'train_files': [os.path.abspath(p) for p in train_files],
            'val_files': [os.path.abspath(p) for p in val_files],
            'fit_files': [os.path.abspath(p) for p in fit_files],
            'test_files': [os.path.abspath(p) for p in test_files],
        },
        'columns': {
            'modified_sequence': default_modified_sequence_column,
            'target': default_target_column,
            'source': default_source_column,
        },
        'chronologer_unimod': {
            'max_peptide_len': int(chronologer_max_peptide_len),
            'source_min_rows_per_split': int(source_min_rows_per_split),
            'train_fdr': float(train_fdr),
        },
        'hyperparameters': dict(hyperparameters),
        'training_parameters': _serialize_training_parameters(),
        'tokenizer': tokenizer_metadata(),
        'source_vocabulary': list(source_vocab),
        'source_vocabulary_size': int(len(source_vocab)),
        'source_filtering': {
            'kept_fit_rows': kept_fit_rows,
            'kept_test_rows': kept_test_rows,
            'filtered_fit_rows': int(fit_scan['rows_valid'] - kept_fit_rows),
            'filtered_test_rows': int(test_scan['rows_valid'] - kept_test_rows),
            'dropped_sources': dropped_sources,
        },
        'scan': {
            'train': train_scan,
            'val': val_scan,
            'fit': fit_scan,
            'test': test_scan,
        },
        'best_test_loss': float(best_loss),
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def train_chronologer_unimod(dataset_root,
                             output_file_name,
                             device='auto',
                             num_workers=0,
                             start_model=None,
                             n_epochs=None):
    print('Chronologer UNIMOD parquet training initiated')

    train_files = discover_split_files(dataset_root, 'train')
    val_files = discover_split_files(dataset_root, 'val')
    test_files = discover_split_files(dataset_root, 'test')
    fit_files = train_files + val_files

    if len(train_files) == 0:
        raise RuntimeError('No train parquet files found in ' + dataset_root)
    if len(test_files) == 0:
        raise RuntimeError('No test parquet files found in ' + dataset_root)

    print('Found ' + str(len(train_files)) + ' train shards, ' +
          str(len(val_files)) + ' val shards, ' +
          str(len(test_files)) + ' test shards')
    if len(val_files) > 0:
        print('Using train + val shards for fitting; test shards for evaluation.')
    else:
        print('No val shards found; using train shards only for fitting.')

    print('Scanning train split for valid rows and source counts...')
    train_scan = scan_parquet_files(
        train_files,
        target_column=default_target_column,
        source_column=default_source_column,
        modified_sequence_column=default_modified_sequence_column,
        max_peptide_len=chronologer_max_peptide_len,
    )
    _print_scan_summary('Train', train_scan)

    if len(val_files) > 0:
        print('Scanning val split for valid rows and source counts...')
        val_scan = scan_parquet_files(
            val_files,
            target_column=default_target_column,
            source_column=default_source_column,
            modified_sequence_column=default_modified_sequence_column,
            max_peptide_len=chronologer_max_peptide_len,
        )
    else:
        val_scan = {
            'rows_total': 0,
            'rows_valid': 0,
            'source_counts': {},
            'skip_counts': {},
        }
    _print_scan_summary('Val', val_scan)

    fit_scan = _merge_scans([train_scan, val_scan])
    _print_scan_summary('Fit', fit_scan)

    print('Scanning test split for valid rows and source counts...')
    test_scan = scan_parquet_files(
        test_files,
        target_column=default_target_column,
        source_column=default_source_column,
        modified_sequence_column=default_modified_sequence_column,
        max_peptide_len=chronologer_max_peptide_len,
    )
    _print_scan_summary('Test', test_scan)

    source_vocab = build_source_vocabulary(
        fit_scan['source_counts'],
        test_scan['source_counts'],
        min_rows_per_split=source_min_rows_per_split,
    )
    if len(source_vocab) == 0:
        raise RuntimeError('No sources meet min_rows_per_split=' + str(source_min_rows_per_split))

    source_to_index = {source: i for i, source in enumerate(source_vocab)}
    print('Source vocabulary size: ' + str(len(source_vocab)))

    datasets = {
        'train': ChronologerIrtParquetDataset(
            fit_files,
            source_to_index,
            target_column=default_target_column,
            source_column=default_source_column,
            modified_sequence_column=default_modified_sequence_column,
            max_peptide_len=chronologer_max_peptide_len,
            shuffle_files=True,
        ),
        'test': ChronologerIrtParquetDataset(
            test_files,
            source_to_index,
            target_column=default_target_column,
            source_column=default_source_column,
            modified_sequence_column=default_modified_sequence_column,
            max_peptide_len=chronologer_max_peptide_len,
            shuffle_files=False,
        ),
    }

    model = initialize_chronologer_unimod_model(model_file=start_model, map_location='cpu')
    if start_model:
        print('Loaded start model: ' + start_model)

    loss_fx = LogL_Loss(
        n_sources=len(source_vocab),
        family=training_parameters['loss_family'],
        fdr=train_fdr,
    )

    parameters = list(model.parameters()) + list(loss_fx.parameters())
    optimizer = training_parameters['optimizer'](
        parameters,
        lr=training_parameters['learning_rate'],
    )

    total_epochs = n_epochs if n_epochs is not None else training_parameters['n_epochs']

    print('Ready to begin Chronologer UNIMOD training')
    best_loss = train_model(
        model,
        datasets,
        training_parameters['initial_batch_size'],
        training_parameters['max_batch_size'],
        training_parameters['epochs_to_2x_batch'],
        loss_fx,
        optimizer,
        total_epochs,
        device,
        device,
        output_file_name,
        progress_tick_rows=progress_tick_rows,
        num_workers=num_workers,
        start_epoch=1,
    )

    metadata_path = output_file_name + '.metadata.json'
    _write_metadata(
        metadata_path,
        output_file_name,
        dataset_root,
        train_files,
        val_files,
        fit_files,
        test_files,
        train_scan,
        val_scan,
        fit_scan,
        test_scan,
        source_vocab,
        best_loss,
    )

    print('Best test loss: ' + format(float(best_loss), '.6f'))
    print('Wrote metadata: ' + metadata_path)
    return float(best_loss), metadata_path


def main():
    args = parse_args(sys.argv[1:])
    os.makedirs(args.output_dir, exist_ok=True)

    output_file_name = os.path.abspath(os.path.join(args.output_dir, args.output_file))
    train_chronologer_unimod(
        dataset_root=args.dataset_root,
        output_file_name=output_file_name,
        device=args.device,
        num_workers=args.num_workers,
        start_model=args.start_model,
        n_epochs=args.n_epochs,
    )


if __name__ == '__main__':
    main()
    sys.exit()
