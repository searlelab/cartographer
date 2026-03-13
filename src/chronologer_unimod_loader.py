import glob
import os
import random
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from chronologer_unimod_settings import (
    chronologer_max_peptide_len,
    default_modified_sequence_column,
    default_source_column,
    default_target_column,
)
from chronologer_unimod_tokenizer import codedseq_to_array, unimod_to_chronologer_codedseq


def discover_split_files(dataset_root, split):
    pattern = os.path.join(dataset_root, 'data', split + '-*.parquet')
    return sorted(glob.glob(pattern))


def _to_finite_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _normalize_source(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == '' or text.lower() == 'nan':
        return None
    return text


def scan_parquet_files(parquet_files,
                       target_column=default_target_column,
                       source_column=default_source_column,
                       modified_sequence_column=default_modified_sequence_column,
                       max_peptide_len=chronologer_max_peptide_len):
    source_counts = Counter()
    skip_counts = Counter()
    rows_total = 0
    rows_valid = 0

    required_columns = [modified_sequence_column, target_column, source_column]

    for filepath in parquet_files:
        pf = pq.ParquetFile(filepath)
        available = set(pf.schema_arrow.names)
        missing = [col for col in required_columns if col not in available]
        if len(missing) > 0:
            raise KeyError('Missing required parquet columns in ' + filepath + ': ' + ', '.join(missing))

        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=required_columns)
            mod_seqs = table.column(modified_sequence_column).to_pylist()
            targets = table.column(target_column).to_pylist()
            sources = table.column(source_column).to_pylist()

            for mod_seq, target, source in zip(mod_seqs, targets, sources):
                rows_total += 1

                coded = unimod_to_chronologer_codedseq(mod_seq,
                                                       max_len=max_peptide_len,
                                                       skip_counts=skip_counts)
                if coded is None:
                    continue

                target_value = _to_finite_float(target)
                if target_value is None:
                    skip_counts['invalid_target'] += 1
                    continue

                source_value = _normalize_source(source)
                if source_value is None:
                    skip_counts['invalid_source'] += 1
                    continue

                rows_valid += 1
                source_counts[source_value] += 1

    return {
        'rows_total': int(rows_total),
        'rows_valid': int(rows_valid),
        'source_counts': dict(sorted(source_counts.items())),
        'skip_counts': dict(sorted(skip_counts.items())),
    }


def build_source_vocabulary(train_source_counts, test_source_counts, min_rows_per_split):
    train_keys = set(train_source_counts.keys())
    test_keys = set(test_source_counts.keys())
    shared = sorted(train_keys.intersection(test_keys))
    kept = [
        source
        for source in shared
        if train_source_counts[source] >= min_rows_per_split and test_source_counts[source] >= min_rows_per_split
    ]
    return kept


class ChronologerIrtParquetDataset(IterableDataset):
    def __init__(self,
                 parquet_files,
                 source_to_index,
                 target_column=default_target_column,
                 source_column=default_source_column,
                 modified_sequence_column=default_modified_sequence_column,
                 max_peptide_len=chronologer_max_peptide_len,
                 shuffle_files=False):
        super().__init__()
        self.parquet_files = sorted(parquet_files)
        self.source_to_index = dict(source_to_index)
        self.target_column = target_column
        self.source_column = source_column
        self.modified_sequence_column = modified_sequence_column
        self.max_peptide_len = int(max_peptide_len)
        self.shuffle_files = bool(shuffle_files)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _partition_files(self):
        files = list(self.parquet_files)
        worker_info = get_worker_info()
        if worker_info is not None:
            files = [f for i, f in enumerate(files) if i % worker_info.num_workers == worker_info.id]
        if self.shuffle_files:
            rng = random.Random(self.epoch)
            rng.shuffle(files)
        return files

    def __iter__(self):
        files = self._partition_files()
        seq_size = self.max_peptide_len + 2
        n_sources = len(self.source_to_index)
        columns = [self.modified_sequence_column, self.target_column, self.source_column]

        for filepath in files:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=columns)
                mod_seqs = table.column(self.modified_sequence_column).to_pylist()
                targets = table.column(self.target_column).to_pylist()
                sources = table.column(self.source_column).to_pylist()

                for mod_seq, target, source in zip(mod_seqs, targets, sources):
                    coded = unimod_to_chronologer_codedseq(mod_seq, max_len=self.max_peptide_len)
                    if coded is None:
                        continue

                    target_value = _to_finite_float(target)
                    if target_value is None:
                        continue

                    source_value = _normalize_source(source)
                    if source_value is None:
                        continue

                    source_index = self.source_to_index.get(source_value)
                    if source_index is None:
                        continue

                    seq_array = codedseq_to_array(coded, max_size=seq_size)
                    rt_array = np.asarray([target_value], dtype='float32')
                    source_array = np.zeros(n_sources, dtype='float32')
                    source_array[source_index] = 1.0

                    yield (
                        torch.from_numpy(seq_array),
                        torch.from_numpy(rt_array),
                        torch.from_numpy(source_array),
                    )

