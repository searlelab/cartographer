import os
import shutil
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from chronologer_unimod_loader import (
    ChronologerIrtParquetDataset,
    build_source_vocabulary,
    discover_split_files,
    scan_parquet_files,
)


class ChronologerUnimodLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='chronologer_unimod_loader_test_')
        self.data_dir = os.path.join(self.temp_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        self._write_split(
            'train-00000-of-00001.parquet',
            [
                ('[]-PEPTM[UNIMOD:35]IDE-[]', 10.0, 'A'),
                ('[]-PEPTN[UNIMOD:7]IDE-[]', 11.0, 'A'),
                ('[]-PEPTK[UNIMOD:121]IDE-[]', 12.0, 'B'),
                ('[]-PEPTIDEK[UNIMOD:737][UNIMOD:1]-[]', 13.0, 'A'),
            ],
        )
        self._write_split(
            'test-00000-of-00001.parquet',
            [
                ('[]-PEPTC[UNIMOD:4]IDE-[]', 20.0, 'A'),
                ('[]-PEPTT[UNIMOD:21]IDE-[]', 21.0, 'A'),
                ('[]-PEPTS[UNIMOD:43]IDE-[]', 22.0, 'C'),
                ('[]-PEPTIDEK[UNIMOD:737][UNIMOD:34]-[]', 23.0, 'A'),
            ],
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _write_split(self, filename, rows):
        seqs = [r[0] for r in rows]
        rts = [r[1] for r in rows]
        packages = [r[2] for r in rows]
        table = pa.table(
            {
                'modified_sequence': seqs,
                'indexed_retention_time': rts,
                'package': packages,
            }
        )
        pq.write_table(table, os.path.join(self.data_dir, filename))

    def test_scan_and_source_vocabulary(self):
        train_files = discover_split_files(self.temp_dir, 'train')
        test_files = discover_split_files(self.temp_dir, 'test')

        train_scan = scan_parquet_files(train_files)
        test_scan = scan_parquet_files(test_files)

        self.assertEqual(train_scan['rows_total'], 4)
        self.assertEqual(test_scan['rows_total'], 4)
        self.assertGreater(train_scan['skip_counts'].get('stacked_mods', 0), 0)
        self.assertGreater(test_scan['skip_counts'].get('stacked_mods', 0), 0)

        vocab = build_source_vocabulary(
            train_scan['source_counts'],
            test_scan['source_counts'],
            min_rows_per_split=2,
        )
        self.assertEqual(vocab, ['A'])

    def test_dataset_tensor_shapes_and_source_consistency(self):
        train_files = discover_split_files(self.temp_dir, 'train')
        test_files = discover_split_files(self.temp_dir, 'test')

        train_scan = scan_parquet_files(train_files)
        test_scan = scan_parquet_files(test_files)
        vocab = build_source_vocabulary(
            train_scan['source_counts'],
            test_scan['source_counts'],
            min_rows_per_split=2,
        )
        source_to_index = {s: i for i, s in enumerate(vocab)}

        train_ds = ChronologerIrtParquetDataset(train_files, source_to_index, shuffle_files=False)
        test_ds = ChronologerIrtParquetDataset(test_files, source_to_index, shuffle_files=False)

        train_rows = list(iter(train_ds))
        test_rows = list(iter(test_ds))

        self.assertEqual(len(train_rows), 2)
        self.assertEqual(len(test_rows), 2)

        seq_tensor, rt_tensor, src_tensor = train_rows[0]
        self.assertEqual(tuple(seq_tensor.shape), (52,))
        self.assertEqual(tuple(rt_tensor.shape), (1,))
        self.assertEqual(tuple(src_tensor.shape), (1,))


if __name__ == '__main__':
    unittest.main()

