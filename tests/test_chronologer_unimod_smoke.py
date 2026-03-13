import json
import os
import shutil
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from chronologer_unimod_trainer import train_chronologer_unimod


class ChronologerUnimodSmokeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='chronologer_unimod_smoke_')
        self.data_dir = os.path.join(self.temp_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        self._write_split(
            'train-00000-of-00001.parquet',
            [
                ('[]-PEPTM[UNIMOD:35]IDE-[]', 10.0, 'A'),
                ('[]-PEPTN[UNIMOD:7]IDE-[]', 11.0, 'A'),
                ('[]-PEPTK[UNIMOD:121]IDE-[]', 12.0, 'A'),
                ('[]-PEPTS[UNIMOD:43]IDE-[]', 13.0, 'B'),
            ],
        )
        self._write_split(
            'test-00000-of-00001.parquet',
            [
                ('[]-PEPTC[UNIMOD:4]IDE-[]', 20.0, 'A'),
                ('[]-PEPTT[UNIMOD:21]IDE-[]', 21.0, 'A'),
                ('[]-PEPTY[UNIMOD:21]IDE-[]', 22.0, 'A'),
                ('[]-PEPTQ[UNIMOD:7]IDE-[]', 23.0, 'C'),
            ],
        )
        self._write_split(
            'val-00000-of-00001.parquet',
            [
                ('[]-PEPTW[UNIMOD:35]IDE-[]', 30.0, 'A'),
                ('[]-PEPTN[UNIMOD:43]IDE-[]', 31.0, 'A'),
            ],
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _write_split(self, filename, rows):
        table = pa.table(
            {
                'modified_sequence': [r[0] for r in rows],
                'indexed_retention_time': [r[1] for r in rows],
                'package': [r[2] for r in rows],
            }
        )
        pq.write_table(table, os.path.join(self.data_dir, filename))

    def test_single_epoch_smoke(self):
        out_file = os.path.join(self.temp_dir, 'Chronologer_UNIMOD_smoke.pt')
        best_loss, metadata_path = train_chronologer_unimod(
            dataset_root=self.temp_dir,
            output_file_name=out_file,
            device='cpu',
            num_workers=0,
            start_model=None,
            n_epochs=1,
        )

        self.assertTrue(os.path.isfile(out_file))
        self.assertTrue(os.path.isfile(metadata_path))
        self.assertTrue(best_loss == best_loss)  # NaN guard

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.assertIn('source_vocabulary', metadata)
        self.assertIn('best_test_loss', metadata)
        self.assertEqual(metadata['columns']['target'], 'indexed_retention_time')
        self.assertEqual(metadata['columns']['source'], 'package')
        self.assertEqual(len(metadata['splits']['train_files']), 1)
        self.assertEqual(len(metadata['splits']['val_files']), 1)
        self.assertEqual(len(metadata['splits']['fit_files']), 2)
        self.assertEqual(metadata['scan']['fit']['rows_total'], 6)
        self.assertEqual(metadata['scan']['val']['rows_total'], 2)


if __name__ == '__main__':
    unittest.main()
