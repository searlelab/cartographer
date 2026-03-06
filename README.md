# Cartographer

Peptide library predictor based on the Chronologer model ([github.com/searlelab/chronologer](https://github.com/searlelab/chronologer)).

Cartographer predicts MS2 fragment ion intensities for peptides with post-translational modifications.
It currently supports HCD fragmentation and predicts six ion channels (y+1, y+2, y+3, b+1, b+2, b+3).

## Requirements

| Package | Version |
| --- | --- |
| Python | 3.10+ |
| PyTorch | 2.0+ |
| NumPy | 1.23+ |
| PyArrow | 10+ |

## Training

Cartographer trains on parquet shards. Download datasets and point `--dataset_root` at the top-level directory containing the `data/` folder with `train-*.parquet` and `test-*.parquet` files.

```
python src/cartographer_trainer.py --dataset_root /path/to/prospect-ptms-ms2 \
                                   --device auto \
                                   --num_workers 4 \
                                   --prtc_report prtc_log.tsv
```

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset_root` | *(required)* | Path to Prospect-PTMs MS2 dataset |
| `--output_file` | `Cartographer_<timestamp>.pt` | Model output filename |
| `--output_dir` | `models/` | Directory for saved models |
| `--device` | `auto` | Training device (`auto`, `mps`, `cuda`, `cpu`) |
| `--num_workers` | `0` | DataLoader worker processes |
| `--prtc_report` | *none* | TSV file to log PRTC peptide predictions per epoch |

The `auto` device setting selects MPS (Apple Silicon), CUDA, or CPU in that order.

## Exporting

After training, export a model to TorchScript with a preprocessing metadata JSON:

```
python src/export_cartographer.py --model_file models/Cartographer_<timestamp>.pt
```

This produces two files alongside the model:
- `<model_name>.torchscript.pt` — traced TorchScript model
- `<model_name>.preprocessing.json` — tokenization rules, input/output shapes, and modification mappings

## Supported Modifications

Cartographer encodes peptide modifications as single-character tokens. The following modifications are supported:

| Modification | Residues | Token |
| --- | --- | --- |
| Carbamidomethyl | C | c |
| Oxidation | M, W | m, w |
| Phospho | S, T, Y | s, t, y |
| Acetyl | K, N-term | a, ^ |
| Methyl | K, R | n/o/p, q/r |
| Succinyl | K | b |
| GlyGly (Ub) | K | u |
| Deamidation | N, Q, R | f, g, k |
| HexNAc | N, S, T | h, i, j |
| TMT0 / TMT6plex | K, N-term | z/x, &/* |
| Pyro-Glu | Q (N-term), E (N-term) | (, ) |

