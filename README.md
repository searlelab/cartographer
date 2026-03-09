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

Coverage below is based on UNIMOD token mappings in `src/tensorize.py` (Chronologer/Cartographer/Electrician) and `src/sculptor_tensorize.py` (Sculptor). Rows are sorted by overall coverage, then by covered tool set, then alphabetically by modification.

| Modification | Sites | UNIMOD | Chronologer | Cartographer | Electrician | Sculptor | Overall Coverage | Covered By |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Acetyl | K, N-term | 1 | Yes | Yes | Yes | Yes | 4/4 | Chronologer, Cartographer, Electrician, Sculptor |
| Carbamidomethyl | C | 4 | Yes | Yes | Yes | Yes | 4/4 | Chronologer, Cartographer, Electrician, Sculptor |
| Deamidation | N, Q, R | 7 | Yes | Yes | Yes | Yes | 4/4 | Chronologer, Cartographer, Electrician, Sculptor |
| Dimethyl | K, R | 36 | Yes | Yes | Yes | Yes | 4/4 | Chronologer, Cartographer, Electrician, Sculptor |
| GlyGly (Ub) | K | 121 | Yes | Yes | Yes | Yes | 4/4 | Chronologer, Cartographer, Electrician, Sculptor |
| HexNAc | N, S, T | 43 | Yes | Yes | Yes | Yes | 4/4 | Chronologer, Cartographer, Electrician, Sculptor |
| Methyl | K, R | 34 | Yes | Yes | Yes | Yes | 4/4 | Chronologer, Cartographer, Electrician, Sculptor |
| Oxidation | M, W | 35 | Yes | Yes | Yes | Yes | 4/4 | Chronologer, Cartographer, Electrician, Sculptor |
| Phospho | S, T, Y | 21 | Yes | Yes | Yes | Yes | 4/4 | Chronologer, Cartographer, Electrician, Sculptor |
| Succinyl | K | 64 | Yes | Yes | Yes | Yes | 4/4 | Chronologer, Cartographer, Electrician, Sculptor |
| Trimethyl | K | 37 | Yes | Yes | Yes | Yes | 4/4 | Chronologer, Cartographer, Electrician, Sculptor |
| Pyro-Glu | Q (N-term), E (N-term) | 28, 27 | Yes | Yes | Yes | No | 3/4 | Chronologer, Cartographer, Electrician |
| TMT0 | K, N-term | 739 | Yes | Yes | Yes | No | 3/4 | Chronologer, Cartographer, Electrician |
| TMT6plex | K, N-term | 737 | Yes | Yes | Yes | No | 3/4 | Chronologer, Cartographer, Electrician |
| Biotin | K | 3 | No | No | No | Yes | 1/4 | Sculptor |
| Butyryl | K | 1289 | No | No | No | Yes | 1/4 | Sculptor |
| Crotonyl | K | 1363 | No | No | No | Yes | 1/4 | Sculptor |
| Cysteinyl | C | 312 | No | No | No | Yes | 1/4 | Sculptor |
| Formyl | K | 122 | No | No | No | Yes | 1/4 | Sculptor |
| Glutarylation | K | 1848 | No | No | No | Yes | 1/4 | Sculptor |
| Glycosyl hydroxyproline | P | 408 | No | No | No | Yes | 1/4 | Sculptor |
| Hydroxyisobutyryl | K | 1849 | No | No | No | Yes | 1/4 | Sculptor |
| Malonyl | K | 747 | No | No | No | Yes | 1/4 | Sculptor |
| Nitro | Y | 354 | No | No | No | Yes | 1/4 | Sculptor |
| Propionyl | K, N-term | 58 | No | No | No | Yes | 1/4 | Sculptor |
