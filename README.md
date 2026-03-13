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

Values are `%RMSE vs model average RMSE` on each model holdout/test set.
`:white_check_mark:` indicates `<120%` and `>1000` PTM occurrences in that model's training data; otherwise `:warning:`.
`Obs` columns are unique modified peptide sequence observations across each model's combined train+test data, collapsed across charge state and NCE.
Rows are reverse-sorted by average `Obs` across the four models (with model-overlap as a secondary sort).

| Modification | Sites | UNIMOD | Chronologer Accuracy | Cartographer Accuracy | Electrician Accuracy | Sculptor Accuracy | Chronologer Peptides | Cartographer Peptides | Electrician Peptides | Sculptor Peptides |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unmodified | - | - | :white_check_mark: 79.1% | :white_check_mark: 100.4% | :white_check_mark: 112.1% | :white_check_mark: 99.4% | 1,548,115 | 728,296 | 513,462 | 1,063,679 |
| TMT6plex | K, N-term | 737 | :warning: 150.7% | :white_check_mark: 97.6% | :white_check_mark: 83.9% | N/A | 166,178 | 529,626 | 526,902 | 0 |
| Oxidation | M, W | 35 | :white_check_mark: 118.8% | :white_check_mark: 97.8% | :white_check_mark: 101.5% | :white_check_mark: 107.9% | 272,341 | 225,089 | 195,502 | 158,199 |
| Carbamidomethyl | C | 4 | :white_check_mark: 108.7% | :white_check_mark: 88.4% | :white_check_mark: 99.3% | :white_check_mark: 118.3% | 343,394 | 171,561 | 166,772 | 167,308 |
| Phospho | S, T, Y | 21 | :warning: 249.7% | :white_check_mark: 96.9% | :white_check_mark: 92.1% | :white_check_mark: 94.7% | 212,361 | 88,915 | 89,771 | 17,434 |
| GlyGly (Ub) | K | 121 | :white_check_mark: 58.0% | :white_check_mark: 86.7% | :warning: 122.2% | :white_check_mark: 93.6% | 88,656 | 45,738 | 45,861 | 15,178 |
| Acetyl | K, N-term | 1 | :warning: 181.2% | :white_check_mark: 100.1% | :white_check_mark: 92.7% | :white_check_mark: 106.6% | 23,190 | 37,567 | 37,620 | 12,842 |
| Methyl | K, R | 34 | :warning: 221.7% | :white_check_mark: 91.2% | :white_check_mark: 97.1% | :warning: 92.2% | 89 | 15,179 | 15,637 | 323 |
| Pyro-Glu | Q (N-term), E (N-term) | 28, 27 | :white_check_mark: 115.2% | :white_check_mark: 113.3% | :white_check_mark: 58.4% | N/A | 4,825 | 6,384 | 6,233 | 0 |
| Deamidation | N, Q, R | 7 | N/A | :white_check_mark: 89.8% | :white_check_mark: 111.7% | :warning: 123.4% | 0 | 2,230 | 2,256 | 2,333 |
| HexNAc | N, S, T | 43 | N/A | :white_check_mark: 88.9% | :warning: 144.5% | :warning: 106.4% | 0 | 2,847 | 2,990 | 99 |
| Malonyl | K | 747 | N/A | N/A | N/A | :white_check_mark: 67.3% | 0 | 0 | 0 | 4,173 |
| TMT0 | K, N-term | 739 | :warning: 148.2% | N/A | N/A | N/A | 2,515 | 0 | 0 | 0 |
| Succinyl | K | 64 | :white_check_mark: 42.5% | N/A | N/A | :warning: 75.3% | 1,125 | 0 | 0 | 199 |
| Dimethyl | K, R | 36 | :warning: 308.9% | N/A | N/A | :warning: 88.8% | 137 | 0 | 0 | 323 |
| Cysteinyl | C | 312 | N/A | N/A | N/A | :warning: 65.6% | 0 | 0 | 0 | 427 |
| Trimethyl | K | 37 | :warning: 445.1% | N/A | N/A | :warning: 64.0% | 39 | 0 | 0 | 177 |
| Glutarylation | K | 1848 | N/A | N/A | N/A | :warning: 86.2% | 0 | 0 | 0 | 195 |
| Hydroxyisobutyryl | K | 1849 | N/A | N/A | N/A | :warning: 89.0% | 0 | 0 | 0 | 191 |
| Butyryl | K | 1289 | N/A | N/A | N/A | :warning: 94.4% | 0 | 0 | 0 | 188 |
| Formyl | K | 122 | N/A | N/A | N/A | :warning: 79.2% | 0 | 0 | 0 | 182 |
| Crotonyl | K | 1363 | N/A | N/A | N/A | :warning: 75.1% | 0 | 0 | 0 | 173 |
| Nitro | Y | 354 | N/A | N/A | N/A | :warning: 49.0% | 0 | 0 | 0 | 106 |
| Biotin | K | 3 | N/A | N/A | N/A | :warning: 41.9% | 0 | 0 | 0 | 100 |
| Glycosyl hydroxyproline | P | 408 | N/A | N/A | N/A | :warning: 115.1% | 0 | 0 | 0 | 97 |
| Propionyl | K, N-term | 58 | N/A | N/A | N/A | :warning: 78.3% | 0 | 0 | 0 | 94 |

### Average Model Loss (RMSE Baseline)

- `Chronologer` average `RMSE(HI)` = `1.281231`
- `Cartographer` average `RMSE(fragment_intensity)` = `0.142406`
- `Electrician` average `RMSE(charge_state_dist)` = `0.097929`
- `Sculptor` average `RMSE(CCS)` = `15.720550`
