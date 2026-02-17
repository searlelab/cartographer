# Changelog

## 2026-02-16

Rewrote the training pipeline to use the Prospect-PTMs MS2 dataset and expanded modification support.

### Training pipeline
- Replaced pickle-based data loading with streaming Parquet reader (`prospect_loader.py`) using `ProspectMS2Dataset` (PyArrow-backed `IterableDataset`)
- Training data is now sourced from pre-split `train-*.parquet` / `test-*.parquet` shards (Prospect-PTMs MS2 format)
- Added CLI flags: `--dataset_root`, `--device`, `--num_workers`, `--prtc_report`
- Added automatic device selection (`auto` resolves to MPS > CUDA > CPU)
- Added per-epoch PRTC peptide report callback for monitoring training convergence
- Added progress tick dots during training for large-scale datasets

### Model
- Made HCD ion channel count configurable (`n_ion_channels`, default 6: y+1/2/3, b+1/2/3)
- HCD decoder now flattens output to a fixed-length vector (`ms2_vector_len = 174`)
- Prosit-specific constants (`max_peptide_len = 31`, `n_ion_channels`, `ms2_vector_len`) separated from Chronologer defaults in `cartographer_settings.py`

### Modifications
- Added UNIMOD-based tokenization (`unimod_to_codedseq`) for Prospect-PTMs parquet format
- New modifications: deamidation (N/Q/R), HexNAc glycosylation (N/S/T), tryptophan oxidation (W)
- Added N-terminal pyro-Glu mappings (Q and E) to `nterm_keys`
- Added TMT0 and TMT6plex regex patterns for K-residue modifications
- Added corresponding masses in `masses.py`

### Export
- New `export_cartographer.py` script to trace and save models as TorchScript
- Generates a `preprocessing.json` sidecar with tokenization rules, input/output shapes, and modification mappings
- Includes automated validation comparing Python and TorchScript outputs

### Loss function
- Fixed `l2_norm` and spectral dot-product reduction to work with flattened (1-D per sample) output tensors instead of assuming a fixed number of dimensions

## Initial version

Initial working version of Cartographer with pickle-based training data, HCD/CID model variants, and spectral library generation (`Generate_Library.py`).
