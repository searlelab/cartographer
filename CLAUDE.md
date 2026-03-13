# AGENTS.md
Agent guidance for this repository.

## Read-first
- Before making changes, read `README.md` at least once to understand architecture, module boundaries, and existing patterns.

## Repo layout (high level)
- Single Python project with script-style modules in `src/`.
- Model families (entrypoints + settings + export):
  - `cartographer_*`: MS2 fragment intensity prediction.
  - `electrician_*`: precursor charge distribution prediction.
  - `sculptor_*`: CCS regression + dataset preparation + architecture sweeps.
  - `chronologer_*`: retention-time model/training.
- Shared infrastructure:
  - `core_layers.py`, `training_loop.py`: shared network/training primitives.
  - `tensorize.py`, `sculptor_tensorize.py`: sequence/token/modification encoding.
  - `prospect_loader.py`, `sculptor_loader.py`: parquet-backed iterable datasets.
- Supporting assets:
  - `configs/`: architecture design JSONs for sweep workflows.
  - `models/`: trained checkpoints, TorchScript exports, preprocessing metadata.
  - `fasta/`: FASTA fixtures for library-generation paths.

## Build and test commands
### Fast syntax compile (all Python modules)
python -m py_compile src/*.py

## What “good autonomy” looks like
- Prefer changes in the smallest relevant `src/` modules; avoid broad rewrites across model families unless requested.
- Reuse existing code paths and utilities, especially tokenizers, dataset loaders, export metadata builders, and `train_model`.
- Do not speculate about code you have not opened. If a conclusion depends on specific behavior, open and read the relevant files first (source, tests, build config, scripts).
-  Minimize blast radius:
  - Change the fewest files possible.
  - Keep diffs small and easy to review.
  - Prefer additive changes over rewiring existing behavior.
- Make success checkable:
  - Add or update a runnable test (prefer focused `pytest` tests for pure functions / parsing logic).
  - If no test harness exists for the path, add a deterministic smoke check (CLI `--help`, `--dry_run`, `--max_rows`, or export validation).
  - Ensure outputs are deterministic (stable sorting, fixed seeds, hash-based splits, explicit tolerances).

## Coding standards
- Preserve formatting and conventions already present in nearby files.
- This repo mixes styles (e.g., spaced calls like `foo( x )`, occasional tabs, and newer PEP8-style files); match local style and do not reformat unrelated code.
- Preserve and maintain comments, update comments when behavior changes and use the same comment style as nearby code.
- Keep the existing script pattern where used: `parse_args(...)`, `main()`, and `if __name__ == '__main__':`.

## Design and data-contract rules
- Keep model construction centralized in `initialize_*_model` functions; avoid duplicating architecture assembly in trainers/exporters.
- Keep tokenizer and export metadata in sync:
  - Cartographer/Electrician: update both `tensorize.py` and `export_*` metadata when vocab or mod mappings change.
  - Sculptor: update both `sculptor_tensorize.py` and `export_sculptor.py` metadata when vocab or UNIMOD mappings change.
- Preserve parquet/data contracts expected by loaders and trainers (column names, tensor shapes, and split naming: `train-*.parquet`, `test-*.parquet`).
- Prefer relative/project-local defaults for new paths and expose configurable paths through CLI flags.

## Training and evaluation rules
- Keep `'auto'` device behavior aligned with `training_loop.resolve_device` (MPS -> CUDA -> CPU).
- Preserve checkpoint safety behavior (do not overwrite source checkpoint when resuming).
- For long-running training/sweep flows, prefer resumable, inspectable outputs (explicit output dirs, per-run checkpoints, summary CSV/markdown/log artifacts).

## Dependencies and tooling boundaries
- Requirements in `README.md` are the source of truth (Python 3.10+, PyTorch, NumPy, PyArrow).
- Do not introduce new dependencies, build systems, or codegen steps unless explicitly requested.
- Keep heavy generated artifacts (large checkpoints/logs) out of source diffs unless the task explicitly asks for them.

## Final output expectation
Always attempt to run at least one Python validation command before finishing (prefer `python -m py_compile src/*.py`, plus targeted smoke/tests when relevant). When done, report:
- Commands you ran (including focused tests and/or full tests)
- What changed (files/modules/scripts)
- How correctness was verified (tests, smoke checks, fixtures, or export validation)
- When you make a claim about behavior, include the evidence path: file names and the exact methods or sections you relied on.
