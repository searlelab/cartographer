from __future__ import annotations
import itertools
from pathlib import Path
import pyarrow.dataset as ds

DATASETS = [
    ("IM2Deep_CCS", Path("/Users/searle.brian/Documents/huggingface/data/IM2Deep_CCS")),
    ("prospect-ptms-charge", Path("/Users/searle.brian/Documents/huggingface/data/prospect-ptms-charge")),
    ("prospect-ptms-irt", Path("/Users/searle.brian/Documents/huggingface/data/prospect-ptms-irt")),
    ("prospect-ptms-ms2", Path("/Users/searle.brian/Documents/huggingface/data/prospect-ptms-ms2")),
]
SPLITS = ("train", "test", "val")
SEQUENCE_COLUMNS = (
    "modified_peptide_sequence",
    "ModifiedPeptideSequence",
    "modified_sequence",
    "ModifiedSequence",
    "peptide_mod_seq",
    "PeptideModSeq",
    "peptide",
    "Peptide",
    "sequence",
    "Sequence",
)

def discover_split_files(dataset_dir: Path, split: str) -> list[Path]:
    patterns = [f"{split}-*.parquet", f"{split}.parquet"]
    if split == "val":
        patterns.extend(["validation-*.parquet", "validation.parquet"])
    files = []
    for pattern in patterns:
        files.extend(sorted(dataset_dir.glob(pattern)))
    return files

def resolve_sequence_column(files: list[Path]) -> str:
    schema = ds.dataset([str(path) for path in files], format="parquet").schema
    for candidate in SEQUENCE_COLUMNS:
        if candidate in schema.names:
            return candidate
    raise ValueError(f"Could not find sequence column in schema: {sorted(schema.names)}")

def read_sequences(files: list[Path], column: str) -> set[str]:
    dataset = ds.dataset([str(path) for path in files], format="parquet")
    values = set()
    for batch in dataset.scanner(columns=[column]).to_batches():
        for value in batch.column(0).to_pylist():
            if value is not None:
                values.add(str(value))
    return values

def compute_rows(named_sets: dict[str, set[str]]) -> list[tuple[str, int]]:
    rows = []
    names = tuple(named_sets)
    for size in range(1, len(names) + 1):
        for combo in itertools.combinations(names, size):
            shared = set.intersection(*(named_sets[name] for name in combo))
            rows.append((" ∩ ".join(combo), len(shared)))
    rows.sort(key=lambda item: (item[0].count("∩"), item[0]))
    return rows

lines = [
    "# Peptide Overlap Report",
    "",
    "Counts are intersections of unique modified peptide sequences within each split.",
    "",
]

for split in SPLITS:
    lines.extend([f"## {split}", "", "| Intersection | Count |", "|---|---:|"])
    split_sets = {}
    for dataset_name, dataset_dir in DATASETS:
        files = discover_split_files(dataset_dir, split)
        if not files:
            raise ValueError(f"No parquet files found for dataset={dataset_name} split={split}")
        split_sets[dataset_name] = read_sequences(files, resolve_sequence_column(files))
    for label, count in compute_rows(split_sets):
        lines.append(f"| {label} | {count} |")
    lines.append("")

Path("/Users/searle.brian/Documents/projects/cartographer/peptide_overlap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
