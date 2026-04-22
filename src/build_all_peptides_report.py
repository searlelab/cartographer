from __future__ import annotations

import json
from pathlib import Path

import pyarrow.dataset as ds


DATASETS = [
    ("IM2Deep_CCS", Path("/Users/searle.brian/Documents/huggingface/data/IM2Deep_CCS")),
    ("prospect-ptms-charge", Path("/Users/searle.brian/Documents/huggingface/data/prospect-ptms-charge")),
    ("prospect-ptms-irt", Path("/Users/searle.brian/Documents/huggingface/data/prospect-ptms-irt")),
    ("prospect-ptms-ms2", Path("/Users/searle.brian/Documents/huggingface/data/prospect-ptms-ms2")),
]

SPLITS = ("train", "test", "val")
PLAIN_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")
OUTPUT_PATH = Path("/Users/searle.brian/Documents/projects/cartographer/all_peptides.md")


def discover_split_files(dataset_dir: Path, split: str) -> list[Path]:
    bases = [dataset_dir, dataset_dir / "data", dataset_dir / "holdout"]
    patterns = [f"{split}-*.parquet", f"{split}.parquet"]
    if split == "val":
        patterns.extend(["validation-*.parquet", "validation.parquet"])

    files: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        if not base.exists():
            continue
        for pattern in patterns:
            for path in sorted(base.glob(pattern)):
                if path not in seen:
                    files.append(path)
                    seen.add(path)
    return files


def load_im2deep_decoder() -> tuple[dict[int, str], dict[str, str], dict[str, tuple[str, str]]]:
    metadata_path = DATASETS[0][1] / "sculptor_dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    aa_to_int = metadata["tokenizer"]["aa_to_int"]
    int_to_token = {value: key for key, value in aa_to_int.items()}
    nterm_token_to_unimod = {
        token: unimod for unimod, token in metadata["tokenizer"]["nterm_unimod_map"].items()
    }
    residue_token_to_mod = {
        entry["token"]: (entry["residue"], entry["unimod"])
        for entry in metadata["tokenizer"]["residue_unimod_map"]
    }
    return int_to_token, nterm_token_to_unimod, residue_token_to_mod


def decode_im2deep_sequence(
    seq_tokens: list[int],
    int_to_token: dict[int, str],
    nterm_token_to_unimod: dict[str, str],
    residue_token_to_mod: dict[str, tuple[str, str]],
) -> str | None:
    coded_tokens: list[str] = []
    for raw_value in seq_tokens:
        token_value = int(raw_value)
        if token_value == 0:
            break
        token = int_to_token[token_value]
        if token == "_":
            break
        coded_tokens.append(token)

    if not coded_tokens:
        return None

    nterm_token = coded_tokens[0]
    body_tokens = coded_tokens[1:]
    nterm_text = "[]" if nterm_token == "-" else f"[{nterm_token_to_unimod[nterm_token]}]"

    body_parts: list[str] = []
    for token in body_tokens:
        if token in PLAIN_RESIDUES:
            body_parts.append(token)
            continue
        residue, unimod = residue_token_to_mod[token]
        body_parts.append(f"{residue}[{unimod}]")

    return f"{nterm_text}-{''.join(body_parts)}-[]"


def read_modified_sequences(files: list[Path]) -> set[str]:
    dataset = ds.dataset([str(path) for path in files], format="parquet")
    sequences: set[str] = set()
    for batch in dataset.scanner(columns=["modified_sequence"]).to_batches():
        for value in batch.column(0).to_pylist():
            if value is not None:
                sequences.add(str(value))
    return sequences


def read_im2deep_sequences(files: list[Path]) -> set[str]:
    int_to_token, nterm_token_to_unimod, residue_token_to_mod = load_im2deep_decoder()
    dataset = ds.dataset([str(path) for path in files], format="parquet")
    sequences: set[str] = set()
    for batch in dataset.scanner(columns=["seq_tokens"]).to_batches():
        for value in batch.column(0).to_pylist():
            decoded = decode_im2deep_sequence(
                value, int_to_token, nterm_token_to_unimod, residue_token_to_mod
            )
            if decoded is not None:
                sequences.add(decoded)
    return sequences


def read_dataset_sequences(dataset_name: str, dataset_dir: Path) -> set[str]:
    files: list[Path] = []
    for split in SPLITS:
        files.extend(discover_split_files(dataset_dir, split))

    if dataset_name == "IM2Deep_CCS":
        return read_im2deep_sequences(files)
    return read_modified_sequences(files)


def main() -> None:
    all_sequences: set[str] = set()
    for dataset_name, dataset_dir in DATASETS:
        all_sequences.update(read_dataset_sequences(dataset_name, dataset_dir))

    lines = ["# All Peptides", ""]
    lines.extend(f"- {sequence}" for sequence in sorted(all_sequences))
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
