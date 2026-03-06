#!/usr/bin/env python3
"""Compare two CCS datasets with different peptidoform encodings."""

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


CHARGE_PATTERN = re.compile(r"^(.+)/(\d+)$")
REFERENCE_TO_MULTI_SLOPE = 1.0583
REFERENCE_TO_MULTI_INTERCEPT = -10.101

# PTM name mapping to requested UNIMOD accessions.
PTM_TO_UNIMOD = {
    "Acetyl": 1,
    "Biotin": 3,
    "Carbamidomethyl": 4,
    "Citrullination": 7,
    "Deamidated": 7,
    "Deamidation": 7,
    "Methyl": 34,
    "Oxidation": 35,
    "Dimethyl": 36,
    "Trimethyl": 37,
    "HexNAc": 43,
    "Propionyl": 58,
    "Succinyl": 64,
    "GlyGly": 121,
    "Formyl": 122,
    "Cysteinyl": 312,
    "Hydroxyproline": 408,
    "Malonyl": 747,
    "Butyryl": 1289,
    "Crotonyl": 1363,
    "Gluratylation": 1848,
    "Glutarylation": 1848,
    "hydroxyisobutyryl": 1849,
    "Hydroxyisobutyryl": 1849,
    "Phospho": 21,
    "Nitro": 354,
}


def canonicalize(sequence, charge, mods):
    """Return a stable key for a peptidoform + charge entry."""
    return (sequence, int(charge), tuple(sorted(mods, key=lambda x: (x[0], x[1]))))


def parse_reference_peptidoform(raw):
    """Parse reference_ccs.csv style: [Acetyl]-PEPTIDEC[Carbamidomethyl]K/3."""
    match = CHARGE_PATTERN.match(raw.strip())
    if not match:
        raise ValueError(f"Could not parse charge from peptidoform: {raw!r}")

    encoded = match.group(1)
    charge = int(match.group(2))
    mods = []

    body = encoded
    if body.startswith("["):
        dash_idx = body.find("-")
        if dash_idx > 0:
            nterm_section = body[:dash_idx]
            tokens = re.findall(r"\[([^\]]+)\]", nterm_section)
            rebuilt = "".join(f"[{token}]" for token in tokens)
            if tokens and rebuilt == nterm_section:
                for token in tokens:
                    mods.append((0, token))
                body = body[dash_idx + 1 :]

    sequence_chars = []
    residue_index = 0
    i = 0
    while i < len(body):
        ch = body[i]
        if "A" <= ch <= "Z":
            residue_index += 1
            sequence_chars.append(ch)
            i += 1

            while i < len(body) and body[i] == "[":
                close_idx = body.find("]", i + 1)
                if close_idx == -1:
                    raise ValueError(f"Unterminated modification in {raw!r}")
                mod_name = body[i + 1 : close_idx].strip()
                mods.append((residue_index, mod_name))
                i = close_idx + 1
            continue

        raise ValueError(f"Unexpected character {ch!r} in {raw!r}")

    sequence = "".join(sequence_chars)
    return canonicalize(sequence, charge, mods)


def parse_multi_reference_row(sequence, mod_string, charge):
    """Parse multi_reference_ccs.csv style sequence/mods/charge row."""
    sequence = sequence.strip()
    mods = []

    mod_string = mod_string.strip()
    if mod_string:
        tokens = [token.strip() for token in mod_string.split("|")]
        if len(tokens) % 2 != 0:
            raise ValueError(f"Invalid modification token count: {mod_string!r}")

        for idx in range(0, len(tokens), 2):
            site = int(tokens[idx])
            mod_name = tokens[idx + 1]
            mods.append((site, mod_name))

    return canonicalize(sequence, int(charge), mods)


def load_reference(path):
    unique = set()
    entry_to_ccs = defaultdict(list)
    total_rows = 0

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "peptidoform" not in reader.fieldnames:
            raise ValueError(f"{path} missing expected 'peptidoform' column")

        for row in reader:
            total_rows += 1
            entry = parse_reference_peptidoform(row["peptidoform"])
            unique.add(entry)
            entry_to_ccs[entry].append(row["CCS"].strip())

    return unique, total_rows, entry_to_ccs


def load_multi_reference(path):
    unique = set()
    entry_to_ccs = defaultdict(list)
    total_rows = 0

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"seq", "modifications", "charge", "ccs_observed"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} missing one of required columns: {sorted(required)}")

        for row in reader:
            total_rows += 1
            entry = parse_multi_reference_row(
                row["seq"], row["modifications"], row["charge"]
            )
            unique.add(entry)
            entry_to_ccs[entry].append(row["ccs_observed"].strip())

    return unique, total_rows, entry_to_ccs


def ptm_type_counts(entries):
    counts = Counter()
    for _, _, mods in entries:
        for mod_name in {mod_name for _, mod_name in mods}:
            counts[mod_name] += 1
    return counts


def charge_distribution(entries):
    return Counter(charge for _, charge, _ in entries)


def print_counter(counter, key_sort=None):
    if not counter:
        print("  (none)")
        return

    if key_sort is None:
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    else:
        ordered = sorted(counter.items(), key=key_sort)

    for key, value in ordered:
        print(f"  {key}: {value}")


def collapse_ccs_values(values):
    if not values:
        return ""
    unique_values = sorted(set(values), key=lambda value: float(value))
    return ";".join(unique_values)


def ccs_to_text(value):
    return f"{value:.15f}".rstrip("0").rstrip(".")


def format_modifications(mods):
    if not mods:
        return ""
    flat = []
    for site, mod_name in mods:
        flat.extend([str(site), mod_name])
    return "|".join(flat)


def to_cartographer_unimod_sequence(sequence, mods):
    nterm_mods = [mod_name for site, mod_name in mods if site == 0]
    residue_mods = defaultdict(list)
    for site, mod_name in mods:
        if site == 0:
            continue
        if site < 1 or site > len(sequence):
            raise ValueError(
                f"Invalid modification site {site} for sequence {sequence!r}"
            )
        residue_mods[site].append(mod_name)

    if len(nterm_mods) > 1:
        raise ValueError(
            f"Multiple N-term mods are not supported in Cartographer export: {mods!r}"
        )

    if nterm_mods:
        mod_name = nterm_mods[0]
        if mod_name not in PTM_TO_UNIMOD:
            raise ValueError(f"Unknown PTM for UNIMOD mapping: {mod_name!r}")
        nterm = f"[UNIMOD:{PTM_TO_UNIMOD[mod_name]}]"
    else:
        nterm = "[]"

    body_parts = []
    for site, aa in enumerate(sequence, start=1):
        body_parts.append(aa)
        for mod_name in residue_mods.get(site, []):
            if mod_name not in PTM_TO_UNIMOD:
                raise ValueError(f"Unknown PTM for UNIMOD mapping: {mod_name!r}")
            body_parts.append(f"[UNIMOD:{PTM_TO_UNIMOD[mod_name]}]")

    body = "".join(body_parts)
    return f"{nterm}-{body}-[]"


def corrected_reference_ccs(value_text):
    return REFERENCE_TO_MULTI_SLOPE * float(value_text) + REFERENCE_TO_MULTI_INTERCEPT


def averaged_union_ccs(reference_values, multi_values):
    converted_reference = [corrected_reference_ccs(value) for value in reference_values]
    all_values = converted_reference + [float(value) for value in multi_values]
    return sum(all_values) / len(all_values)


def write_union_table(reference_ccs, multi_ccs, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    union_keys = sorted(
        set(reference_ccs) | set(multi_ccs), key=lambda key: (key[0], key[1], key[2])
    )

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["modified_sequence", "charge", "ccs"])

        for sequence, charge, mods in union_keys:
            ccs = averaged_union_ccs(
                reference_ccs.get((sequence, charge, mods), []),
                multi_ccs.get((sequence, charge, mods), []),
            )
            writer.writerow(
                [
                    to_cartographer_unimod_sequence(sequence, mods),
                    charge,
                    ccs_to_text(ccs),
                ]
            )

    return len(union_keys)


def write_intersection_table(reference_ccs, multi_ccs, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intersection_keys = sorted(
        set(reference_ccs) & set(multi_ccs), key=lambda key: (key[0], key[1], key[2])
    )

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sequence",
                "modifications",
                "charge",
                "reference_ccs_values",
                "multi_reference_ccs_values",
            ]
        )

        for sequence, charge, mods in intersection_keys:
            writer.writerow(
                [
                    sequence,
                    format_modifications(mods),
                    charge,
                    collapse_ccs_values(reference_ccs[(sequence, charge, mods)]),
                    collapse_ccs_values(multi_ccs[(sequence, charge, mods)]),
                ]
            )

    return len(intersection_keys)


def duplicate_key_count(entry_to_ccs):
    return sum(1 for values in entry_to_ccs.values() if len(values) > 1)


def main():
    parser = argparse.ArgumentParser(
        description="Compare overlap between reference_ccs.csv and multi_reference_ccs.csv"
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--multi-reference", required=True, type=Path)
    parser.add_argument("--union-output", type=Path)
    parser.add_argument("--intersection-output", type=Path)
    args = parser.parse_args()

    reference_entries, reference_rows, reference_ccs = load_reference(args.reference)
    multi_entries, multi_rows, multi_ccs = load_multi_reference(args.multi_reference)

    overlap = reference_entries & multi_entries
    reference_only = reference_entries - multi_entries
    multi_only = multi_entries - reference_entries
    union_entries = reference_entries | multi_entries

    print("Input row counts")
    print(f"  reference rows: {reference_rows}")
    print(f"  multi_reference rows: {multi_rows}")
    print()

    print("Unique peptidoform+charge counts")
    print(f"  reference unique: {len(reference_entries)}")
    print(f"  multi_reference unique: {len(multi_entries)}")
    print()

    print("Duplicate peptidoform+charge keys (should be 0 if unique)")
    print(f"  reference duplicates: {duplicate_key_count(reference_ccs)}")
    print(f"  multi_reference duplicates: {duplicate_key_count(multi_ccs)}")
    print()

    print("Venn counts (peptidoform+charge)")
    print(f"  reference only: {len(reference_only)}")
    print(f"  overlap: {len(overlap)}")
    print(f"  multi_reference only: {len(multi_only)}")
    print(f"  union: {len(union_entries)}")
    print()

    print("PTM type observations (reference unique entries)")
    print_counter(ptm_type_counts(reference_entries))
    print()

    print("PTM type observations (multi_reference unique entries)")
    print_counter(ptm_type_counts(multi_entries))
    print()

    print("PTM type observations (union unique entries)")
    print_counter(ptm_type_counts(union_entries))
    print()

    print("Charge distribution (reference unique entries)")
    print_counter(charge_distribution(reference_entries), key_sort=lambda kv: kv[0])
    print()

    print("Charge distribution (multi_reference unique entries)")
    print_counter(charge_distribution(multi_entries), key_sort=lambda kv: kv[0])
    print()

    print("Charge distribution (union unique entries)")
    print_counter(charge_distribution(union_entries), key_sort=lambda kv: kv[0])

    if args.union_output:
        written_rows = write_union_table(reference_ccs, multi_ccs, args.union_output)
        print()
        print(f"Wrote union peptide/charge table: {args.union_output}")
        print(f"  rows: {written_rows}")

    if args.intersection_output:
        written_rows = write_intersection_table(
            reference_ccs, multi_ccs, args.intersection_output
        )
        print()
        print(f"Wrote intersection peptide/charge table: {args.intersection_output}")
        print(f"  rows: {written_rows}")


if __name__ == "__main__":
    main()
