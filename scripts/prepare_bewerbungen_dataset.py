#!/usr/bin/env python3
"""
Convert a folder-per-sample bewerbungen dataset into a ShareGPT-format JSONL file.

## Input format

Each subdirectory in ``--bewerbungen-dir`` represents one CV sample and must contain:

  <Name>.md   – the CV text extracted from the PDF (markdown)
  <Name>.json – the structured parsed output (JSON object)

Files named ``*_metadata.json`` are ignored.  ``.webp`` images and other
auxiliary files are also ignored.  Folders missing either ``.md`` or ``.json``
produce a warning and are skipped.

## Output format

One JSON object per line (JSONL), ShareGPT-style::

  {"conversations": [
    {"role": "system",    "content": "<German extraction prompt listing fields in a specific order>"},
    {"role": "user",      "content": "<markdown CV text>"},
    {"role": "assistant", "content": "<pretty-printed JSON output with fields in the same order>"}
  ]}

The ``system`` message lists the expected JSON fields in a specific order.  The
assistant output JSON uses the **same field order** as the system prompt.  This
teaches the model to treat the system prompt as a schema template: at inference
time, if the caller lists fields in a different order, the model will honour
that order in its output.

## Field-order augmentation (--augment N)

By default (``--augment 0``) each CV folder produces one training sample with
the canonical field order defined by ``FIELD_DEFINITIONS``.

With ``--augment N`` (N ≥ 1) each folder produces **1 + N** training samples:
the canonical order plus N randomly shuffled permutations.  Each permutation
generates a different system prompt and a correspondingly reordered JSON output.
When combined with ``--split``, only the **train** split is augmented; the
eval split always uses the canonical order so evaluation is comparable.

Example: 415 CVs with ``--augment 4`` → 2 075 training records.

## Usage

  python scripts/prepare_bewerbungen_dataset.py \\
      --bewerbungen-dir ~/bewerbungen \\
      --output data/bewerbungen_train.jsonl

  # With augmentation and train/eval split
  python scripts/prepare_bewerbungen_dataset.py \\
      --bewerbungen-dir ~/bewerbungen \\
      --output data/bewerbungen_train.jsonl \\
      --augment 4 --split 0.1

  # Dry-run: count samples without writing
  python scripts/prepare_bewerbungen_dataset.py \\
      --bewerbungen-dir ~/bewerbungen \\
      --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Field schema
# ---------------------------------------------------------------------------

#: Ordered list of (field_name, description) pairs for the JSON output schema.
#: The order here defines the *canonical* field order used in the default
#: ``SYSTEM_PROMPT`` and assistant output.  Add or remove fields here to change
#: what the model extracts.
FIELD_DEFINITIONS: list[tuple[str, str]] = [
    ("name",          "Vollständiger Name des Bewerbers (string)"),
    ("gender",        'Geschlecht: "male" oder "female" (string)'),
    ("email",         "E-Mail-Adresse (string)"),
    ("address",       "Aktueller Wohnort / Adresse (string)"),
    ("date_of_birth", "Geburtsdatum im Format TT.MM.JJJJ (string)"),
    ("nationality",   "Nationalität (string)"),
    ("languages",     'Liste der Sprachen mit Niveau, z. B. ["Deutsch C1"] (array of strings)'),
    ("education",     "Liste der Ausbildungen / Abschlüsse (array of strings)"),
    ("skills",        "Liste der fachlichen Fähigkeiten und Kenntnisse (array of strings)"),
    ("products",      "Liste der verwendeten Software, Tools oder Produkte (array of strings)"),
]

#: Field names in canonical order (derived from ``FIELD_DEFINITIONS``).
FIELD_NAMES: list[str] = [name for name, _ in FIELD_DEFINITIONS]

#: Fields whose values should be JSON arrays.
_LIST_FIELDS: frozenset[str] = frozenset({"languages", "education", "skills", "products"})

#: Fields whose values should be strings (or null).
_STRING_FIELDS: frozenset[str] = frozenset({"name", "gender", "email", "address",
                                             "date_of_birth", "nationality"})


def make_system_prompt(field_order: list[str]) -> str:
    """
    Build the CV-extraction system prompt with fields listed in *field_order*.

    The prompt instructs the model to return a JSON object whose keys appear in
    the given order.  ``field_order`` must be a permutation of ``FIELD_NAMES``.
    """
    if sorted(field_order) != sorted(FIELD_NAMES) or len(field_order) != len(FIELD_NAMES):
        missing = sorted(set(FIELD_NAMES) - set(field_order))
        unknown = sorted(set(field_order) - set(FIELD_NAMES))
        duplicates = sorted({name for name in field_order if field_order.count(name) > 1})
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        if duplicates:
            details.append(f"duplicates={duplicates}")
        raise ValueError("field_order must be a permutation of FIELD_NAMES" + (f" ({', '.join(details)})" if details else ""))

    field_desc = {name: desc for name, desc in FIELD_DEFINITIONS}
    lines = "\n".join(f"  {name:<15}– {field_desc[name]}" for name in field_order)
    return (
        "Du bist ein Assistent zur Analyse von Bewerbungsunterlagen. "
        "Extrahiere die strukturierten Informationen aus dem folgenden Lebenslauf "
        f"und gib sie als JSON-Objekt mit diesen Feldern zurück:\n\n{lines}\n\n"
        "Antworte ausschließlich mit dem JSON-Objekt, ohne zusätzlichen Text."
    )


def make_assistant_json(json_data: dict, field_order: list[str]) -> str:
    """
    Return a pretty-printed JSON string with keys in *field_order*.

    Keys present in *json_data* but absent from *field_order* are appended at
    the end so no ground-truth data is silently dropped.
    """
    ordered = {k: json_data[k] for k in field_order if k in json_data}
    extras = {k: v for k, v in json_data.items() if k not in ordered}
    return json.dumps({**ordered, **extras}, ensure_ascii=False, indent=2)


#: Default system prompt using the canonical field order (backward-compatible).
SYSTEM_PROMPT: str = make_system_prompt(FIELD_NAMES)


# ---------------------------------------------------------------------------
# Ground-truth validation
# ---------------------------------------------------------------------------


def validate_json_record(data: dict, folder_name: str) -> None:
    """
    Warn to stderr if *data* is missing expected fields or has incorrect types.

    Does not raise; callers decide whether to skip or accept the sample.
    """
    for field in FIELD_NAMES:
        if field not in data:
            print(
                f"  Warning: {folder_name!r} missing field {field!r}",
                file=sys.stderr,
            )
        elif field in _LIST_FIELDS and not isinstance(data[field], list):
            print(
                f"  Warning: {folder_name!r} field {field!r} should be a list"
                f", got {type(data[field]).__name__!r}",
                file=sys.stderr,
            )
        elif field in _STRING_FIELDS and not isinstance(data[field], (str, type(None))):
            print(
                f"  Warning: {folder_name!r} field {field!r} should be a string"
                f", got {type(data[field]).__name__!r}",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

def find_pairs(bewerbungen_dir: Path) -> list[tuple[Path, Path]]:
    """
    Return sorted (md_path, json_path) pairs found in subdirectories of
    *bewerbungen_dir*.  Each subdirectory is expected to contain exactly one
    ``.md`` and one ``.json`` (non-metadata) file.
    """
    pairs: list[tuple[Path, Path]] = []

    for subdir in sorted(bewerbungen_dir.iterdir()):
        if not subdir.is_dir():
            continue

        mds = sorted(subdir.glob("*.md"))
        jsons = sorted(p for p in subdir.glob("*.json") if "_metadata" not in p.name)

        if not mds:
            print(f"  Warning: no .md file in {subdir.name!r} — skipping", file=sys.stderr)
            continue
        if not jsons:
            print(f"  Warning: no .json file in {subdir.name!r} — skipping", file=sys.stderr)
            continue

        if len(mds) > 1:
            print(
                f"  Warning: multiple .md files in {subdir.name!r}, using {mds[0].name!r}",
                file=sys.stderr,
            )
        if len(jsons) > 1:
            print(
                f"  Warning: multiple .json files in {subdir.name!r}, using {jsons[0].name!r}",
                file=sys.stderr,
            )

        pairs.append((mds[0], jsons[0]))

    return pairs


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(
    bewerbungen_dir: Path,
    output_path: Path | None,
    dry_run: bool = False,
    split: float = 0.0,
    seed: int = 42,
    quiet: bool = False,
    augment: int = 0,
) -> None:
    pairs = find_pairs(bewerbungen_dir)
    if not pairs:
        print(f"No complete pairs found in {bewerbungen_dir}", file=sys.stderr)
        sys.exit(1)

    if not dry_run:
        if output_path is None:
            raise ValueError("output_path is required when dry_run=False")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Split by CV pair (before augmentation) so no CV appears in both splits.
    if split > 0.0:
        if len(pairs) < 2:
            raise ValueError("--split requires at least two complete CV pairs to keep train and eval non-empty")
        shuffled_pairs = list(pairs)
        rng.shuffle(shuffled_pairs)
        n_eval_pairs = max(1, round(len(shuffled_pairs) * split))
        if n_eval_pairs >= len(shuffled_pairs):
            raise ValueError("--split would leave no training CVs; lower --split or add more pairs")
        eval_pairs = shuffled_pairs[:n_eval_pairs]
        train_pairs = shuffled_pairs[n_eval_pairs:]
    else:
        train_pairs = list(pairs)
        eval_pairs = []

    def _build_records(
        pair_list: list[tuple[Path, Path]],
        *,
        apply_augment: bool,
    ) -> list[dict]:
        records: list[dict] = []
        for md_path, json_path in pair_list:
            try:
                md_content = md_path.read_text(encoding="utf-8").strip()
                json_data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(
                    f"  Warning: could not read {md_path.parent.name!r} — {exc}",
                    file=sys.stderr,
                )
                continue

            validate_json_record(json_data, md_path.parent.name)

            if not quiet:
                print(f"  {md_path.parent.name}")

            # Canonical-order record (always included)
            records.append({
                "conversations": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": md_content},
                    {"role": "assistant", "content": make_assistant_json(json_data, FIELD_NAMES)},
                ]
            })

            # Augmented records with randomly shuffled field order
            if apply_augment and augment > 0:
                for _ in range(augment):
                    shuffled = FIELD_NAMES[:]
                    rng.shuffle(shuffled)
                    records.append({
                        "conversations": [
                            {"role": "system",    "content": make_system_prompt(shuffled)},
                            {"role": "user",      "content": md_content},
                            {"role": "assistant", "content": make_assistant_json(json_data, shuffled)},
                        ]
                    })
        return records

    if split > 0.0:
        if not quiet:
            print("--- train pairs ---")
        train_records = _build_records(train_pairs, apply_augment=True)
        if not quiet:
            print("--- eval pairs ---")
        eval_records = _build_records(eval_pairs, apply_augment=False)
        rng.shuffle(train_records)
        if not train_records:
            raise ValueError("--split produced no training samples; check source files or lower --split")
        if not eval_records:
            raise ValueError("--split produced no eval samples; check source files or increase --split")

        n_train = len(train_records)
        n_eval = len(eval_records)
        print(f"\nTotal pairs: {len(pairs)} ({len(train_pairs)} train, {len(eval_pairs)} eval)")
        if augment > 0:
            print(f"Augmentation: {augment} extra permutations per train CV → {n_train} train records")
        else:
            print(f"Total: {n_train} train, {n_eval} eval samples")

        if dry_run:
            print("Dry run — no file written.")
            return

        eval_path = output_path.parent / (output_path.stem + "_eval" + output_path.suffix)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in train_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(eval_path, "w", encoding="utf-8") as f:
            for r in eval_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"  Train ({n_train}): {output_path}")
        print(f"  Eval  ({n_eval}): {eval_path}")
    else:
        records = _build_records(train_pairs, apply_augment=True)
        n_total = len(records)
        if not records:
            raise ValueError("No training samples produced; check source files before writing output")
        print(f"\nTotal: {n_total} training samples from {len(pairs)} folders")
        if augment > 0:
            print(f"(1 canonical + {augment} augmented permutations per CV)")

        if dry_run:
            print("Dry run — no file written.")
            return

        with open(output_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert folder-per-sample bewerbungen data into a ShareGPT JSONL dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bewerbungen-dir",
        required=True,
        type=Path,
        help="Directory whose subdirectories each contain one .md and one .json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file path (e.g. data/bewerbungen_train.jsonl). Required unless --dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count samples without writing output.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.0,
        metavar="RATIO",
        help=(
            "Fraction of CV folders to hold out as an eval set (e.g. 0.1 for 10%%). "
            "Writes an additional '<output-stem>_eval<suffix>' file alongside the main output. "
            "Requires --output. Default: 0.0 (no split)."
        ),
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of additional field-order permutations to generate per CV for the train "
            "split.  Each permutation shuffles the JSON field order in both the system prompt "
            "and the assistant output, teaching the model to follow the prompt's schema.  "
            "Default: 0 (canonical order only).  Example: --augment 4 produces 1+4=5 records "
            "per CV (1 canonical + 4 random permutations)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling before the train/eval split (default: 42).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-folder name output; print only totals.",
    )
    args = parser.parse_args()

    if args.split and (args.split <= 0.0 or args.split >= 1.0):
        parser.error("--split must be in the range (0.0, 1.0).")
    if args.augment < 0:
        parser.error("--augment must be greater than or equal to 0.")

    bewerbungen_dir = args.bewerbungen_dir.expanduser()
    output_path = args.output.expanduser() if args.output else None

    if not bewerbungen_dir.is_dir():
        print(
            f"Error: --bewerbungen-dir '{bewerbungen_dir}' is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.dry_run and output_path is None:
        parser.error("--output is required unless --dry-run is set.")

    print(f"Processing bewerbungen in: {bewerbungen_dir}")
    convert(
        bewerbungen_dir,
        output_path,
        dry_run=args.dry_run,
        split=args.split,
        seed=args.seed,
        quiet=args.quiet,
        augment=args.augment,
    )


if __name__ == "__main__":
    main()
