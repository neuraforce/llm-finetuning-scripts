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

Each augmented sample also uses a randomly selected **date format** (chosen
from the non-canonical entries in ``_DATE_FORMATS``) so the model learns to
produce dates in any of the supported notations.  The system prompt explicitly
names the required format — including the language for long-word formats
(German month names vs. English) — and the assistant output uses that same
format.  The canonical sample preserves the source format (DD.MM.YYYY).

Supported date formats:

* ``DD.MM.YYYY``     – e.g. 15.03.1990  (canonical — matches source data)
* ``YYYY-MM-DD``     – e.g. 1990-03-15  (ISO 8601)
* ``DD/MM/YYYY``     – e.g. 15/03/1990
* ``MM/DD/YYYY``     – e.g. 03/15/1990  (US style)
* ``D. Monat JJJJ``  – e.g. 15. März 1990  (German long form)
* ``Month DD, YYYY`` – e.g. March 15, 1990  (English long form)

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Field schema
# ---------------------------------------------------------------------------

#: Ordered list of (field_name, description, is_list) triples for the JSON output schema.
#: ``is_list=True`` → the field value is a JSON array; ``False`` → string or null.
#: The order defines the *canonical* field order for the default ``SYSTEM_PROMPT``.
#: Add or remove entries here — all derived constants update automatically.
_FieldDef = tuple[str, str, bool]  # (name, description, is_list)

FIELD_DEFINITIONS: list[_FieldDef] = [
    ("name",          "Vollständiger Name des Bewerbers (string)",                                                False),
    ("gender",        'Geschlecht: "male" oder "female" (string)',                                                False),
    ("email",         "E-Mail-Adresse (string)",                                                                  False),
    ("address",       "Aktueller Wohnort / Adresse (string)",                                                     False),
    ("date_of_birth", "Geburtsdatum im Format TT.MM.JJJJ (string)",                                              False),
    ("nationality",   "Nationalität (string)",                                                                    False),
    ("languages",     'Liste der Sprachen mit Niveau, z. B. ["Deutsch C1"] (array of strings)',                  True),
    ("education",     "Liste der Ausbildungen / Abschlüsse (array of strings)",                                   True),
    ("skills",        "Liste der fachlichen Fähigkeiten und Kenntnisse (array of strings)",                       True),
    ("products",      "Liste der verwendeten Software, Tools oder Produkte (array of strings)",                   True),
]

#: Field names in canonical order (derived).
FIELD_NAMES: list[str] = [name for name, _, _ in FIELD_DEFINITIONS]

#: Fields that must be JSON arrays (derived from ``FIELD_DEFINITIONS``).
_LIST_FIELDS: frozenset[str] = frozenset(name for name, _, is_list in FIELD_DEFINITIONS if is_list)

#: Fields that must be strings or null (derived from ``FIELD_DEFINITIONS``).
_STRING_FIELDS: frozenset[str] = frozenset(name for name, _, is_list in FIELD_DEFINITIONS if not is_list)

#: Fields whose values are dates stored as "DD.MM.YYYY" in the source data.
_DATE_FIELDS: frozenset[str] = frozenset({"date_of_birth"})

# ---------------------------------------------------------------------------
# Date-format augmentation
# ---------------------------------------------------------------------------

_GERMAN_MONTHS = [
    "", "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]


def _fmt_german_long(dt: datetime) -> str:
    """Format as 'D. Monat JJJJ' with German month name (e.g. '15. März 1990')."""
    return f"{dt.day}. {_GERMAN_MONTHS[dt.month]} {dt.year}"


@dataclass(frozen=True)
class DateFormat:
    """One supported date output format."""
    field_desc_label: str        # Injected into the per-field schema line
    date_rule_line: str          # Full "Datumsangaben …" rule bullet in the system prompt
    fmt_fn: Callable[[datetime], str]  # Converts datetime → target string


#: Available date formats, index 0 = canonical (matches source data: DD.MM.YYYY).
#: The system prompt explicitly names the language for long-word formats so the LLM
#: is never left guessing whether German or English month names are expected.
_DATE_FORMATS: list[DateFormat] = [
    # index 0 — canonical: matches DD.MM.YYYY format already in source .json files
    DateFormat(
        "TT.MM.JJJJ (z. B. 15.03.1990)",
        "- Datumsangaben immer im Format TT.MM.JJJJ (z. B. 15.03.1990).",
        lambda d: d.strftime("%d.%m.%Y"),
    ),
    DateFormat(
        "JJJJ-MM-TT (z. B. 1990-03-15)",
        "- Datumsangaben immer im ISO-Format JJJJ-MM-TT (z. B. 1990-03-15).",
        lambda d: d.strftime("%Y-%m-%d"),
    ),
    DateFormat(
        "TT/MM/JJJJ (z. B. 15/03/1990)",
        "- Datumsangaben immer im Format TT/MM/JJJJ (z. B. 15/03/1990).",
        lambda d: d.strftime("%d/%m/%Y"),
    ),
    DateFormat(
        "MM/TT/JJJJ (z. B. 03/15/1990)",
        "- Datumsangaben immer im US-Format MM/TT/JJJJ (z. B. 03/15/1990).",
        lambda d: d.strftime("%m/%d/%Y"),
    ),
    DateFormat(
        "TT. Monat JJJJ – ausgeschriebener Monatsname auf Deutsch (z. B. 15. März 1990)",
        "- Datumsangaben immer im Format TT. Monat JJJJ mit ausgeschriebenem deutschen"
        " Monatsnamen (z. B. 15. März 1990).",
        _fmt_german_long,
    ),
    DateFormat(
        "Month DD, YYYY – English month name written out in full (e.g. March 15, 1990)",
        "- Always format dates as Month DD, YYYY with the English month name written out"
        " in full (e.g. March 15, 1990).",
        lambda d: d.strftime("%B %d, %Y"),
    ),
]

#: Index into ``_DATE_FORMATS`` for the canonical format (DD.MM.YYYY = source data format).
_CANONICAL_DATE_FMT_IDX: int = 0

#: Pre-computed list of non-canonical format indices (for augmentation).
_NON_CANONICAL_DATE_IDXS: list[int] = [
    i for i in range(len(_DATE_FORMATS)) if i != _CANONICAL_DATE_FMT_IDX
]

#: Canonical field_desc_label per date field, derived from _DATE_FORMATS[0].
#: Used to substitute the correct label in augmented system prompts.
_DATE_FIELD_CANONICAL_LABEL: dict[str, str] = {
    fname: _DATE_FORMATS[_CANONICAL_DATE_FMT_IDX].field_desc_label
    for fname in _DATE_FIELDS
}


def _parse_date(value: str) -> datetime | None:
    """Parse a DD.MM.YYYY date string. Returns None if unparseable."""
    try:
        return datetime.strptime(value.strip(), "%d.%m.%Y")
    except (ValueError, AttributeError):
        return None


def _reformat_date(value: str | None, fmt: DateFormat) -> str | None:
    """Reformat *value* (DD.MM.YYYY) using *fmt*.  Returns the original on parse failure."""
    if value is None:
        return None
    dt = _parse_date(value)
    return fmt.fmt_fn(dt) if dt is not None else value


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def make_system_prompt(field_order: list[str], date_fmt_idx: int = _CANONICAL_DATE_FMT_IDX) -> str:
    """
    Build the CV-extraction system prompt with fields in *field_order*.

    *date_fmt_idx* selects a ``DateFormat`` from ``_DATE_FORMATS``.  The chosen
    format's ``field_desc_label`` is substituted into date-field description lines
    and its ``date_rule_line`` replaces the generic date rule bullet.  Long-word
    formats explicitly name the required language so the LLM is never ambiguous.

    Raises ``ValueError`` for unknown field names.
    """
    known = {name for name, _, _ in FIELD_DEFINITIONS}
    unknown = [f for f in field_order if f not in known]
    if unknown:
        raise ValueError(f"make_system_prompt: unknown fields: {unknown!r}")

    date_fmt = _DATE_FORMATS[date_fmt_idx]
    field_desc = {name: desc for name, desc, _ in FIELD_DEFINITIONS}

    for fname, canonical_label in _DATE_FIELD_CANONICAL_LABEL.items():
        if fname in field_desc:
            field_desc[fname] = field_desc[fname].replace(canonical_label, date_fmt.field_desc_label)

    schema_lines = "\n".join(f"  {name:<15}– {field_desc[name]}" for name in field_order)
    rules = "\n".join([
        "- Antworte AUSSCHLIESSLICH mit dem rohen JSON-Objekt (beginnend mit { und endend mit }).",
        "- Kein Markdown, keine Code-Blöcke (```), keine Erklärungen, kein sonstiger Text.",
        "- Fehlende oder unbekannte Strings: verwende null (nicht leerer String \"\").",
        "- Fehlende oder leere Listen: verwende [] (nicht null).",
        date_fmt.date_rule_line,
        "- Unterschied skills vs. products:",
        "    skills   = fachliche Fähigkeiten und Kenntnisse"
        " (z. B. \"Python\", \"Machine Learning\", \"Projektmanagement\")",
        "    products = konkrete Software, Tools, Plattformen oder Produkte"
        " (z. B. \"VS Code\", \"SAP S/4HANA\", \"AWS EC2\", \"Jira\")",
        "  Ein Skill ist etwas, das man kann; ein Produkt ist etwas, das man benutzt.",
    ])
    return (
        "Du bist ein Assistent zur Analyse von Bewerbungsunterlagen.\n\n"
        "Extrahiere die strukturierten Informationen aus dem folgenden Lebenslauf "
        "und gib sie als JSON-Objekt zurück. Das Objekt muss genau diese Felder in "
        f"genau dieser Reihenfolge enthalten:\n\n{schema_lines}\n\n"
        f"Wichtige Regeln:\n{rules}\n"
    )


def make_assistant_json(
    json_data: dict,
    field_order: list[str],
    date_fmt_idx: int = _CANONICAL_DATE_FMT_IDX,
) -> str:
    """
    Return a pretty-printed JSON string with keys in *field_order*.

    Date fields are reformatted to match *date_fmt_idx*.  Keys present in
    *json_data* but absent from *field_order* are appended at the end so no
    ground-truth data is silently dropped.
    """
    data = dict(json_data)
    fmt = _DATE_FORMATS[date_fmt_idx]
    for fname in _DATE_FIELDS:
        if fname in data:
            data[fname] = _reformat_date(data[fname], fmt)
    ordered = {k: data[k] for k in field_order if k in data}
    extras  = {k: v for k, v in data.items() if k not in ordered}
    return json.dumps({**ordered, **extras}, ensure_ascii=False, indent=2)


#: Default system prompt using the canonical field order (backward-compatible).
SYSTEM_PROMPT: str = make_system_prompt(FIELD_NAMES)

# ---------------------------------------------------------------------------
# Ground-truth validation
# ---------------------------------------------------------------------------


def validate_json_record(data: dict, folder_name: str) -> bool:
    """
    Warn to stderr for missing fields, wrong types, or unexpected keys.
    Returns True only when the record matches the expected schema exactly.
    """
    valid = True
    for name, _, is_list in FIELD_DEFINITIONS:
        if name not in data:
            valid = False
            print(f"  Warning: {folder_name!r} missing field {name!r}", file=sys.stderr)
        elif is_list and not isinstance(data[name], list):
            valid = False
            print(
                f"  Warning: {folder_name!r} field {name!r} should be a list,"
                f" got {type(data[name]).__name__!r}",
                file=sys.stderr,
            )
        elif not is_list and not isinstance(data[name], (str, type(None))):
            valid = False
            print(
                f"  Warning: {folder_name!r} field {name!r} should be a string,"
                f" got {type(data[name]).__name__!r}",
                file=sys.stderr,
            )
    for field in data:
        if field not in _STRING_FIELDS | _LIST_FIELDS:
            valid = False
            print(f"  Warning: {folder_name!r} has unexpected field {field!r}", file=sys.stderr)
    return valid


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

def find_pairs(bewerbungen_dir: Path) -> list[tuple[Path, Path]]:
    """
    Return sorted (md_path, json_path) pairs from subdirectories of *bewerbungen_dir*.

    Recognises optional language subdirectories (``de/``, ``en/``); if present,
    all of them are scanned.  Each sample subdirectory must contain exactly one
    ``.md`` and one non-metadata ``.json`` file.
    """
    lang_dirs = sorted(
        d for d in bewerbungen_dir.iterdir()
        if d.is_dir() and d.name in ("de", "en")
    )
    search_dirs = lang_dirs if lang_dirs else [bewerbungen_dir]

    pairs: list[tuple[Path, Path]] = []
    for lang_dir in search_dirs:
        for subdir in sorted(lang_dir.iterdir()):
            if not subdir.is_dir():
                continue
            mds   = list(subdir.glob("*.md"))
            jsons = [p for p in subdir.glob("*.json") if "_metadata" not in p.name]

            if not mds:
                print(f"  Warning: no .md file in {subdir.name!r} — skipping", file=sys.stderr)
                continue
            if not jsons:
                print(f"  Warning: no .json file in {subdir.name!r} — skipping", file=sys.stderr)
                continue
            if len(mds) > 1:
                print(f"  Warning: multiple .md in {subdir.name!r}, using {mds[0].name!r}", file=sys.stderr)
            if len(jsons) > 1:
                print(f"  Warning: multiple .json in {subdir.name!r}, using {jsons[0].name!r}", file=sys.stderr)

            pairs.append((mds[0], jsons[0]))
    return pairs


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write *records* as JSONL to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------

def build_records(
    pair_list: list[tuple[Path, Path]],
    rng: random.Random,
    augment: int = 0,
    apply_augment: bool = True,
    quiet: bool = False,
) -> list[dict]:
    """
    Convert (md_path, json_path) pairs into ShareGPT conversation records.

    When *apply_augment* is True and *augment* > 0, each CV produces one
    canonical record (DD.MM.YYYY, canonical field order) plus *augment*
    records with randomly shuffled field order and a randomly selected
    non-canonical date format.
    """
    records: list[dict] = []
    for md_path, json_path in pair_list:
        try:
            md_content = md_path.read_text(encoding="utf-8").strip()
            json_data  = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  Warning: could not read {md_path.parent.name!r} — {exc}", file=sys.stderr)
            continue

        if not validate_json_record(json_data, md_path.parent.name):
            print(
                f"  Warning: skipping {md_path.parent.name!r} due to invalid JSON schema",
                file=sys.stderr,
            )
            continue

        if not quiet:
            print(f"  {md_path.parent.name}")

        records.append(_make_record(json_data, md_content, FIELD_NAMES, _CANONICAL_DATE_FMT_IDX))

        if apply_augment and augment > 0:
            for _ in range(augment):
                shuffled  = FIELD_NAMES[:]
                rng.shuffle(shuffled)
                date_idx  = rng.choice(_NON_CANONICAL_DATE_IDXS)
                records.append(_make_record(json_data, md_content, shuffled, date_idx))

    return records


def _make_record(json_data: dict, md_content: str, field_order: list[str], date_fmt_idx: int) -> dict:
    return {
        "conversations": [
            {"role": "system",    "content": make_system_prompt(field_order, date_fmt_idx)},
            {"role": "user",      "content": md_content},
            {"role": "assistant", "content": make_assistant_json(json_data, field_order, date_fmt_idx)},
        ]
    }


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

    if not dry_run and output_path is None:
        raise ValueError("output_path is required when dry_run=False")

    rng = random.Random(seed)

    if split > 0.0:
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        n_eval_pairs = max(1, round(len(shuffled) * split))
        eval_pairs   = shuffled[:n_eval_pairs]
        train_pairs  = shuffled[n_eval_pairs:]
    else:
        train_pairs, eval_pairs = list(pairs), []

    if split > 0.0:
        if not quiet:
            print("--- train pairs ---")
        train_records = build_records(train_pairs, rng, augment=augment, apply_augment=True,  quiet=quiet)
        if not quiet:
            print("--- eval pairs ---")
        eval_records  = build_records(eval_pairs,  rng, augment=augment, apply_augment=False, quiet=quiet)

        if not train_records:
            print("Error: train split produced no valid samples.", file=sys.stderr)
            sys.exit(1)
        if not eval_records:
            print("Error: eval split produced no valid samples.", file=sys.stderr)
            sys.exit(1)

        rng.shuffle(train_records)
        n_train, n_eval = len(train_records), len(eval_records)

        print(f"\nTotal pairs: {len(pairs)} ({len(train_pairs)} train, {len(eval_pairs)} eval)")
        if augment > 0:
            print(
                f"Augmentation: {augment} extra permutations per train CV → {n_train} train records"
                f" (field order + date format varied)"
            )
        else:
            print(f"Total: {n_train} train, {n_eval} eval samples")

        if dry_run:
            print("Dry run — no file written.")
            return

        eval_path = output_path.parent / (output_path.stem + "_eval" + output_path.suffix)
        write_jsonl(output_path, train_records)
        write_jsonl(eval_path, eval_records)
        print(f"  Train ({n_train}): {output_path}")
        print(f"  Eval  ({n_eval}): {eval_path}")

    else:
        records = build_records(train_pairs, rng, augment=augment, apply_augment=True, quiet=quiet)

        if not records:
            print("Error: no valid training samples produced.", file=sys.stderr)
            sys.exit(1)

        n_total = len(records)
        print(f"\nTotal: {n_total} training samples from {len(pairs)} folders")
        if augment > 0:
            print(f"(1 canonical + {augment} augmented permutations per CV, field order + date format varied)")

        if dry_run:
            print("Dry run — no file written.")
            return

        write_jsonl(output_path, records)
        print(f"Written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert folder-per-sample bewerbungen data into a ShareGPT JSONL dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bewerbungen-dir", required=True, type=Path,
        help="Directory whose subdirectories each contain one .md and one .json file.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL file path (e.g. data/bewerbungen_train.jsonl). Required unless --dry-run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and count samples without writing output.",
    )
    parser.add_argument(
        "--split", type=float, default=0.0, metavar="RATIO",
        help=(
            "Fraction of samples to hold out as an eval set (e.g. 0.1 for 10%%). "
            "Writes an additional '<output-stem>_eval<suffix>' file. "
            "Requires --output. Default: 0.0 (no split)."
        ),
    )
    parser.add_argument(
        "--augment", type=int, default=0, metavar="N",
        help=(
            "Number of additional field-order permutations per CV for the train split. "
            "Each also gets a random non-canonical date format. "
            "Default: 0 (canonical only). Example: --augment 4 → 5 records per CV."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling and augmentation (default: 42).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-folder name output; print only totals.",
    )
    args = parser.parse_args()

    bewerbungen_dir = args.bewerbungen_dir.expanduser()
    output_path     = args.output.expanduser() if args.output else None

    if not bewerbungen_dir.is_dir():
        print(f"Error: --bewerbungen-dir '{bewerbungen_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    if not args.dry_run and output_path is None:
        parser.error("--output is required unless --dry-run is set.")
    if args.split and not (0.0 < args.split < 1.0):
        parser.error("--split must be in the range (0.0, 1.0).")

    print(f"Processing bewerbungen in: {bewerbungen_dir}")
    convert(
        bewerbungen_dir, output_path,
        dry_run=args.dry_run, split=args.split, seed=args.seed,
        quiet=args.quiet, augment=args.augment,
    )


if __name__ == "__main__":
    main()
