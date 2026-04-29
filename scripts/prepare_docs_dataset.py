#!/usr/bin/env python3
"""
Convert paired .md / .txt document files into a ShareGPT-format JSONL dataset.

## Input format

Each document is represented by two files with the same base name:

  <name>.md   – The document content (combined with DOCS_SYSTEM_PREAMBLE as
                the system message).
  <name>.txt  – Q&A pairs that the model should learn from this document.

### .txt format

Samples are separated by a line containing only "--".

Each sample starts with a question line prefixed by "> ", followed by one or
more lines that form the answer.

**Option expansion** – A question may contain one or more `{opt1, opt2, ...}`
blocks.  All combinations of options across all blocks are expanded into
separate training samples, each paired with the same answer.

  - `_` (underscore) denotes the empty option.
  - When `_` is chosen, the block is replaced by an empty string.
  - When any other option is chosen, the block is replaced by a space
    followed by the option text.  This ensures the option attaches naturally
    to the preceding word even when the `{` is not preceded by a space in
    the template.
  - After substitution, consecutive spaces are collapsed to a single space
    and the result is stripped.

Example:

  > Was ist das Thema{_, dieses Briefs, dieses Dokuments}?

  Expands to three questions:
    "Was ist das Thema?"
    "Was ist das Thema dieses Briefs?"
    "Was ist das Thema dieses Dokuments?"

  > {Was, Wie} ist der Absender {dieser Rechnung, dieses Belegs}?

  Expands to four questions (all combinations):
    "Was ist der Absender dieser Rechnung?"
    "Was ist der Absender dieses Belegs?"
    "Wie ist der Absender dieser Rechnung?"
    "Wie ist der Absender dieses Belegs?"

## System message

Each training record uses the following as the system message:

  DOCS_SYSTEM_PREAMBLE + <document markdown content>

The same preamble is used at evaluation time (imported by evaluate_model.py).
DOCS_SYSTEM_PREAMBLE already ends with a trailing newline so the document
content follows immediately on the next line.

## Output format

One JSON object per line (JSONL), ShareGPT-style:

  {"conversations": [
    {"role": "system",    "content": "<DOCS_SYSTEM_PREAMBLE + md content>"},
    {"role": "user",      "content": "<expanded question>"},
    {"role": "assistant", "content": "<answer>"}
  ]}

## Usage

  python scripts/prepare_docs_dataset.py --docs-dir ~/docs --output data/train.jsonl

  # Dry-run: print stats without writing
  python scripts/prepare_docs_dataset.py --docs-dir ~/docs --dry-run

  # Split into train / eval (10% eval)
  python scripts/prepare_docs_dataset.py --docs-dir ~/docs --output data/docs_train.jsonl --split 0.1
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# System preamble (shared with evaluate_model.py)
# ---------------------------------------------------------------------------

# This preamble is prepended to every document in both training data and at
# evaluation time (evaluate_model.py imports this constant).  Keep both in
# sync: regenerate the JSONL whenever this value changes.
DOCS_SYSTEM_PREAMBLE = """\
Du bist ein präziser Dokumentenextraktor. Deine einzige Aufgabe ist es, \
Fragen über das folgende Dokument zu beantworten.

Regeln (strikt einhalten):
1. Antworte AUSSCHLIESSLICH mit dem exakten Wert aus dem Dokument.
2. Kein einleitender Satz, keine Erklärung, kein Kommentar.
3. Keine Markdown-Formatierung (kein **fett**, kein *kursiv*, keine Listen).
4. Mehrzeilige Werte (z. B. Adressen) zeilenweise ausgeben, genau wie im Dokument.
5. Füge niemals Text hinzu, der nicht wörtlich aus dem Dokument stammt.

Beispiel:
  Frage: Von wem ist der Brief?
  Richtige Antwort: Muster GmbH
  Falsche Antwort: Der Brief stammt von der Muster GmbH.

--- Dokument ---
"""

# ---------------------------------------------------------------------------
# Option expansion
# ---------------------------------------------------------------------------

_OPTION_BLOCK = re.compile(r"\{([^}]+)\}")


def expand_options(template: str) -> list[str]:
    """
    Return all expanded strings produced by substituting every combination of
    options in all `{opt1, opt2, ...}` blocks found in *template*.

    - `_` → empty string (the block vanishes).
    - Any other option → a space followed by the option text.

    After each substitution, consecutive spaces are collapsed and the string
    is stripped.
    """
    # Collect each block and its option list
    blocks: list[tuple[str, list[str]]] = []
    for match in _OPTION_BLOCK.finditer(template):
        raw_options = [o.strip() for o in match.group(1).split(",")]
        blocks.append((match.group(0), raw_options))

    if not blocks:
        return [template.strip()]

    # Build a list of (placeholder → replacement) lists, one per block
    per_block_choices: list[list[tuple[str, str]]] = []
    for placeholder, options in blocks:
        choices = []
        for opt in options:
            replacement = "" if opt == "_" else f" {opt}"
            choices.append((placeholder, replacement))
        per_block_choices.append(choices)

    results: list[str] = []
    for combo in itertools.product(*per_block_choices):
        text = template
        for placeholder, replacement in combo:
            # Replace only the first occurrence so that duplicate placeholders
            # are handled independently in each iteration.
            text = text.replace(placeholder, replacement, 1)
        # Collapse multiple spaces and strip
        text = re.sub(r" {2,}", " ", text).strip()
        results.append(text)

    return results


# ---------------------------------------------------------------------------
# .txt parser
# ---------------------------------------------------------------------------

def parse_txt(txt_content: str) -> list[tuple[str, str]]:
    """
    Parse a .txt Q&A file into a list of (raw_question, answer) tuples.

    Raw questions may contain option blocks; callers are responsible for
    expanding them.  Each sample is delimited by a line containing only "--".
    The question is the first line (stripped of the leading "> ").
    All subsequent non-empty lines form the (possibly multi-line) answer.
    """
    samples: list[tuple[str, str]] = []

    for block in re.split(r"^--\s*$", txt_content, flags=re.MULTILINE):
        lines = [line.rstrip() for line in block.splitlines()]
        # Drop leading/trailing blank lines
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()

        if not lines:
            continue

        # First non-empty line must be the question
        question_line = lines[0]
        if not question_line.startswith("> "):
            # Malformed block – skip with a warning
            print(f"  Warning: skipping block without '> ' prefix: {question_line!r}", file=sys.stderr)
            continue

        question = question_line[2:]  # strip "> "
        answer_lines = lines[1:]

        # Drop leading blank lines between question and answer
        while answer_lines and not answer_lines[0]:
            answer_lines.pop(0)

        answer = "\n".join(answer_lines)
        if not answer:
            print(f"  Warning: skipping Q&A block with no answer: {question!r}", file=sys.stderr)
            continue
        samples.append((question, answer))

    return samples


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

def find_pairs(docs_dir: Path) -> list[tuple[Path, Path]]:
    """Return sorted (md_path, txt_path) pairs found in *docs_dir*."""
    pairs: list[tuple[Path, Path]] = []
    for md_path in sorted(docs_dir.glob("*.md")):
        txt_path = md_path.with_suffix(".txt")
        if txt_path.exists():
            pairs.append((md_path, txt_path))
        else:
            print(f"  Warning: no matching .txt for {md_path.name}", file=sys.stderr)
    return pairs


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    docs_dir: Path,
    output_path: Path | None,
    dry_run: bool = False,
    split: float = 0.0,
    seed: int = 42,
) -> None:
    pairs = find_pairs(docs_dir)
    if not pairs:
        print(f"No .md/.txt pairs found in {docs_dir}", file=sys.stderr)
        sys.exit(1)

    if not dry_run:
        if output_path is None:
            raise ValueError("output_path is required when dry_run=False")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    if split > 0.0:
        shuffled_pairs = list(pairs)
        rng.shuffle(shuffled_pairs)
        n_eval_pairs = max(1, round(len(shuffled_pairs) * split))
        eval_pairs = shuffled_pairs[:n_eval_pairs]
        train_pairs = shuffled_pairs[n_eval_pairs:]
        if not train_pairs:
            print(
                "Error: train split would be empty. Use a smaller --split or add more documents.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        train_pairs = list(pairs)
        eval_pairs = []

    def _build_records(pair_list: list[tuple[Path, Path]]) -> list[dict]:
        records: list[dict] = []
        for md_path, txt_path in pair_list:
            doc_content = md_path.read_text(encoding="utf-8")
            system_content = DOCS_SYSTEM_PREAMBLE + doc_content
            txt_content = txt_path.read_text(encoding="utf-8")

            raw_samples = parse_txt(txt_content)
            file_samples = 0

            for raw_question, answer in raw_samples:
                for expanded_question in expand_options(raw_question):
                    records.append({
                        "conversations": [
                            {"role": "system",    "content": system_content},
                            {"role": "user",      "content": expanded_question},
                            {"role": "assistant", "content": answer},
                        ]
                    })
                    file_samples += 1

            print(f"  {md_path.name}: {len(raw_samples)} Q&A pairs → {file_samples} samples")
        return records

    if dry_run:
        records = _build_records(pairs)
        if not records:
            print("No training samples produced.", file=sys.stderr)
            sys.exit(1)
        print(f"\nTotal: {len(records)} training samples from {len(pairs)} document(s)")
        print("Dry run — no file written.")
        return

    if split > 0.0:
        print("--- train documents ---")
        train_records = _build_records(train_pairs)
        print("--- eval documents ---")
        eval_records = _build_records(eval_pairs)
        if not train_records:
            print("Error: train split produced no samples.", file=sys.stderr)
            sys.exit(1)
        if not eval_records:
            print("Error: eval split produced no samples.", file=sys.stderr)
            sys.exit(1)
        rng.shuffle(train_records)
        n_train = len(train_records)
        n_eval = len(eval_records)
        eval_path = output_path.parent / (output_path.stem + "_eval" + output_path.suffix)

        with open(output_path, "w", encoding="utf-8") as f:
            for r in train_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(eval_path, "w", encoding="utf-8") as f:
            for r in eval_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"\nTotal: {n_train + n_eval} samples from {len(pairs)} document(s) (split seed={seed})")
        print(f"  Train ({n_train}): {output_path}")
        print(f"  Eval  ({n_eval}): {eval_path}")
    else:
        records = _build_records(train_pairs)
        if not records:
            print("Error: no training samples produced.", file=sys.stderr)
            sys.exit(1)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nTotal: {len(records)} training samples from {len(pairs)} document(s)")
        print(f"Written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert paired .md/.txt document files into a ShareGPT JSONL dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--docs-dir",
        required=True,
        type=Path,
        help="Directory containing .md and .txt file pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file path (e.g. data/train.jsonl). Required unless --dry-run is set.",
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
            "Fraction of samples to hold out as an eval set (e.g. 0.1 for 10%%). "
            "Writes an additional '<output-stem>_eval<suffix>' file alongside the main output. "
            "Requires --output. Default: 0.0 (no split)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling before the train/eval split (default: 42).",
    )
    args = parser.parse_args()

    docs_dir = args.docs_dir.expanduser()
    output_path = args.output.expanduser() if args.output else None

    if not docs_dir.is_dir():
        print(f"Error: --docs-dir '{docs_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run and output_path is None:
        parser.error("--output is required unless --dry-run is set.")

    if args.split and (args.split <= 0.0 or args.split >= 1.0):
        parser.error("--split must be in the range (0.0, 1.0).")

    print(f"Processing docs in: {docs_dir}")
    convert(docs_dir, output_path, dry_run=args.dry_run, split=args.split, seed=args.seed)


if __name__ == "__main__":
    main()
