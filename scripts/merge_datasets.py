#!/usr/bin/env python3
"""
Merge multiple JSONL dataset files into a single output file.

Useful for combining datasets from different sources (e.g. bewerbungen +
docs Q&A) before joint fine-tuning.

## Usage

  # Concatenate two files
  python scripts/merge_datasets.py \\
      --inputs data/bewerbungen_train.jsonl data/docs_train.jsonl \\
      --output data/joint_train.jsonl

  # Shuffle the merged output (recommended for joint training)
  python scripts/merge_datasets.py \\
      --inputs data/bewerbungen_train.jsonl data/docs_train.jsonl \\
      --output data/joint_train.jsonl \\
      --shuffle

  # Dry run — print counts without writing
  python scripts/merge_datasets.py \\
      --inputs data/bewerbungen_train.jsonl data/docs_train.jsonl \\
      --output data/joint_train.jsonl \\
      --dry-run

  # Use a fixed random seed for reproducibility
  python scripts/merge_datasets.py \\
      --inputs data/bewerbungen_train.jsonl data/docs_train.jsonl \\
      --output data/joint_train.jsonl \\
      --shuffle --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"  Warning: {path.name}:{lineno}: JSON decode error — {exc}",
                    file=sys.stderr,
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple JSONL files into one dataset file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        metavar="FILE",
        help="One or more input JSONL files to merge.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print record counts without writing the output file.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the merged records before writing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is set (default: 42).",
    )
    args = parser.parse_args()

    all_records: list[dict] = []
    for input_path in args.inputs:
        input_path = input_path.expanduser()
        if not input_path.exists():
            print(f"Error: input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        records = load_jsonl(input_path)
        print(f"  {input_path.name}: {len(records)} records")
        all_records.extend(records)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(all_records)
        print(f"\nShuffled with seed {args.seed}")

    if args.dry_run:
        print(f"Total: {len(all_records)} records (dry run — nothing written)")
        return

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Total: {len(all_records)} records written to {output_path}")


if __name__ == "__main__":
    main()
