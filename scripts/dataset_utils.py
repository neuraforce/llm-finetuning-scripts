"""
Dataset loading and Qwen 3 chat template formatting utilities.

Supports all task types:
  - Instruction / single-turn chat
  - Multi-turn conversation
  - Vision-language (image + text)
  - Tool use (tool_call / tool_response)
  - Thinking / reasoning (<think> tokens preserved)

Expected dataset format (ShareGPT-style JSONL):
  {"conversations": [{"role": "system"|"user"|"assistant"|"tool", "content": ...}, ...]}

For vision turns, "content" may be a list:
  [{"type": "image", "image": "<path_or_url>"}, {"type": "text", "text": "..."}]
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from datasets import Dataset


class ConversationDataset:
    """Minimal list-backed dataset for records Arrow cannot represent."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.column_names = list(records[0].keys()) if records else []

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _make_dataset(records: list[dict]) -> Dataset | ConversationDataset:
    """Use HF Dataset for text-only data and list-backed storage for multimodal data."""
    if has_multimodal_content(records):
        return ConversationDataset(records)
    return Dataset.from_list(records)


def load_dataset_from_config(cfg: dict) -> tuple[Dataset | ConversationDataset, Dataset | ConversationDataset | None]:
    """
    Load train (and optionally eval) datasets from paths in the config dict.

    Returns:
        (train_dataset, eval_dataset)  – eval_dataset is None if not configured.
    """
    train_records = load_jsonl(cfg["data"]["train_file"])
    max_samples = cfg["data"].get("max_samples")
    if max_samples is not None:
        train_records = train_records[:max_samples]
    train_ds = _make_dataset(train_records)

    eval_ds = None
    eval_file = cfg["data"].get("eval_file")
    if eval_file:
        eval_path = Path(eval_file)
        if eval_path.exists():
            eval_records = load_jsonl(eval_path)
            if max_samples is not None:
                eval_records = eval_records[:max_samples]
            eval_ds = _make_dataset(eval_records)
        else:
            warnings.warn(
                f"eval_file '{eval_file}' not found — skipping evaluation.",
                stacklevel=2,
            )

    return train_ds, eval_ds


def has_multimodal_content(records: Dataset | list[dict] | None) -> bool:
    """Return True if any conversation turn contains list-style vision content."""
    if records is None:
        return False

    iterable = records
    for record in iterable:
        for turn in record.get("conversations", []):
            content = turn.get("content")
            if isinstance(content, list):
                return True
    return False


def format_conversation_for_qwen(
    conversations: list[dict[str, Any]],
    tokenizer,
    add_generation_prompt: bool = False,
) -> str:
    """
    Apply the Qwen 3 chat template to a list of conversation turns.

    Handles:
      - Plain text content (str)
      - Multimodal content (list of {"type": "image"|"text", ...})
      - tool_call / tool_response embedded in assistant/tool messages

    Returns the rendered string (not tokenized).
    """
    messages = [{"role": turn["role"], "content": turn["content"]} for turn in conversations]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def make_formatting_func(tokenizer):
    """
    Return a formatting function bound to a tokenizer,
    suitable for passing to SFTTrainer(formatting_func=...).

    Handles both single-example and batched calls:
    - Single: example = {"conversations": [...]}
    - Batched: example = {"conversations": [[...], [...]]}  (UnslothSFTTrainer)
    """
    def _fmt(example: dict) -> list[str]:
        convs = example.get("conversations", [])
        # UnslothSFTTrainer calls the formatter in batched mode: convs is a list of lists
        if convs and isinstance(convs[0], list):
            return [format_conversation_for_qwen(c, tokenizer) for c in convs]
        return [format_conversation_for_qwen(convs, tokenizer)]
    return _fmt
