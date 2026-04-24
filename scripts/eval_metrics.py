#!/usr/bin/env python3
"""
Evaluation metrics for comparing model responses to ground-truth answers.

All public functions accept raw strings and handle normalization internally.
The <think>...</think> block (and its contents) is stripped from model output
before any metric is computed.

Metrics returned are all in the range [0.0, 1.0].
"""

from __future__ import annotations

import difflib
import re

from rouge_score import rouge_scorer

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove closed or truncated <think> blocks from model output."""
    text = _THINK_BLOCK.sub("", text)
    start = re.search(r"<think>", text, flags=re.IGNORECASE)
    if start:
        text = text[:start.start()]
    return text.strip()


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def exact_match(prediction: str, reference: str) -> float:
    """1.0 if normalized strings are identical, else 0.0."""
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 between prediction and reference.

    Computes precision and recall over whitespace-split tokens (after
    normalization) and returns the harmonic mean.  Returns 0.0 if either
    string is empty after normalization.
    """
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_set = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1

    ref_set = {}
    for t in ref_tokens:
        ref_set[t] = ref_set.get(t, 0) + 1

    common = sum(min(pred_set.get(t, 0), ref_set.get(t, 0)) for t in ref_set)
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def edit_similarity(prediction: str, reference: str) -> float:
    """
    Normalized edit similarity: 1 − (edit_distance / max_len).

    Uses difflib's SequenceMatcher ratio (operates on characters after
    normalization), which is equivalent to 2*LCS / (len_a + len_b).
    """
    a = normalize(prediction)
    b = normalize(reference)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F-scores.

    Returns a dict with keys: rouge1_f, rouge2_f, rougeL_f.
    rouge_scorer handles Unicode correctly and does not require stemming.
    """
    scores = _ROUGE.score(reference, prediction)
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 6),
        "rouge2_f": round(scores["rouge2"].fmeasure, 6),
        "rougeL_f": round(scores["rougeL"].fmeasure, 6),
    }


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def compute_all(raw_prediction: str, reference: str) -> dict[str, float]:
    """
    Strip thinking tokens from *raw_prediction*, then compute all metrics
    against *reference*.

    Returns a dict with keys:
      exact_match, token_f1, rouge1_f, rouge2_f, rougeL_f, edit_sim
    All values are floats in [0.0, 1.0].
    """
    prediction = strip_thinking(raw_prediction)

    result: dict[str, float] = {
        "exact_match": exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "edit_sim": edit_similarity(prediction, reference),
    }
    result.update(rouge_scores(prediction, reference))
    return result
