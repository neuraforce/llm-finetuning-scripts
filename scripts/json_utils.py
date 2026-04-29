"""
Robust JSON parsing utilities for LLM outputs.

LLMs occasionally return:
- Reasoning wrapped in <think>...</think> tags
- Markdown code fences (```json ... ```)
- Trailing commas before closing braces
- Single-quoted strings instead of double-quoted
- Truncated output (missing closing braces)
- Prose mixed with the JSON object

``parse_llm_json`` applies a recovery chain of strategies and returns the
first successfully parsed dict, or ``None`` if all strategies fail.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator

__all__ = ["parse_llm_json"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _iter_outside_strings(text: str) -> Iterator[tuple[int, str]]:
    """
    Yield ``(index, char)`` for every character in *text* that lies **outside**
    a double-quoted JSON string literal.

    Correctly handles ``\\``-escaped characters inside strings (e.g. ``\\"``
    does not close the string).
    """
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            yield i, ch


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------


def _strip_think_blocks(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning sections."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_code_fences(text: str) -> str:
    """Remove optional `` ```json `` / `` ``` `` or ``~~~`` markdown code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json|python|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"^~~~(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*~~~\s*$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Recovery helpers
# ---------------------------------------------------------------------------


def _extract_outermost_object(text: str) -> str | None:
    """Return the substring spanning the first top-level ``{…}`` block."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in _iter_outside_strings(text[start:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : start + i + 1]
    return None


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` or ``]``."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_single_quotes(text: str) -> str:
    """
    Best-effort conversion of single-quoted JSON strings to double-quoted.
    Only handles simple cases; may corrupt strings containing apostrophes.
    """
    text = re.sub(r"'([^']*?)'(\s*:)", r'"\1"\2', text)
    text = re.sub(r"(:\s*)'([^']*?)'", r'\1"\2"', text)
    return text


def _balance_braces(text: str) -> str:
    """Append missing ``}`` and ``]`` to close truncated JSON."""
    opens = opens_sq = 0
    for _, ch in _iter_outside_strings(text):
        if ch == "{":
            opens += 1
        elif ch == "}":
            opens -= 1
        elif ch == "[":
            opens_sq += 1
        elif ch == "]":
            opens_sq -= 1
    return text + "]" * max(0, opens_sq) + "}" * max(0, opens)


def _try_json_repair(text: str) -> dict | None:
    """Use the optional ``json_repair`` library if installed."""
    try:
        import json_repair  # type: ignore[import-untyped]
        result = json_repair.repair_json(text, return_objects=True)
        if isinstance(result, dict):
            return result
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_llm_json(text: str) -> dict | None:
    """
    Parse a JSON object from an LLM response using a multi-step recovery chain.

    Returns the parsed dict on success, or ``None`` if all strategies fail.
    Each successful recovery is logged at DEBUG level.
    """
    if not text or not text.strip():
        return None

    # Pre-processing: strip think blocks and code fences unconditionally
    clean = _strip_code_fences(_strip_think_blocks(text))

    # Strategy 1 — direct parse after pre-processing (ideal case)
    try:
        result = json.loads(clean)
        if isinstance(result, dict):
            logger.debug("parse_llm_json: direct parse succeeded")
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2 — extract first top-level {…} (removes surrounding prose)
    extracted = _extract_outermost_object(clean)
    if extracted and extracted != clean:
        try:
            result = json.loads(extracted)
            if isinstance(result, dict):
                logger.debug("parse_llm_json: outermost-object extraction succeeded")
                return result
        except json.JSONDecodeError:
            pass
        clean = extracted  # narrow scope for subsequent strategies

    # Strategy 3 — strip trailing commas
    no_trailing = _strip_trailing_commas(clean)
    try:
        result = json.loads(no_trailing)
        if isinstance(result, dict):
            logger.debug("parse_llm_json: trailing-comma removal succeeded")
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 4 — fix single-quoted strings
    fixed_quotes = _fix_single_quotes(no_trailing)
    try:
        result = json.loads(fixed_quotes)
        if isinstance(result, dict):
            logger.debug("parse_llm_json: single-quote fix succeeded")
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 5 — balance truncated braces/brackets
    balanced = _balance_braces(fixed_quotes)
    try:
        result = json.loads(balanced)
        if isinstance(result, dict):
            logger.debug("parse_llm_json: brace balancing succeeded")
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 6 — optional json_repair library (last resort)
    repaired = _try_json_repair(balanced)
    if repaired is not None:
        logger.debug("parse_llm_json: json_repair library succeeded")
        return repaired

    logger.warning("parse_llm_json: all strategies failed; raw text: %.120r", text)
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning sections."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_code_fences(text: str) -> str:
    """Remove optional ```json / ``` or ~~~ markdown code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json|python|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"^~~~(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*~~~\s*$", "", text)
    return text.strip()


def _extract_outermost_object(text: str) -> str | None:
    """Return the substring spanning the first top-level ``{...}`` block."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` or ``]``."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_single_quotes(text: str) -> str:
    """
    Best-effort conversion of single-quoted JSON strings to double-quoted.
    Only handles simple cases; may corrupt strings that contain apostrophes.
    """
    # Replace 'key': and ': 'value' patterns
    text = re.sub(r"'([^']*?)'(\s*:)", r'"\1"\2', text)
    text = re.sub(r"(:\s*)'([^']*?)'", r'\1"\2"', text)
    return text


def _balance_braces(text: str) -> str:
    """Append missing ``}`` and ``]`` to close truncated JSON."""
    opens = 0
    opens_sq = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            opens += 1
        elif ch == "}":
            opens -= 1
        elif ch == "[":
            opens_sq += 1
        elif ch == "]":
            opens_sq -= 1

    suffix = "]" * max(0, opens_sq) + "}" * max(0, opens)
    return text + suffix


def _try_json_repair(text: str) -> dict | None:
    """Use the ``json_repair`` library if available."""
    try:
        import json_repair  # type: ignore[import-untyped]  # optional dep
        result = json_repair.repair_json(text, return_objects=True)
        if isinstance(result, dict):
            return result
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_llm_json(text: str) -> dict | None:
    """
    Parse a JSON object from an LLM response using a multi-step recovery chain.

    Returns the parsed dict on success, or ``None`` if all strategies fail.
    Each successful recovery is logged at DEBUG level.
    """
    if not text or not text.strip():
        return None

    # Strategy 1 — strip think blocks
    clean = _strip_think_blocks(text)

    # Strategy 2 — strip code fences
    clean = _strip_code_fences(clean)

    # Strategy 3 — direct parse (ideal case)
    try:
        result = json.loads(clean)
        if isinstance(result, dict):
            logger.debug("parse_llm_json: direct parse succeeded")
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 4 — extract outermost {...} and retry
    extracted = _extract_outermost_object(clean)
    if extracted and extracted != clean:
        try:
            result = json.loads(extracted)
            if isinstance(result, dict):
                logger.debug("parse_llm_json: outermost-object extraction succeeded")
                return result
        except json.JSONDecodeError:
            pass
        clean = extracted  # carry forward for subsequent strategies

    # Strategy 5 — strip trailing commas
    no_trailing = _strip_trailing_commas(clean)
    try:
        result = json.loads(no_trailing)
        if isinstance(result, dict):
            logger.debug("parse_llm_json: trailing-comma removal succeeded")
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 6 — fix single-quoted strings
    fixed_quotes = _fix_single_quotes(no_trailing)
    try:
        result = json.loads(fixed_quotes)
        if isinstance(result, dict):
            logger.debug("parse_llm_json: single-quote fix succeeded")
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 7 — balance truncated braces
    balanced = _balance_braces(fixed_quotes)
    try:
        result = json.loads(balanced)
        if isinstance(result, dict):
            logger.debug("parse_llm_json: brace balancing succeeded")
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 8 — optional json_repair library (use most-processed form)
    repaired = _try_json_repair(balanced)
    if repaired is not None:
        logger.debug("parse_llm_json: json_repair library succeeded")
        return repaired

    logger.warning("parse_llm_json: all strategies failed; raw text: %.120r", text)
    return None
