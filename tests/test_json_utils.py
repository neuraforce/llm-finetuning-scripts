"""Unit tests for scripts/json_utils.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from json_utils import (
    _balance_braces,
    _extract_outermost_object,
    _fix_single_quotes,
    _iter_outside_strings,
    _strip_code_fences,
    _strip_think_blocks,
    _strip_trailing_commas,
    parse_llm_json,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestStripThinkBlocks:
    def test_removes_think_block(self) -> None:
        assert _strip_think_blocks("<think>reasoning</think>{\"a\": 1}") == '{"a": 1}'

    def test_removes_multiline_think_block(self) -> None:
        text = "<think>\nline 1\nline 2\n</think>\n{\"x\": 2}"
        assert _strip_think_blocks(text) == '{"x": 2}'

    def test_no_think_block_unchanged(self) -> None:
        assert _strip_think_blocks('{"a": 1}') == '{"a": 1}'

    def test_empty_string(self) -> None:
        assert _strip_think_blocks("") == ""


class TestStripCodeFences:
    def test_strips_json_fence(self) -> None:
        assert _strip_code_fences("```json\n{\"a\": 1}\n```") == '{"a": 1}'

    def test_strips_bare_fence(self) -> None:
        assert _strip_code_fences("```\n{\"a\": 1}\n```") == '{"a": 1}'

    def test_strips_tilde_fence(self) -> None:
        assert _strip_code_fences("~~~json\n{\"a\": 1}\n~~~") == '{"a": 1}'

    def test_no_fence_unchanged(self) -> None:
        assert _strip_code_fences('{"a": 1}') == '{"a": 1}'


class TestExtractOutermostObject:
    def test_extracts_embedded_object(self) -> None:
        text = 'Here is the result: {"name": "Max"} done.'
        assert _extract_outermost_object(text) == '{"name": "Max"}'

    def test_nested_objects(self) -> None:
        text = '{"a": {"b": 1}}'
        assert _extract_outermost_object(text) == '{"a": {"b": 1}}'

    def test_no_object_returns_none(self) -> None:
        assert _extract_outermost_object("no json here") is None

    def test_brace_in_string_ignored(self) -> None:
        text = '{"key": "val{ue"}'
        assert _extract_outermost_object(text) == '{"key": "val{ue"}'


class TestStripTrailingCommas:
    def test_removes_trailing_comma_before_brace(self) -> None:
        assert _strip_trailing_commas('{"a": 1,}') == '{"a": 1}'

    def test_removes_trailing_comma_before_bracket(self) -> None:
        assert _strip_trailing_commas('{"a": [1, 2,]}') == '{"a": [1, 2]}'

    def test_no_trailing_comma_unchanged(self) -> None:
        assert _strip_trailing_commas('{"a": 1}') == '{"a": 1}'


class TestFixSingleQuotes:
    def test_converts_single_quoted_key_and_value(self) -> None:
        result = _fix_single_quotes("{'name': 'Max'}")
        assert '"name"' in result
        assert '"Max"' in result

    def test_leaves_double_quoted_unchanged(self) -> None:
        text = '{"name": "Max"}'
        assert _fix_single_quotes(text) == text


class TestBalanceBraces:
    def test_adds_missing_closing_brace(self) -> None:
        result = _balance_braces('{"name": "Max"')
        assert result == '{"name": "Max"}'

    def test_adds_missing_bracket_and_brace(self) -> None:
        result = _balance_braces('{"items": [1, 2')
        assert result == '{"items": [1, 2]}'

    def test_already_balanced_unchanged(self) -> None:
        text = '{"a": 1}'
        assert _balance_braces(text) == text


# ---------------------------------------------------------------------------
# parse_llm_json — end-to-end tests
# ---------------------------------------------------------------------------


class TestParseLlmJson:
    def test_direct_parse(self) -> None:
        assert parse_llm_json('{"name": "Alice"}') == {"name": "Alice"}

    def test_code_fence(self) -> None:
        text = '```json\n{"name": "Alice"}\n```'
        assert parse_llm_json(text) == {"name": "Alice"}

    def test_think_block_stripped(self) -> None:
        text = "<think>some reasoning</think>\n{\"name\": \"Bob\"}"
        assert parse_llm_json(text) == {"name": "Bob"}

    def test_embedded_json_with_prose(self) -> None:
        text = 'Here is the extracted data: {"name": "Carol"} That is all.'
        assert parse_llm_json(text) == {"name": "Carol"}

    def test_trailing_comma(self) -> None:
        assert parse_llm_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_truncated_object_balanced(self) -> None:
        result = parse_llm_json('{"name": "Dan", "skills": ["Python"')
        assert result is not None
        assert result.get("name") == "Dan"

    def test_nested_arrays(self) -> None:
        text = '{"languages": ["German", "English"], "name": "Eve"}'
        result = parse_llm_json(text)
        assert result == {"languages": ["German", "English"], "name": "Eve"}

    def test_garbage_returns_none(self) -> None:
        assert parse_llm_json("this is not json at all") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_llm_json("") is None

    def test_none_like_input_returns_none(self) -> None:
        assert parse_llm_json("   ") is None

    def test_think_block_and_fence_combined(self) -> None:
        text = '<think>reasoning</think>\n```json\n{"key": "value"}\n```'
        assert parse_llm_json(text) == {"key": "value"}

    def test_real_world_trailing_comma_in_array(self) -> None:
        text = '{"skills": ["Python", "SQL",], "name": "Frank",}'
        result = parse_llm_json(text)
        assert result is not None
        assert "Python" in result["skills"]
        assert result["name"] == "Frank"

    def test_returns_dict_not_list(self) -> None:
        # A top-level JSON array should return None (we expect objects)
        assert parse_llm_json("[1, 2, 3]") is None


# ---------------------------------------------------------------------------
# _iter_outside_strings
# ---------------------------------------------------------------------------


class TestIterOutsideStrings:
    def _collect(self, text: str) -> list[tuple[int, str]]:
        return list(_iter_outside_strings(text))

    def test_simple_object_chars_outside_strings(self) -> None:
        text = '{"a":1}'
        outside = self._collect(text)
        chars = [c for _, c in outside]
        # { and } are outside strings
        assert "{" in chars
        assert "}" in chars

    def test_string_value_excluded(self) -> None:
        text = '{"key": "hello world"}'
        outside = self._collect(text)
        chars = [c for _, c in outside]
        # letters inside the string values should not appear outside
        # 'h' is only in a string value here
        assert "h" not in chars

    def test_escaped_quote_inside_string_does_not_end_it(self) -> None:
        text = '{"k": "say \\"hi\\""}'
        outside = self._collect(text)
        indices = [i for i, _ in outside]
        # The closing } is at the last position; confirm it is yielded
        assert len(text) - 1 in indices

    def test_empty_string(self) -> None:
        assert self._collect("") == []

    def test_no_strings_yields_all(self) -> None:
        text = "{123}"
        result = self._collect(text)
        assert len(result) == len(text)


# ---------------------------------------------------------------------------
# make_system_prompt validation (prepare_bewerbungen_dataset)
# ---------------------------------------------------------------------------


class TestMakeSystemPromptValidation:
    def test_valid_field_order_returns_string(self) -> None:
        import importlib
        import types

        # We import the module; it may print on import but that's fine.
        module_path = Path(__file__).parent.parent / "scripts" / "prepare_bewerbungen_dataset.py"
        spec = importlib.util.spec_from_file_location("prepare_bewerbungen_dataset", module_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        make = mod.make_system_prompt
        field_names = mod.FIELD_NAMES

        result = make(field_names)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_unknown_field_raises(self) -> None:
        import importlib

        module_path = Path(__file__).parent.parent / "scripts" / "prepare_bewerbungen_dataset.py"
        spec = importlib.util.spec_from_file_location("prepare_bewerbungen_dataset", module_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="unknown fields"):
            mod.make_system_prompt(["nonexistent_field_xyz"])
