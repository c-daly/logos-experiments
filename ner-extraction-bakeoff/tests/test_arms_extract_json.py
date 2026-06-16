"""Tests for _extract_json -- runs without the hermes venv (all hermes
imports in arms.py are deferred inside async functions)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arms import _extract_json


def test_extract_json_plain():
    raw = '{"entities": [], "relations": []}'
    assert _extract_json(raw) == {"entities": [], "relations": []}


def test_extract_json_strips_think_block():
    raw = (
        '<think>some internal reasoning</think>\n'
        '{"entities": [{"name": "cat", "type": "animal"}], "relations": []}'
    )
    result = _extract_json(raw)
    assert result["entities"] == [{"name": "cat", "type": "animal"}]


def test_extract_json_think_multiline():
    raw = "<think>\nstep 1: ...\nstep 2: ...\n</think>\n{\"entities\": [], \"relations\": []}"
    assert _extract_json(raw) == {"entities": [], "relations": []}


def test_extract_json_fenced_block():
    raw = "```json\n{\"entities\": [], \"relations\": []}\n```"
    assert _extract_json(raw) == {"entities": [], "relations": []}


def test_extract_json_brace_substring():
    raw = "Sure, here is the JSON:\n{\"entities\": [], \"relations\": []}\nHope that helps!"
    assert _extract_json(raw) == {"entities": [], "relations": []}


def test_extract_json_raises_on_no_json():
    with pytest.raises(ValueError, match="no valid JSON found"):
        _extract_json("<think>thinking</think>\nSorry, I cannot help with that.")
