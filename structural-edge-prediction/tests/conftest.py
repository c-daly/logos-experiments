"""Shared fixtures + import shim for the structural-edge-prediction tests.

Makes ``harness`` and ``eval`` importable (the experiment dir is the parent of
tests/) and exposes the loaded toy snapshot + a train/test split. No live
stack, no LLM, no graph \u2014 pure offline fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make \u0027harness\u0027/\u0027eval\u0027 importable: experiment dir is the parent of tests/.
_EXP_DIR = Path(__file__).resolve().parent.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from harness.snapshot_io import load_snapshot  # noqa: E402

TOY_FIXTURE = _EXP_DIR / "fixtures" / "toy_graph.json"


@pytest.fixture
def toy_snapshot():
    """The validated toy positive-control snapshot."""
    return load_snapshot(TOY_FIXTURE)


@pytest.fixture
def toy_path():
    """Filesystem path to the toy fixture (for CLI/end-to-end tests)."""
    return TOY_FIXTURE
