"""Blessed batch3 corpus invariants + reseed corpus selection (issue #13).

The graded reseed input is ``corpus/corpus_batch3.jsonl`` -- approved by
Chris on 2026-06-05 and checked in BYTE-IDENTICAL to the vault source. The
md5 below is pinned to the blessed source file; any drift (re-export,
normalization, editor touch) must fail loudly here, because the graded
fixtures are only reproducible from the exact bytes.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pytest

from harness.reseed import (
    GRADED_CORPUS,
    SMOKE_CORPUS,
    ReseedInputError,
    resolve_corpus_path,
    validate_corpus_items,
)

# md5 of the blessed vault source (and therefore of the checked-in copy).
BLESSED_BATCH3_MD5 = "aec00caa62b40e19fc05cc2ee549877e"

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
BATCH3 = CORPUS_DIR / GRADED_CORPUS


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in BATCH3.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_batch3_is_byte_identical_to_the_blessed_source():
    digest = hashlib.md5(BATCH3.read_bytes()).hexdigest()
    assert digest == BLESSED_BATCH3_MD5


def test_batch3_has_350_blocks_across_8_domains():
    rows = _rows()
    assert len(rows) == 350
    domains = collections.Counter(row["domain"] for row in rows)
    assert len(domains) == 8


def test_batch3_rows_pass_corpus_validation():
    # batch3 rows carry {block, domain, text, topic}: validate_corpus_items
    # requires text+domain and must accept the extra keys.
    rows = _rows()
    validate_corpus_items(rows)
    assert all(set(row) == {"block", "domain", "text", "topic"} for row in rows)


def test_validate_corpus_items_accepts_batch3_shaped_row():
    validate_corpus_items(
        [
            {
                "block": 0,
                "domain": "cell_biology",
                "text": "The cell is the basic unit of life.",
                "topic": "Cell (biology)",
            }
        ]
    )


# ---------------------------------------------------------------------------
# resolve_corpus_path: smoke default vs graded default vs explicit
# ---------------------------------------------------------------------------


def test_smoke_default_is_the_curated_corpus():
    path = resolve_corpus_path(None, graded=False, corpus_dir=CORPUS_DIR)
    assert path == CORPUS_DIR / SMOKE_CORPUS


def test_graded_default_is_batch3():
    path = resolve_corpus_path(None, graded=True, corpus_dir=CORPUS_DIR)
    assert path == BATCH3


def test_explicit_corpus_name_wins_over_graded_default():
    path = resolve_corpus_path(SMOKE_CORPUS, graded=True, corpus_dir=CORPUS_DIR)
    assert path == CORPUS_DIR / SMOKE_CORPUS


def test_explicit_corpus_resolves_under_corpus_dir_first():
    path = resolve_corpus_path(GRADED_CORPUS, graded=False, corpus_dir=CORPUS_DIR)
    assert path == BATCH3


def test_explicit_absolute_corpus_path_is_honored(tmp_path):
    other = tmp_path / "other.jsonl"
    other.write_text("{}\n", encoding="utf-8")
    path = resolve_corpus_path(str(other), graded=True, corpus_dir=CORPUS_DIR)
    assert path == other


def test_missing_explicit_corpus_raises():
    with pytest.raises(ReseedInputError, match="not found"):
        resolve_corpus_path("nope.jsonl", graded=False, corpus_dir=CORPUS_DIR)


def test_missing_default_corpus_raises(tmp_path):
    with pytest.raises(ReseedInputError, match="default corpus missing"):
        resolve_corpus_path(None, graded=True, corpus_dir=tmp_path)


# ---------------------------------------------------------------------------
# reseed driver gate (env-gated smoke unchanged; never touches a stack here)
# ---------------------------------------------------------------------------


def test_reseed_driver_refuses_without_reseed_live(monkeypatch, capsys):
    from harness import reseed

    monkeypatch.delenv("RESEED_LIVE", raising=False)
    assert reseed.main(["--graded"]) == 2
    assert "RESEED_LIVE=1" in capsys.readouterr().err
