"""_ingest must use the proven /llm echo ingestion path (#18)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import reseed  # noqa: E402


class _Resp:
    def raise_for_status(self):
        return None


def test_ingest_posts_llm_echo(monkeypatch):
    calls = []

    import httpx

    def fake_post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return _Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    reseed._ingest("a red block sits on the table", "blocks", "http://h:17000")
    assert len(calls) == 1
    url, payload, _ = calls[0]
    assert url == "http://h:17000/llm"
    assert payload["provider"] == "echo"
    assert payload["prompt"] == "a red block sits on the table"
    assert payload["metadata"]["domain"] == "blocks"


def test_ingest_retries_then_raises(monkeypatch):
    import httpx

    attempts = {"n": 0}

    def fail_post(url, *, json, timeout):
        attempts["n"] += 1
        raise ConnectionError("down")

    monkeypatch.setattr(httpx, "post", fail_post)
    monkeypatch.setattr(reseed.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="after 5 retries"):
        reseed._ingest("x", "d", "http://h:17000")
    assert attempts["n"] == 5


class _CountingClient:
    def __init__(self, counts):
        self._counts = list(counts)

    def _execute_query(self, query, params):
        c = self._counts.pop(0) if self._counts else 999
        return [{"c": c}]


def test_settle_graph_waits_for_stable_count(monkeypatch):
    monkeypatch.setattr(reseed.time, "sleep", lambda s: None)
    monkeypatch.setattr(reseed, "_pending_proposals", lambda: 0)
    client = _CountingClient([100, 130, 130, 130, 130])
    assert reseed._settle_graph(client, stable_polls=3, interval=0) == 130


def test_settle_graph_blocks_on_nonempty_queue(monkeypatch):
    """Count-quiet with a deep queue must NOT settle (#18: 317 pending
    proposals behind a stable count froze a 9%-coverage fixture set)."""
    monkeypatch.setattr(reseed.time, "sleep", lambda s: None)
    depths = iter([5, 5, 0, 0, 0, 0])
    monkeypatch.setattr(reseed, "_pending_proposals", lambda: next(depths, 0))
    client = _CountingClient([200] * 10)
    assert reseed._settle_graph(client, stable_polls=3, interval=0) == 200
    # 2 queue-deep polls reset stability; settle required 3 quiet polls after.


def test_settle_graph_fails_loudly_at_cap(monkeypatch):
    monkeypatch.setattr(reseed.time, "sleep", lambda s: None)
    monkeypatch.setattr(reseed, "_pending_proposals", lambda: 0)
    ticks = iter(range(1000))
    monkeypatch.setattr(reseed.time, "monotonic", lambda: float(next(ticks)))
    client = _CountingClient(range(1000))  # never stabilizes
    with pytest.raises(RuntimeError, match="did not settle"):
        reseed._settle_graph(client, stable_polls=3, interval=0, cap=10.0)


class _GrowthClient:
    """Counts: returned in sequence to both _node_count pollers."""

    def __init__(self, counts):
        self._counts = list(counts)
        self.queries = 0

    def _execute_query(self, query, params):
        self.queries += 1
        c = self._counts.pop(0) if self._counts else self._last
        self._last = c
        return [{"c": c}]


def test_ingest_corpus_waits_for_extraction_evidence_then_depth(monkeypatch):
    posted = []
    monkeypatch.setattr(reseed, "_ingest", lambda t, d, u: posted.append(t))
    monkeypatch.setattr(reseed.time, "sleep", lambda s: None)
    # block1: pre-depth 0 -> two polls of no evidence -> queue grows to 1
    # (extraction done) -> depth 1 <= 4 proceeds. block2: pre-depth 1 ->
    # node count grows instead (proposal consumed) -> depth fine.
    depths = iter([0, 0, 0, 1, 1, 1, 1, 1])
    monkeypatch.setattr(reseed, "_pending_proposals", lambda: next(depths, 1))
    client = _GrowthClient([50, 50, 50, 50, 50, 51, 51])
    corpus = [
        {"text": "a", "domain": "d"},
        {"text": "b", "domain": "d"},
    ]
    reseed._ingest_corpus(client, corpus, "http://h:17000")
    assert posted == ["a", "b"]


def test_check_yield_fails_below_floor():
    client = _GrowthClient([20])  # 20 entity-kind nodes from 350 blocks < 10%
    with pytest.raises(RuntimeError, match="implausibly low"):
        reseed._check_yield(client, n_blocks=350)


def test_check_yield_passes_above_floor(capsys):
    client = _GrowthClient([200])
    reseed._check_yield(client, n_blocks=350)
    assert "200 entities" in capsys.readouterr().out


def test_check_yield_counts_entity_kind_not_literal_entity():
    """The query must exclude structural types, not require type='entity':
    live ingestion mints a type per mention group (#18)."""

    class _QueryCapture:
        def _execute_query(self, query, params):
            assert "NOT n.type IN" in query
            assert "type_definition" in params["structural"]
            assert "edge" in params["structural"]
            return [{"c": 440}]

    reseed._check_yield(_QueryCapture(), n_blocks=350)


class _CoverageSync:
    def __init__(self, have):
        self._have = have

    def get_embedding(self, node_type, uuid):
        return {"embedding": [0.1]} if uuid in self._have else None


class _UuidClient:
    def __init__(self, uuids):
        self._uuids = uuids

    def _execute_query(self, query, params):
        return [{"u": u} for u in self._uuids]


def test_embedding_coverage_passes_at_full_coverage(capsys):
    client = _UuidClient(["a", "b", "c"])
    reseed._check_embedding_coverage(client, _CoverageSync({"a", "b", "c"}))
    assert "3/3" in capsys.readouterr().out


def test_embedding_coverage_fails_below_floor():
    client = _UuidClient([f"u{i}" for i in range(20)])
    sync = _CoverageSync({f"u{i}" for i in range(10)})  # 50% coverage
    with pytest.raises(RuntimeError, match="embedding coverage too low"):
        reseed._check_embedding_coverage(client, sync)
