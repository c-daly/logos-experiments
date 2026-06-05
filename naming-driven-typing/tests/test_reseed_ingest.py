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
    client = _CountingClient([100, 130, 130, 130, 130])
    assert reseed._settle_graph(client, stable_polls=3, interval=0) == 130


def test_settle_graph_fails_loudly_at_cap(monkeypatch):
    monkeypatch.setattr(reseed.time, "sleep", lambda s: None)
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


def test_ingest_corpus_polls_for_growth_between_blocks(monkeypatch):
    posted = []
    monkeypatch.setattr(reseed, "_ingest", lambda t, d, u: posted.append(t))
    monkeypatch.setattr(reseed.time, "sleep", lambda s: None)
    # baseline 10; block1 grows on second poll (10 -> 15); block2 grows
    # immediately (15 -> 22).
    client = _GrowthClient([10, 10, 15, 22])
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
