"""Tests for the naming-driven-typing offline harness (T6).

Label-free, replay-by-default, in-process. No Neo4j/Milvus/Redis/network here:
every dependency is injected as a stub and fixtures live in tmp_path.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from harness import run_experiment as rx


def _content(d: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(d)}}]}


def test_replayer_returns_frozen_completion_per_repeat():
    responses = {
        "c1::0": _content({"groups": [{"assign_to": "NEW", "name": "vehicle"}]}),
        "c1::1": _content({"groups": [{"assign_to": "NEW", "name": "conveyance"}]}),
    }
    replayer = rx.FrozenLLMReplayer(responses).for_cluster("c1")
    # repeat 0
    replayer.set_repeat(0)
    out0 = asyncio.run(replayer(messages=[{"role": "user", "content": "x"}]))
    assert (
        json.loads(out0["choices"][0]["message"]["content"])["groups"][0]["name"]
        == "vehicle"
    )
    # repeat 1
    replayer.set_repeat(1)
    out1 = asyncio.run(replayer(messages=[{"role": "user", "content": "x"}]))
    assert (
        json.loads(out1["choices"][0]["message"]["content"])["groups"][0]["name"]
        == "conveyance"
    )


def test_replayer_raises_on_missing_key():
    replayer = rx.FrozenLLMReplayer({}).for_cluster("c1")
    replayer.set_repeat(0)
    with pytest.raises(KeyError):
        asyncio.run(replayer(messages=[{"role": "user", "content": "x"}]))
