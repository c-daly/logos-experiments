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


def test_build_snapshot_shape_and_coverage_flag():
    cluster_results = [
        {
            "cluster_id": "cX",
            "current_name": "vehicle",
            "member_count": 2,
            "repeats": [
                {
                    "repeat": 0,
                    "request_id": "cX::0",
                    "response": {"groups": []},
                    "raw_partition_ok": True,
                    "sample_coverage": 1.0,
                    "cascade": {},
                },
                {
                    "repeat": 1,
                    "request_id": "cX::1",
                    "response": {"groups": []},
                    "raw_partition_ok": True,
                    "sample_coverage": 0.5,
                    "cascade": {},
                },
            ],
        }
    ]
    snap = rx.build_snapshot(
        cluster_results=cluster_results,
        catalog={
            "catalog_by_uuid": {"u1": {"name": "vehicle"}},
            "by_norm": {"vehicle": ["u1"]},
        },
        ablation="full",
        model="gpt-4.1",
        repeats=2,
        catalog_mode="in_process",
        roots_present=True,
        run_ts="20260605T000000Z",
    )
    assert snap["experiment"] == "naming-driven-typing"
    assert snap["ablation"] == "full"
    assert snap["model"] == "gpt-4.1"
    assert snap["repeats"] == 2
    assert snap["catalog_mode"] == "in_process"
    assert snap["roots_present"] is True
    assert snap["label_free"] is True
    assert snap["catalog_size"] == 1
    assert snap["clusters"][0]["cluster_id"] == "cX"
    # the 0.5-coverage repeat must be surfaced so partition==0 isn't misread
    assert "cX" in snap["coverage_flags"]


def test_assert_non_mutation_passes_when_unchanged():
    probe = {
        "type_def_count_before": 42,
        "type_def_count_after": 42,
        "redis_key_before": "abc",
        "redis_key_after": "abc",
        "prod_hermes_targeted": False,
    }
    rx.assert_non_mutation(probe)  # no raise


def test_assert_non_mutation_raises_on_type_def_drift():
    probe = {
        "type_def_count_before": 42,
        "type_def_count_after": 43,
        "redis_key_before": "abc",
        "redis_key_after": "abc",
        "prod_hermes_targeted": False,
    }
    with pytest.raises(rx.NonMutationViolation, match="type-def count"):
        rx.assert_non_mutation(probe)


def test_assert_non_mutation_raises_on_redis_drift():
    probe = {
        "type_def_count_before": 42,
        "type_def_count_after": 42,
        "redis_key_before": "abc",
        "redis_key_after": "MUTATED",
        "prod_hermes_targeted": False,
    }
    with pytest.raises(rx.NonMutationViolation, match="redis"):
        rx.assert_non_mutation(probe)


def test_assert_non_mutation_raises_when_prod_hermes_targeted():
    probe = {
        "type_def_count_before": 42,
        "type_def_count_after": 42,
        "redis_key_before": "abc",
        "redis_key_after": "abc",
        "prod_hermes_targeted": True,
    }
    with pytest.raises(rx.NonMutationViolation, match="prod Hermes"):
        rx.assert_non_mutation(probe)
