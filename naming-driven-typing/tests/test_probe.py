"""Tests for the real non-mutation probe (harness/probe.py, issue #13).

Pure pieces only, exercised with fakes: no Neo4j, no Redis, no network. The
live readers (``build_live_readers``) are live-path-only and stay untested
here by design (env-gated, operator-driven).
"""

from __future__ import annotations

import pytest

from harness import probe as pb
from harness.run_experiment import NonMutationViolation, assert_non_mutation


def _readers(count_fn, blob_fn, registry_fn=None):
    return pb.StackReaders(
        type_def_count=count_fn,
        redis_key_bytes=blob_fn,
        hermes_registry_count=registry_fn,
    )


# ---------------------------------------------------------------------------
# snapshot_stack_state
# ---------------------------------------------------------------------------


def test_snapshot_reads_all_three_invariants():
    state = pb.snapshot_stack_state(
        _readers(lambda: 42, lambda: b"payload", lambda: 7)
    )
    assert state["type_def_count"] == 42
    assert state["redis_key_hash"] == pb.hash_key_content(b"payload")
    assert state["hermes_registry_count"] == 7


def test_snapshot_records_absent_redis_key_as_sentinel():
    state = pb.snapshot_stack_state(_readers(lambda: 1, lambda: None))
    assert state["redis_key_hash"] == pb.ABSENT_KEY


def test_snapshot_records_unavailable_registry_as_none():
    # No reader at all -> None.
    state = pb.snapshot_stack_state(_readers(lambda: 1, lambda: b"x"))
    assert state["hermes_registry_count"] is None
    # Reader present but the in-process registry is not installed -> None
    # (covered by the Redis key hash, see the module docstring).
    state = pb.snapshot_stack_state(_readers(lambda: 1, lambda: b"x", lambda: None))
    assert state["hermes_registry_count"] is None


def test_snapshot_wraps_neo4j_reader_failure_fail_closed():
    def boom():
        raise RuntimeError("bolt connection refused")

    with pytest.raises(NonMutationViolation, match="type-def count"):
        pb.snapshot_stack_state(_readers(boom, lambda: b"x"))


def test_snapshot_wraps_redis_reader_failure_fail_closed():
    def boom():
        raise RuntimeError("redis down")

    with pytest.raises(NonMutationViolation, match="Redis key"):
        pb.snapshot_stack_state(_readers(lambda: 1, boom))


def test_snapshot_wraps_registry_reader_failure_fail_closed():
    def boom():
        raise RuntimeError("registry exploded")

    with pytest.raises(NonMutationViolation, match="TypeRegistry"):
        pb.snapshot_stack_state(_readers(lambda: 1, lambda: b"x", boom))


# ---------------------------------------------------------------------------
# hash_key_content
# ---------------------------------------------------------------------------


def test_hash_is_content_sensitive_and_str_bytes_agnostic():
    assert pb.hash_key_content(b"abc") == pb.hash_key_content("abc")
    assert pb.hash_key_content(b"abc") != pb.hash_key_content(b"abd")
    assert pb.hash_key_content(None) == pb.ABSENT_KEY
    # An EMPTY value is not the same as an ABSENT key.
    assert pb.hash_key_content(b"") != pb.ABSENT_KEY


# ---------------------------------------------------------------------------
# make_live_probe + assert_non_mutation end-to-end (fakes)
# ---------------------------------------------------------------------------


class _MutableStack:
    """A fake stack whose state the test mutates between before/after."""

    def __init__(self):
        self.count = 10
        self.blob = b"snapshot-v1"
        self.registry = 10

    def readers(self):
        return _readers(
            lambda: self.count, lambda: self.blob, lambda: self.registry
        )


def test_unchanged_stack_passes_assert():
    stack = _MutableStack()
    probe = pb.make_live_probe(stack.readers())
    assert_non_mutation(probe())  # no raise


def test_type_def_drift_raises_before_persist():
    stack = _MutableStack()
    probe = pb.make_live_probe(stack.readers())
    stack.count += 1  # something minted a type mid-run
    with pytest.raises(NonMutationViolation, match="type-def count"):
        assert_non_mutation(probe())


def test_redis_key_drift_raises():
    stack = _MutableStack()
    probe = pb.make_live_probe(stack.readers())
    stack.blob = b"snapshot-v2"  # the prod snapshot key was overwritten
    with pytest.raises(NonMutationViolation, match="redis"):
        assert_non_mutation(probe())


def test_redis_key_deletion_raises():
    stack = _MutableStack()
    probe = pb.make_live_probe(stack.readers())
    stack.blob = None  # key deleted mid-run
    with pytest.raises(NonMutationViolation, match="redis"):
        assert_non_mutation(probe())


def test_hermes_registry_drift_raises():
    stack = _MutableStack()
    probe = pb.make_live_probe(stack.readers())
    stack.registry += 3
    with pytest.raises(NonMutationViolation, match="TypeRegistry"):
        assert_non_mutation(probe())


def test_registry_becoming_available_midrun_is_drift():
    stack = _MutableStack()
    stack.registry = None
    probe = pb.make_live_probe(stack.readers())
    stack.registry = 5
    with pytest.raises(NonMutationViolation, match="TypeRegistry"):
        assert_non_mutation(probe())


def test_replay_probe_without_registry_keys_stays_dormant():
    # The landed replay probe carries no hermes_registry_count_* keys; the
    # extended assert must not fire on their absence.
    assert_non_mutation(
        {
            "type_def_count_before": 0,
            "type_def_count_after": 0,
            "redis_key_before": "",
            "redis_key_after": "",
            "prod_hermes_targeted": False,
        }
    )


def test_probe_payload_shape():
    before = {"type_def_count": 1, "redis_key_hash": "h", "hermes_registry_count": None}
    after = {"type_def_count": 1, "redis_key_hash": "h", "hermes_registry_count": None}
    payload = pb.probe_payload(before, after)
    assert payload == {
        "type_def_count_before": 1,
        "type_def_count_after": 1,
        "redis_key_before": "h",
        "redis_key_after": "h",
        "hermes_registry_count_before": None,
        "hermes_registry_count_after": None,
        "prod_hermes_targeted": False,
    }


# ---------------------------------------------------------------------------
# require_live_env (the LIVE_RUN idiom, mirroring RESEED_LIVE)
# ---------------------------------------------------------------------------


def test_require_live_env_refuses_without_live_run():
    with pytest.raises(pb.LiveGateError, match="LIVE_RUN=1"):
        pb.require_live_env({})


def test_require_live_env_refuses_wrong_live_run_value():
    with pytest.raises(pb.LiveGateError, match="LIVE_RUN=1"):
        pb.require_live_env({"LIVE_RUN": "true", "HERMES_URL": "http://x:1"})


def test_require_live_env_refuses_without_hermes_url():
    with pytest.raises(pb.LiveGateError, match="HERMES_URL"):
        pb.require_live_env({"LIVE_RUN": "1"})


def test_require_live_env_refuses_blank_hermes_url():
    with pytest.raises(pb.LiveGateError, match="HERMES_URL"):
        pb.require_live_env({"LIVE_RUN": "1", "HERMES_URL": "   "})


def test_require_live_env_returns_explicit_url():
    url = pb.require_live_env(
        {"LIVE_RUN": "1", "HERMES_URL": "http://disposable:17000"}
    )
    assert url == "http://disposable:17000"
