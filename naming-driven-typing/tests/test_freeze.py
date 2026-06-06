"""Tests for the gated K-sample freeze (harness/freeze.py, issue #13).

Everything runs with a FAKE transport: no network, no spend. The live
execution path (deployed hermes gateway) is operator-driven and env-gated;
these tests pin the pure mechanics: capture shape, sampling pins, per-arm
prompt variants, fail-closed behavior, cost estimate and the CLI gates.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from harness import freeze as fz
from harness.ablations import arm_message_transform
from harness.fixtures_io import (
    LLMResponsesFixtureError,
    freeze_llm_responses,
    validate_llm_responses,
)

MODEL_SNAPSHOT = "gpt-4.1-2026-04-14"


def _completion(content: str) -> dict:
    return {
        "id": "resp-1",
        "provider": "openai",
        "model": MODEL_SNAPSHOT,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "choices": [{"message": {"content": content}}],
    }


def _v2_content() -> str:
    # 2026-06-06 contract: name the cluster + (optional) parent + outliers.
    return json.dumps({"name": "mammal", "parent": "entity", "outliers": []})


def _fake_send(recorded_systems: list[str]):
    """Arm-aware fake gateway. Naive prompts -> {name, root}; v2 prompts ->
    {name, parent, outliers} (the namer never sees ids; outliers are names)."""

    def send(messages, *, temperature, max_tokens, metadata):
        assert temperature == 0.0  # the pin must reach the wire
        system = messages[0]["content"]
        recorded_systems.append(system)
        if "naming assistant" in system:
            content = json.dumps({"name": "mammal", "root": "entity"})
        else:
            content = _v2_content()
        return _completion(content)

    return send


def _clusters() -> list[dict]:
    return [
        {
            "cluster_id": "c1",
            "current_name": "entity",
            "members": [
                {"id": "u1", "name": "cheetah"},
                {"id": "u2", "name": "narwhal"},
            ],
            "sample_coverage": 1.0,
        },
        {
            "cluster_id": "c2",
            "current_name": "entity",
            "members": [
                {"id": "u3", "name": "sloop"},
                {"id": "u4", "name": "ketch"},
            ],
            "sample_coverage": 1.0,
        },
    ]


def _catalog() -> dict:
    def rec(uuid, name, chain, is_root=False):
        return {
            "uuid": uuid,
            "name": name,
            "norm_name": name,
            "chain": chain,
            "is_root": is_root,
        }

    return {
        "catalog_by_uuid": {
            "u-entity": rec("u-entity", "entity", ["entity"], True),
            "u-concept": rec("u-concept", "concept", ["concept"], True),
            "u-process": rec("u-process", "process", ["process"], True),
            "u-vehicle": rec("u-vehicle", "vehicle", ["entity", "vehicle"]),
            "u-vessel": rec("u-vessel", "vehicle", ["entity", "vehicle"]),
        },
        "by_norm": {"vehicle": ["u-vehicle", "u-vessel"]},
        "roots_present_in_live_catalog": True,
    }


# ---------------------------------------------------------------------------
# CapturingTransport
# ---------------------------------------------------------------------------


def _drive(transport, cluster_id="c1", repeat=0):
    transport.for_cluster(cluster_id)
    transport.set_repeat(repeat)
    return asyncio.run(
        transport(
            messages=[
                {"role": "system", "content": "You are an ontology typing assistant."},
                {"role": "user", "content": "- cheetah (id: u1)\n\nType them."},
            ],
            temperature=0.0,
            max_tokens=512,
        )
    )


def test_transport_captures_exact_replayer_shape():
    systems: list[str] = []
    transport = fz.CapturingTransport(_fake_send(systems))
    returned = _drive(transport)
    assert set(transport.captured) == {"c1::0"}
    frozen = transport.captured["c1::0"]
    assert set(frozen) == {"choices"}
    assert set(frozen["choices"][0]) == {"message"}
    assert set(frozen["choices"][0]["message"]) == {"content"}
    # The endpoint sees exactly what --replay will see for this key.
    assert returned == frozen
    assert transport.model_ids == {MODEL_SNAPSHOT}
    assert transport.usage_totals["total_tokens"] == 15


def test_transport_rejects_duplicate_key():
    systems: list[str] = []
    transport = fz.CapturingTransport(_fake_send(systems))
    _drive(transport)
    with pytest.raises(fz.FreezeError, match="duplicate"):
        _drive(transport)


def test_transport_rejects_unbound_cluster():
    transport = fz.CapturingTransport(_fake_send([]))
    with pytest.raises(fz.FreezeError, match="no active cluster"):
        asyncio.run(transport(messages=[{"role": "user", "content": "x"}]))


def test_transport_enforces_pinned_temperature():
    transport = fz.CapturingTransport(_fake_send([]))
    transport.for_cluster("c1")
    with pytest.raises(fz.FreezeError, match="temperature"):
        asyncio.run(
            transport(
                messages=[{"role": "user", "content": "x"}], temperature=0.7
            )
        )


def test_transport_enforces_default_top_p():
    transport = fz.CapturingTransport(_fake_send([]))
    transport.for_cluster("c1")
    with pytest.raises(fz.FreezeError, match="top_p"):
        asyncio.run(
            transport(
                messages=[{"role": "user", "content": "x"}],
                temperature=0.0,
                top_p=0.9,
            )
        )


def test_transport_fails_closed_on_missing_content():
    def bad_send(messages, *, temperature, max_tokens, metadata):
        return {"model": MODEL_SNAPSHOT, "choices": []}

    transport = fz.CapturingTransport(bad_send)
    transport.for_cluster("c1")
    transport.set_repeat(0)
    with pytest.raises(fz.FreezeError, match="choices"):
        asyncio.run(
            transport(
                messages=[{"role": "user", "content": "x"}], temperature=0.0
            )
        )
    assert transport.captured == {}


def test_transport_applies_message_transform():
    systems: list[str] = []
    tag = " [TRANSFORMED]"

    def add_tag(messages):
        out = [dict(msg) for msg in messages]
        out[0] = {**out[0], "content": out[0]["content"] + tag}
        return out

    transport = fz.CapturingTransport(_fake_send(systems), message_transform=add_tag)
    _drive(transport)
    assert len(systems) == 1
    assert systems[0].endswith(tag)


# ---------------------------------------------------------------------------
# arm_message_transform (ablations)
# ---------------------------------------------------------------------------


def test_no_arm_rewrites_the_prompt():
    for arm in ("full", "naive_llm", "no_reuse", "no_graft", "no_gate"):
        assert arm_message_transform(arm) is None


def test_transform_rejects_unknown_arm():
    with pytest.raises(ValueError, match="unknown ablation arm"):
        arm_message_transform("bogus")


# ---------------------------------------------------------------------------
# frozen llm_responses writer (fixtures_io)
# ---------------------------------------------------------------------------


def _frozen(content="x"):
    return {"choices": [{"message": {"content": content}}]}


def test_freeze_llm_responses_roundtrip_and_determinism(tmp_path):
    path = tmp_path / "llm_responses.json"
    responses = {"c1::1": _frozen("b"), "c1::0": _frozen("a")}
    freeze_llm_responses(responses, path)
    first = path.read_bytes()
    # Re-freeze with a different insertion order -> byte-identical.
    freeze_llm_responses({"c1::0": _frozen("a"), "c1::1": _frozen("b")}, path)
    assert path.read_bytes() == first
    assert json.loads(first.decode("utf-8")) == responses


@pytest.mark.parametrize(
    "bad",
    [
        {"no-separator": _frozen()},
        {"::0": _frozen()},
        {"c1::x": _frozen()},
        {"c1::0": {"choices": []}},
        {"c1::0": {"choices": [{"message": {"content": ""}}]}},
        {"c1::0": {"choices": [{"message": {"content": "x"}, "extra": 1}]}},
        {"c1::0": {"choices": [{"message": {"content": "x"}}], "model": "m"}},
        "not-a-dict",
    ],
)
def test_validate_llm_responses_rejects_non_replayer_shapes(bad):
    with pytest.raises(LLMResponsesFixtureError):
        validate_llm_responses(bad)


def test_checked_in_fixtures_conform_to_replayer_shape():
    from pathlib import Path

    fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"
    for name in (
        "llm_responses.json",
        "llm_responses_naive_llm.json",
        "llm_responses_no_graft.json",
    ):
        validate_llm_responses(
            json.loads((fixtures_dir / name).read_text(encoding="utf-8"))
        )


# ---------------------------------------------------------------------------
# estimate_freeze_cost / format_cost_estimate
# ---------------------------------------------------------------------------


def test_estimate_counts_clusters_x_repeats_x_arms():
    est = fz.estimate_freeze_cost(_clusters(), _catalog(), repeats=2)
    assert est["n_clusters"] == 2
    assert est["repeats"] == 2
    assert est["arms"] == list(fz.FREEZE_ARMS)
    assert est["calls"] == 2 * 2 * len(fz.FREEZE_ARMS)
    assert est["est_input_tokens"] > 0
    assert est["est_output_tokens_cap"] > 0
    # The catalog block + bigger base make the full arm dearer than naive.
    assert (
        est["per_arm"]["full"]["est_input_tokens"]
        > est["per_arm"]["naive_llm"]["est_input_tokens"]
    )


def test_format_cost_estimate_names_the_flag():
    est = fz.estimate_freeze_cost(_clusters(), _catalog(), repeats=5)
    text = fz.format_cost_estimate(est)
    assert "COST ESTIMATE" in text
    assert "--yes-i-will-pay" in text
    assert str(est["calls"]) in text


# ---------------------------------------------------------------------------
# run_freeze end-to-end with the fake transport
# ---------------------------------------------------------------------------


def test_run_freeze_writes_per_arm_fixtures_and_meta(tmp_path):
    systems: list[str] = []
    meta = fz.run_freeze(
        clusters=_clusters(),
        catalog=_catalog(),
        send=_fake_send(systems),
        fixtures_dir=tmp_path,
        repeats=2,
        hermes_gateway="http://disposable:17000",
        frozen_at="20260605T000000Z",
    )
    expected_keys = {"c1::0", "c1::1", "c2::0", "c2::1"}
    for name in (
        "llm_responses.json",
        "llm_responses_naive_llm.json",
        "llm_responses_no_graft.json",
    ):
        frozen = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert set(frozen) == expected_keys, name
        validate_llm_responses(frozen)
    # The naive arm froze the minimal name+root shape, not the groups shape.
    naive = json.loads(
        (tmp_path / "llm_responses_naive_llm.json").read_text(encoding="utf-8")
    )
    naive_content = json.loads(naive["c1::0"]["choices"][0]["message"]["content"])
    assert set(naive_content) == {"name", "root"}  # naive: name+root only
    # Meta records the model snapshot id and the pinned sampling.
    meta_on_disk = json.loads(
        (tmp_path / fz.FREEZE_META_FILENAME).read_text(encoding="utf-8")
    )
    assert meta_on_disk == meta
    assert meta["model_snapshot_ids"] == [MODEL_SNAPSHOT]
    assert meta["pinned_sampling"]["temperature"] == 0.0
    assert meta["calls"] == 12
    assert meta["repeats"] == 2
    assert meta["arm_fixtures"]["no_reuse"] == "llm_responses.json"
    assert meta["arm_fixtures"]["no_gate"] == "llm_responses.json"
    assert meta["usage_totals"]["total_tokens"] == 12 * 15
    assert meta["hermes_gateway"] == "http://disposable:17000"


def test_run_freeze_replays_byte_stable(tmp_path):
    """A frozen fixture must drive the replayer to the very same content."""
    from harness.run_experiment import FrozenLLMReplayer

    fz.run_freeze(
        clusters=_clusters(),
        catalog=_catalog(),
        send=_fake_send([]),
        fixtures_dir=tmp_path,
        repeats=1,
        arms=("full",),
    )
    frozen = json.loads((tmp_path / "llm_responses.json").read_text("utf-8"))
    replayer = FrozenLLMReplayer(frozen).for_cluster("c1")
    replayer.set_repeat(0)
    replayed = asyncio.run(replayer(messages=[{"role": "user", "content": "x"}]))
    assert replayed == frozen["c1::0"]


def test_run_freeze_parks_failed_cluster_and_continues(tmp_path):
    """A per-cluster failure parks that cluster (deterministic temp-0 drop)
    and the arm completes; the parked cluster is recorded in freeze_meta and
    its truncated content is NOT written to the fixtures (#18 park-and-
    continue; one bad cluster must not torch a paid run)."""
    calls = {"n": 0}

    def flaky_send(messages, *, temperature, max_tokens, metadata):
        calls["n"] += 1
        # c1 fully succeeds (calls 1,2 over 2 repeats); every c2 call raises.
        if calls["n"] > 2:
            raise RuntimeError("gateway 502")
        return _completion(_v2_content())

    meta = fz.run_freeze(
        clusters=_clusters(),
        catalog=_catalog(),
        send=flaky_send,
        fixtures_dir=tmp_path,
        repeats=2,
        arms=("full",),
    )
    # Arm completed: fixtures + meta written, no FreezeError raised.
    frozen = json.loads((tmp_path / "llm_responses.json").read_text("utf-8"))
    assert set(frozen) == {"c1::0", "c1::1"}  # c1 captured, c2 discarded
    assert "c2::0" not in frozen
    assert meta["parked_clusters"] == {"full": ["c2"]}
    assert (tmp_path / fz.FREEZE_META_FILENAME).exists()


def test_run_freeze_restores_hermes_seams(tmp_path):
    import hermes.main as m

    prev_registry = m._type_registry
    prev_generate = m.generate_completion
    fz.run_freeze(
        clusters=_clusters(),
        catalog=_catalog(),
        send=_fake_send([]),
        fixtures_dir=tmp_path,
        repeats=1,
        arms=("full", "naive_llm"),
    )
    assert m._type_registry is prev_registry
    assert m.generate_completion is prev_generate


# ---------------------------------------------------------------------------
# CLI gates: estimate + acknowledgement + env, in that order, fail-closed
# ---------------------------------------------------------------------------


@pytest.fixture
def _no_live_env(monkeypatch):
    monkeypatch.delenv("LIVE_RUN", raising=False)
    monkeypatch.delenv("HERMES_URL", raising=False)


def test_main_freeze_refuses_without_pay_flag(_no_live_env, capsys):
    from harness import run_experiment as rx

    assert rx.main(["--freeze"]) == 2
    out = capsys.readouterr().out
    assert "COST ESTIMATE" in out
    assert "refusing to spend" in out


def test_main_freeze_refuses_without_env_gates(_no_live_env, capsys):
    from harness import run_experiment as rx

    assert rx.main(["--freeze", "--yes-i-will-pay"]) == 2
    captured = capsys.readouterr()
    assert "COST ESTIMATE" in captured.out
    assert "LIVE_RUN=1" in captured.err


def test_main_live_refuses_without_pay_flag(_no_live_env, capsys):
    from harness import run_experiment as rx

    assert rx.main(["--live"]) == 2
    out = capsys.readouterr().out
    assert "COST ESTIMATE" in out
    assert "--yes-i-will-pay" in out


def test_main_live_refuses_without_env_gates(_no_live_env, capsys):
    from harness import run_experiment as rx

    assert rx.main(["--live", "--yes-i-will-pay"]) == 2
    captured = capsys.readouterr()
    assert "LIVE_RUN=1" in captured.err


def test_main_live_refuses_without_hermes_url(monkeypatch, capsys):
    from harness import run_experiment as rx

    monkeypatch.setenv("LIVE_RUN", "1")
    monkeypatch.delenv("HERMES_URL", raising=False)
    assert rx.main(["--live", "--yes-i-will-pay"]) == 2
    assert "HERMES_URL" in capsys.readouterr().err


def test_main_live_missing_password_exits_cleanly(monkeypatch, capsys):
    """LiveGateError from build_live_readers must produce the clean gating
    message and exit 2, not a raw traceback (PR #16 review)."""
    from harness import run_experiment as rx

    monkeypatch.setenv("LIVE_RUN", "1")
    monkeypatch.setenv("HERMES_URL", "http://127.0.0.1:1")
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    assert rx.main(["--live", "--yes-i-will-pay"]) == 2
    err = capsys.readouterr().err
    assert "NEO4J_PASSWORD" in err
    assert "[harness]" in err


def test_main_freeze_error_exits_cleanly(monkeypatch, capsys):
    """A mid-freeze FreezeError must print the clean operator message and
    exit 1, not escape as a traceback (PR #16 review)."""
    from harness import run_experiment as rx

    monkeypatch.setenv("LIVE_RUN", "1")
    monkeypatch.setenv("HERMES_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(fz, "run_freeze", lambda **kw: (_ for _ in ()).throw(
        fz.FreezeError("freeze aborted in arm 'full': gateway 502")
    ))
    assert rx.main(["--freeze", "--yes-i-will-pay"]) == 1
    err = capsys.readouterr().err
    assert "[freeze]" in err and "aborted in arm" in err


def test_main_live_nonmutation_violation_exits_cleanly(monkeypatch, capsys):
    """A probe breach must print the clean operator message and exit 1
    (PR #16 review)."""
    from harness import run_experiment as rx

    monkeypatch.setenv("LIVE_RUN", "1")
    monkeypatch.setenv("HERMES_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    # _run_live imports these from harness.probe at call time; patch the
    # source module so the lazy import resolves to the fakes.
    import harness.probe as probe_mod

    monkeypatch.setattr(probe_mod, "build_live_readers", lambda: object())
    monkeypatch.setattr(probe_mod, "make_live_probe", lambda readers: lambda: {})
    monkeypatch.setattr(rx, "run", lambda **kw: (_ for _ in ()).throw(
        rx.NonMutationViolation("prod Redis key hash changed during the pass")
    ))
    assert rx.main(["--live", "--yes-i-will-pay"]) == 1
    err = capsys.readouterr().err
    assert "[harness] live run aborted" in err and "Redis key hash" in err


def test_main_rejects_freeze_with_replay(_no_live_env):
    from harness import run_experiment as rx

    with pytest.raises(SystemExit):
        rx.main(["--replay", "--freeze"])


def test_run_freeze_partial_write_failure_preserves_freeze_error(
    tmp_path, monkeypatch
):
    """If the final per-arm write fails AND the partial dump also fails, the
    FreezeError must still surface with an accurate note instead of being
    swallowed (PR #16 review; survives the #18 park-and-continue refactor --
    here every cluster SUCCEEDS so the captured set is non-empty and the
    failure is purely on the write path)."""

    def all_ok_send(messages, *, temperature, max_tokens, metadata):
        return _completion(_v2_content())

    def _boom(captured, path):
        raise OSError("disk full")

    monkeypatch.setattr(fz, "freeze_llm_responses", _boom)
    with pytest.raises(
        fz.FreezeError, match=r"aborted in arm .*also failed: disk full"
    ):
        fz.run_freeze(
            clusters=_clusters(),
            catalog=_catalog(),
            send=all_ok_send,
            fixtures_dir=tmp_path,
            repeats=2,
            arms=("full",),
        )
