"""Ablation arms A1-A5 over the frozen-fixture seams (SPEC 7.4, issue #11).

Every arm runs on the SAME frozen ``fixtures/clusters.json`` +
``fixtures/catalog.json``, and every arm snapshot flows through
``run_experiment.build_snapshot`` UNCHANGED (``snapshot["ablation"]`` records
the arm) so ``eval/metrics.py`` consumes all arms identically. Under
``--replay`` the PROMPT-side difference of an arm is embodied by its per-arm
frozen response fixture (what the arm prompt would have returned); the
server-side and cascade-side differences are real code paths.

Arm -> seam mapping
===================

A1 ``naive_llm`` -- single LLM call per whole cluster; NO catalog; name+root
    only. Seam: a direct replayer-driven prompt path that BYPASSES
    ``/type-cluster``: :class:`NaiveLLMClient` duck-types the TestClient
    ``.post()`` surface that ``run_experiment.call_type_cluster`` consumes,
    builds a minimal name+root prompt (no catalog block, no partition, no
    chain, no assign_to instructions), drives the SAME ``FrozenLLMReplayer``
    (keys stay ``cluster_id::repeat``; fixture
    ``fixtures/llm_responses_naive_llm.json``; frozen content is the minimal
    ``{"name": ..., "root": ...}`` shape) and synthesizes ONE whole-cluster
    group with ``assign_to=NEW`` and ``chain=[name, root]``. Cascade seam:
    the T5 ``simulate_cascade`` over a ROOTS-ONLY catalog view with gates off
    (``min_depth=0``, ``enforce_ceiling=False``) -- every group lands
    ``G3_ROOT``; no reuse, no graft, no chain, no gate.

A2 ``no_reuse`` -- full v2 prompt; ``assign_to`` forced NEW. Seam: response
    shaping only. Prompt and registry are IDENTICAL to the full arm, so A2
    shares ``fixtures/llm_responses.json`` byte-for-byte (same prompt =>
    same frozen completions); the arm cascade copies the validated response
    and forces every group to ``assign_to=NEW`` BEFORE the real T5 cascade
    (full catalog, gates on). ``G1_REUSE`` is unreachable; grafting and
    gating stay alive.

A3 ``no_graft`` -- catalog hidden; only roots offered. Seam: registry-stub
    variant. A roots-only ``StubTypeRegistry`` (via
    :func:`roots_only_catalog`) serves ``/type-cluster``, so the prompt
    catalog block offers NO assignable aliases (roots stay GRAFT-ONLY) and
    closed-world coercion forces NEW; the cascade resolves against the SAME
    roots-only view (gates on), so ``resolve_deepest_ancestor`` can only
    land on roots: minted groups place ``G3_ROOT``, never ``G2_GRAFT``.
    Fixture ``fixtures/llm_responses_no_graft.json`` (full chains, all NEW
    -- the catalog was hidden from the namer).

A4 ``no_chain`` -- name + root only, no IS_A chain. Seam: response shaping
    through the REAL ``/type-cluster`` with the FULL registry (reuse stays
    available -- A4 ablates ONLY the chain). Fixture
    ``fixtures/llm_responses_no_chain.json`` carries the degenerate chain
    ``[name, root]`` (no intermediate hypernyms). Cascade seam: real T5 over
    the full catalog with the FLOOR disabled (``min_depth=0``), because the
    floor gates ON chain depth -- the very signal this arm removes (keeping
    it on would residual-ize every group and conflate A4 with gating); the
    CEILING (name-based) stays on. ``chain[1:] == [root]`` means minted
    groups land at roots.

A5 ``no_gate`` -- FLOOR/CEILING disabled. Seam: cascade parameters only.
    Prompt, registry and fixtures are identical to the full arm (shares
    ``fixtures/llm_responses.json``); real T5 cascade with ``min_depth=0``
    (floor off -- covering_depth >= 0 always holds) and
    ``enforce_ceiling=False``. Nothing ever gates to RESIDUAL.
"""

from __future__ import annotations

import asyncio
import json as _jsonlib
from dataclasses import asdict
from typing import Any, Callable, Optional

from harness.cascade import MIN_DEPTH, VALID_TERMINAL_ROOTS, simulate_cascade

# Shared canonicalize() from hermes.canonical (T1), with the same standalone
# shim as harness.cascade so the module stays importable without hermes.
try:  # pragma: no cover - import wiring
    from hermes.canonical import canonicalize  # type: ignore
except Exception:  # pragma: no cover - shim path

    def canonicalize(name: str) -> str:
        return " ".join(name.strip().lower().split())


# Arm -> frozen-response fixture file. Arms whose prompt is byte-identical to
# the full arm (no_reuse, no_gate -- their intervention is response-side or
# cascade-side) SHARE the full fixture: same prompt => same frozen
# completions. Arms whose prompt differs carry their own frozen file.
ARM_RESPONSE_FILES: dict[str, str] = {
    "full": "llm_responses.json",
    "naive_llm": "llm_responses_naive_llm.json",
    "no_reuse": "llm_responses.json",
    "no_graft": "llm_responses_no_graft.json",
    "no_chain": "llm_responses_no_chain.json",
    "no_gate": "llm_responses.json",
}


def _require_arm(ablation: str) -> None:
    if ablation not in ARM_RESPONSE_FILES:
        raise ValueError(f"unknown ablation arm: {ablation!r}")


def responses_filename(ablation: str) -> str:
    """Frozen-response fixture filename for an arm (fail-closed on unknown)."""
    _require_arm(ablation)
    return ARM_RESPONSE_FILES[ablation]


def roots_only_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    """Roots-only VIEW of the frozen catalog (A3 no_graft, A1 naive_llm).

    Same frozen input -- the arm HIDES the non-root catalog, it never edits
    the fixture. ``by_norm`` is rebuilt restricted to the kept uuids.
    """
    by_uuid = {
        type_uuid: rec
        for type_uuid, rec in catalog.get("catalog_by_uuid", {}).items()
        if rec.get("is_root")
    }
    by_norm: dict[str, list[str]] = {}
    for norm, uuids in catalog.get("by_norm", {}).items():
        kept = [u for u in uuids if u in by_uuid]
        if kept:
            by_norm[norm] = kept
    out = dict(catalog)
    out["catalog_by_uuid"] = by_uuid
    out["by_norm"] = by_norm
    return out


def arm_registry_factory(
    ablation: str,
) -> Optional[Callable[[dict[str, Any]], Any]]:
    """Per-arm StubTypeRegistry factory for run() (None => default, full).

    ``no_graft``: roots-only registry, so the /type-cluster catalog block
    offers no assignable aliases. ``naive_llm``: roots-only as well -- the
    endpoint is bypassed, but no arm code path may expose the non-root
    catalog.
    """
    _require_arm(ablation)
    if ablation not in ("no_graft", "naive_llm"):
        return None

    def factory(catalog: dict[str, Any]) -> Any:
        from harness.run_experiment import StubTypeRegistry  # deferred: no cycle

        return StubTypeRegistry.from_catalog(roots_only_catalog(catalog))

    return factory


def arm_client_factory(
    ablation: str, replayer: Any
) -> Optional[Callable[[Any], Any]]:
    """Per-arm endpoint-client factory for run() (None => default TestClient).

    Only A1 ``naive_llm`` replaces the client: its prompt path bypasses
    ``/type-cluster`` entirely (single minimal call per whole cluster).
    """
    _require_arm(ablation)
    if ablation != "naive_llm":
        return None
    return lambda app: NaiveLLMClient(replayer)


class _NaiveResponse:
    """Minimal duck-typed stand-in for the TestClient response surface."""

    def __init__(self, body: dict[str, Any], status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._body

    @property
    def text(self) -> str:
        return _jsonlib.dumps(self._body, ensure_ascii=False)


class NaiveLLMClient:
    """A1 naive-LLM seam: the ``client.post`` surface, without the endpoint.

    One minimal LLM call per WHOLE cluster (no catalog, no partition, no
    chain, no reuse): the frozen content is ``{"name": ..., "root": ...}``
    and the client synthesizes a single whole-cluster group in the exact
    response shape ``run_cluster_repeats`` already consumes, so the snapshot
    flows through ``build_snapshot`` unchanged. Fail-closed: a missing frozen
    key or unparseable frozen content raises -- never a silent default.
    """

    def __init__(self, replayer: Any) -> None:
        self._replayer = replayer

    @staticmethod
    def build_messages(members: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Minimal name+root prompt: members only, no catalog, no rules."""
        lines = []
        for member in members:
            mname = str(member.get("name", ""))
            mid = str(member.get("id", ""))
            lines.append(f"- {mname} (id: {mid})")
        members_block = "\n".join(lines)
        schema = _jsonlib.dumps(
            {"name": "<lowercase singular noun>", "root": "entity|concept|process"}
        )
        system_msg = (
            "You are an ontology naming assistant. Name the ONE natural kind "
            "that covers ALL the cluster members and pick its realm root. "
            "Return ONLY a JSON object: " + schema
        )
        user_msg = (
            "These entities were grouped together (embedding-coarse cluster):\n"
            f"{members_block}\n\nName them."
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def post(self, path: str, json: Any = None) -> _NaiveResponse:
        if path != "/type-cluster":
            # Duck-typing the TestClient surface must stay fail-closed: a
            # mistyped path gets a loud error, not a naive-LLM response.
            raise ValueError(f"NaiveLLMClient only serves /type-cluster, got {path!r}")
        payload: dict[str, Any] = json or {}
        members = list(payload.get("members", []))
        completion = asyncio.run(
            self._replayer(
                messages=self.build_messages(members),
                metadata={"purpose": "naive-llm-ablation"},
            )
        )
        content = (
            completion.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        try:
            data = _jsonlib.loads(content)
        except Exception as exc:
            raise ValueError(
                f"naive_llm replay: unparseable frozen content: {content!r}"
            ) from exc
        if not isinstance(data, dict):
            # The LLM-response parse is an external input boundary in live
            # mode: a non-object completion fails closed with a clear error,
            # mirroring the /type-cluster 502 convention (SPEC 3.4).
            raise ValueError(
                "naive_llm: completion parsed to "
                f"{type(data).__name__}, expected a JSON object"
            )
        raw_name = data.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("naive_llm replay: frozen response has no name")
        name = canonicalize(raw_name)
        raw_root = data.get("root")
        root = canonicalize(raw_root) if isinstance(raw_root, str) else ""
        if root not in VALID_TERMINAL_ROOTS:
            root = "entity"  # deterministic default realm root (mirrors T4)
        confidence = data.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        confidence = max(0.0, min(1.0, float(confidence)))
        member_ids = [
            str(member.get("id")) for member in members if member.get("id")
        ]
        body: dict[str, Any] = {
            "request_id": payload.get("request_id"),
            "groups": [
                {
                    "assign_to": "NEW",  # no catalog: nothing to reuse
                    "name": name,
                    "chain": [name, root],  # name+root only -- no chain
                    "member_ids": member_ids,
                    "confidence": confidence,
                    "description": "",
                    "over_specified": False,  # gate machinery is ablated in A1
                }
            ],
            "residual_ids": [],
            # the single whole-cluster group covers every input id by
            # construction, so the raw partition is trivially valid
            "raw_partition_ok": True,
        }
        return _NaiveResponse(body)


def _records_to_cascade(
    records: list[Any], residual_ids: list[str]
) -> dict[str, Any]:
    """PlacementRecords -> the snapshot-serializable cascade dict (T6 shape)."""
    branches = [
        {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(rec).items()}
        for rec in records
    ]
    return {"branches": branches, "residual_ids": list(residual_ids)}


def simulate_arm_cascade(
    response: dict[str, Any],
    catalog: dict[str, Any],
    *,
    ablation: str,
) -> dict[str, Any]:
    """Arm-shaped T5 cascade (the ``cascade_fn`` seam shape for A1-A5).

    The ``full`` arm stays in ``run_experiment.simulate_cascade_response``;
    this dispatch owns the five ablation arms only. The input response is
    never mutated (groups are shallow-copied before any arm shaping).
    """
    _require_arm(ablation)
    if ablation == "full":
        raise ValueError(
            "simulate_arm_cascade handles A1-A5 only; the full arm runs in "
            "run_experiment.simulate_cascade_response"
        )
    groups = [dict(g) for g in response.get("groups", [])]
    view = catalog
    min_depth = MIN_DEPTH
    enforce_ceiling = True
    if ablation == "no_reuse":
        # A2: reuse off; graft + gates stay on (full catalog).
        for group in groups:
            group["assign_to"] = "NEW"
    elif ablation == "no_graft":
        # A3: only roots are resolvable; gates stay on.
        view = roots_only_catalog(catalog)
    elif ablation == "no_chain":
        # A4: the floor gates ON chain depth -- the ablated signal -- so it
        # is disabled; the (name-based) ceiling stays on; reuse stays on.
        min_depth = 0
    elif ablation == "no_gate":
        # A5: FLOOR off (min_depth=0) + CEILING off.
        min_depth = 0
        enforce_ceiling = False
    else:  # naive_llm
        # A1: no catalog (roots-only view), no gates; groups arrive NEW.
        view = roots_only_catalog(catalog)
        min_depth = 0
        enforce_ceiling = False
    records = simulate_cascade(
        groups,
        view.get("catalog_by_uuid", {}),
        view.get("by_norm", {}),
        min_depth=min_depth,
        enforce_ceiling=enforce_ceiling,
    )
    return _records_to_cascade(records, list(response.get("residual_ids", [])))
