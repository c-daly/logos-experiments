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

(A4 ``no_chain`` RETIRED 2026-06-06 -- the contract has no LLM-supplied
chain to ablate; placement chain is the graph's, derived from the parent.)

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

from harness.cascade import VALID_TERMINAL_ROOTS, simulate_cluster_placement

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
        # 2026-06-06 contract: no catalog shown -> the naive arm can only
        # place under a realm root. {name, parent=root, no outliers}.
        _ = (confidence, member_ids)  # parsed/ignored: placement keys off name
        body: dict[str, Any] = {
            "request_id": payload.get("request_id"),
            "name": name,
            "parent": root,
            "over_specified": False,
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
    cluster_id: str = "cluster",
    member_ids: Optional[list[str]] = None,
    ablation: str,
) -> dict[str, Any]:
    """Arm-shaped placement (the cascade_fn seam for A1/A2/A3/A5).

    2026-06-06 contract: one cluster -> one placement. The full arm runs in
    run_experiment.simulate_cascade_response; this owns the ablations, each a
    parent-choice / catalog-view coercion on {name, parent, residual_ids}.
    The chain arm (A4) is retired -- there is no LLM chain to ablate.

    Coercions (the measurement semantics -- review here):
      A2 no_reuse  : a would-be reuse (parent is None) is coerced to mint
                     `name` under the entity root; real grafts (parent set)
                     are untouched. Isolates reuse's lift vs mint-at-root.
      A3 no_graft  : parent forced to the entity root AND a roots-only catalog
                     view -> every placement lands G3_ROOT (no graft under a
                     non-root, no reuse). Isolates graft/parent-specificity.
      A5 no_gate   : CEILING off (FLOOR is always-legal under the re-parent
                     model, so the only gate left to disable is the ceiling).
      A1 naive_llm : the naive client already emits parent=root; roots-only
                     view + ceiling off -> a pure name+root baseline.
    """
    _require_arm(ablation)
    if ablation == "full":
        raise ValueError(
            "simulate_arm_cascade handles the ablation arms only; the full "
            "arm runs in run_experiment.simulate_cascade_response"
        )
    members = list(member_ids or [])
    name = str(response.get("name", ""))
    parent = response.get("parent")
    residual_ids = list(response.get("residual_ids", []))
    view = catalog
    enforce_ceiling = True

    if ablation == "no_reuse":
        if parent is None:
            parent = "entity"
    elif ablation == "no_graft":
        parent = "entity"
        view = roots_only_catalog(catalog)
    elif ablation == "no_gate":
        enforce_ceiling = False
    elif ablation == "naive_llm":
        view = roots_only_catalog(catalog)
        enforce_ceiling = False

    rec = simulate_cluster_placement(
        cluster_id=cluster_id,
        member_ids=members,
        name=name,
        parent=parent,
        residual_ids=residual_ids,
        catalog_by_uuid=view.get("catalog_by_uuid", {}),
        by_norm=view.get("by_norm", {}),
        enforce_ceiling=enforce_ceiling,
    )
    return _records_to_cascade([rec], list(rec.residual_ids))


def arm_message_transform(
    ablation: str,
) -> Optional[Callable[[list[dict[str, Any]]], list[dict[str, Any]]]]:
    """Per-arm live-prompt transform (None => prompt used as built).

    2026-06-06 contract: no arm rewrites the prompt anymore. The chain arm
    (A4) is retired (no LLM chain to suppress); the remaining arms differ by
    catalog view (no_graft / naive_llm) or response-side parent coercion
    (no_reuse), never by prompt text. Kept as a stable seam so the freeze
    driver can call it uniformly.
    """
    _require_arm(ablation)
    return None
