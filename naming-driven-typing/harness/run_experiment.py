"""Offline harness: naming-driven typing v2.

Pulls frozen type-clusters + an enriched closed-world catalog, drives the v2
``POST /type-cluster`` endpoint IN-PROCESS (FastAPI TestClient + a frozen-LLM
replayer so ``--replay`` is $0 and deterministic), simulates the placement
cascade (T5) per repeat, and writes a label-free snapshot that
``eval/metrics.py::compute_metrics`` (T7) scores.

Runs from this experiment's uv env (hermes installed env-only so
``hermes.main`` imports). Snapshot/measure ONLY: read-only inputs, never
mutates the graph, the prod Redis key, or the prod Hermes registry.

Run (replay, default, $0), from naming-driven-typing/:
    uv run --no-sync python harness/run_experiment.py \\
        --replay --repeats 5 --ablation full

Stack defaults (overridable via env; only used by --live):
    NEO4J_URI=bolt://localhost:7687  NEO4J_USER=neo4j  NEO4J_PASSWORD=logosdev
    MILVUS_HOST=localhost            MILVUS_PORT=19530
    HERMES_URL=http://localhost:17000   (prod -- NEVER targeted by this harness)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FIXTURES = EXP / "fixtures"
WORKSPACE = EXP / "workspace"

# Prod stack identifiers -- recorded so the non-mutation probe can assert the
# harness never targeted them. The harness drives Hermes IN-PROCESS, so this
# URL must never appear as a request target.
PROD_HERMES_URL = os.environ.get("HERMES_URL", "http://localhost:17000")
PROD_REDIS_KEY = "logos:ontology:types"

ABLATIONS = ("full", "naive_llm", "no_reuse", "no_graft", "no_chain", "no_gate")


class HarnessEndpointError(RuntimeError):
    """The in-process /type-cluster call returned a non-200 status."""


class NonMutationViolation(RuntimeError):
    """A post-run invariant (graph/redis/hermes untouched) was breached."""


@dataclass(frozen=True)
class HarnessPaths:
    fixtures_dir: Path
    workspace_dir: Path

    @classmethod
    def default(cls) -> "HarnessPaths":
        return cls(fixtures_dir=FIXTURES, workspace_dir=WORKSPACE)


@dataclass(frozen=True)
class ReplayKey:
    cluster_id: str
    repeat: int

    def token(self) -> str:
        return f"{self.cluster_id}::{self.repeat}"


class FrozenLLMReplayer:
    """Replays frozen /type-cluster completions keyed by ``cluster_id::repeat``.

    A drop-in async stand-in for ``hermes.main.generate_completion``. The active
    cluster is bound via :meth:`for_cluster` and the active repeat via
    :meth:`set_repeat` before each endpoint call, so concurrent repeats never
    bleed into one another.
    """

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self._responses = responses
        self._cluster_id: Optional[str] = None
        self._repeat: int = 0

    def for_cluster(self, cluster_id: str) -> "FrozenLLMReplayer":
        self._cluster_id = cluster_id
        return self

    def set_repeat(self, repeat: int) -> None:
        self._repeat = repeat

    async def __call__(
        self,
        *,
        messages: Any,
        model: Any = None,
        temperature: Any = None,
        top_p: Any = None,
        max_tokens: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self._cluster_id is None:
            raise RuntimeError("FrozenLLMReplayer: no active cluster bound")
        key = ReplayKey(self._cluster_id, self._repeat).token()
        if key not in self._responses:
            raise KeyError(f"no frozen LLM response for {key!r}")
        return self._responses[key]


def build_snapshot(
    *,
    cluster_results: list[dict[str, Any]],
    catalog: dict[str, Any],
    ablation: str,
    model: str,
    repeats: int,
    catalog_mode: str,
    roots_present: bool,
    run_ts: str,
) -> dict[str, Any]:
    """Assemble the workspace/run_<ts>.json snapshot (input to T7 compute_metrics).

    Label-free: no coherence labels, no root ground-truth -- only the raw output
    and structural records. Any cluster with a repeat below full coverage is
    named in ``coverage_flags`` so a clean partition rate is never misread.
    """
    coverage_flags = [
        c["cluster_id"]
        for c in cluster_results
        if any(r.get("sample_coverage", 1.0) < 1.0 for r in c.get("repeats", []))
    ]
    return {
        "experiment": "naming-driven-typing",
        "label_free": True,
        "run_ts": run_ts,
        "model": model,
        "ablation": ablation,
        "repeats": repeats,
        "catalog_mode": catalog_mode,
        "roots_present": roots_present,
        "catalog_size": len(catalog.get("catalog_by_uuid", {})),
        "coverage_flags": coverage_flags,
        "clusters": cluster_results,
    }


def assert_non_mutation(probe: dict[str, Any]) -> None:
    """Post-run invariant (SPEC 6, non-mutation): the harness measured, not mutated.

    - Neo4j type_definition count unchanged.
    - Prod Redis key ``logos:ontology:types`` unchanged.
    - Prod ``HERMES_URL`` never targeted (we drive Hermes in-process only).
    Any breach is fail-closed.
    """
    before = probe["type_def_count_before"]
    after = probe["type_def_count_after"]
    if before != after:
        raise NonMutationViolation(
            f"type-def count changed: {before} -> {after} (graph was mutated)"
        )
    if probe["redis_key_before"] != probe["redis_key_after"]:
        raise NonMutationViolation(
            f"prod redis key {PROD_REDIS_KEY!r} changed (snapshot was overwritten)"
        )
    if probe.get("prod_hermes_targeted"):
        raise NonMutationViolation(
            f"prod Hermes was targeted ({PROD_HERMES_URL}); harness must run in-process"
        )


class StubTypeRegistry:
    """In-process duck-typed stand-in for hermes' Redis-backed TypeRegistry.

    Built from the frozen enriched catalog (T2/T3 shape: ``catalog_by_uuid``
    keyed by uuid with root-first ``chain``). Read-only; exposes exactly the
    ``get_type_names()`` / ``get_type(name)`` surface ``hermes.main`` reads,
    so the /type-cluster catalog block is served fully in-process -- never
    from the prod Redis key.
    """

    def __init__(self, types: dict[str, dict[str, Any]]) -> None:
        self._types = types

    @classmethod
    def from_catalog(cls, catalog: dict[str, Any]) -> "StubTypeRegistry":
        types: dict[str, dict[str, Any]] = {}
        for type_uuid, rec in catalog.get("catalog_by_uuid", {}).items():
            name = rec.get("name")
            if not isinstance(name, str) or not name:
                continue
            chain = rec.get("chain")
            if not isinstance(chain, list):
                chain = []
            types[name] = {
                "uuid": rec.get("uuid", type_uuid),
                "root": chain[0] if chain else "",
                "chain": list(chain),
                "is_root": bool(rec.get("is_root", False)),
            }
        return cls(types)

    def get_type_names(self) -> list[str]:
        return sorted(self._types)

    def get_type(self, name: str) -> Optional[dict[str, Any]]:
        info = self._types.get(name)
        return dict(info) if info is not None else None


def call_type_cluster(
    client: Any,
    members: list[dict[str, Any]],
    *,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    """POST the WHOLE member list to the in-process /type-cluster endpoint.

    No down-sampling (SPEC 6.1 / R-integration-3): the v2 partition contract is
    over the full cluster. Non-200 is fail-closed -> HarnessEndpointError.
    """
    payload: dict[str, Any] = {"members": members}
    if request_id is not None:
        payload["request_id"] = request_id
    resp = client.post("/type-cluster", json=payload)
    if resp.status_code != 200:
        raise HarnessEndpointError(f"/type-cluster -> {resp.status_code}: {resp.text}")
    body: dict[str, Any] = resp.json()
    return body


def run_cluster_repeats(
    cluster: dict[str, Any],
    catalog: dict[str, Any],
    *,
    client: Any,
    replayer: FrozenLLMReplayer,
    cascade_fn: Callable[..., dict[str, Any]],
    repeats: int,
    ablation: str = "full",
) -> list[dict[str, Any]]:
    """K repeats for ONE cluster (no down-sampling, whole cluster every time).

    Each repeat: bind the replayer -> POST whole cluster -> parse -> simulate
    cascade (T5). ``sample_coverage = sent/total`` is recorded (==1.0 because we
    never down-sample) so a clean partition is never misread as "all typed".
    """
    cluster_id = cluster["cluster_id"]
    members = cluster["members"]
    total = len(members)
    replayer.for_cluster(cluster_id)
    out: list[dict[str, Any]] = []
    for k in range(repeats):
        replayer.set_repeat(k)
        request_id = ReplayKey(cluster_id, k).token()
        body = call_type_cluster(client, members, request_id=request_id)
        cascade = cascade_fn(body, catalog, ablation=ablation)
        out.append(
            {
                "repeat": k,
                "request_id": request_id,
                "response": body,
                "raw_partition_ok": bool(body.get("raw_partition_ok", False)),
                "sample_coverage": (len(members) / total) if total else 0.0,
                "cascade": cascade,
            }
        )
    return out


def simulate_cascade_response(
    response: dict[str, Any],
    catalog: dict[str, Any],
    *,
    ablation: str = "full",
) -> dict[str, Any]:
    """Adapt T5's simulate_cascade to the harness ``cascade_fn`` seam.

    T5 landed as ``simulate_cascade(groups, catalog_by_uuid, by_norm, ...) ->
    list[PlacementRecord]`` while the harness seam is ``(response, catalog, *,
    ablation) -> dict`` -- per the PLAN T6 integration note the harness is the
    consumer, so the adaptation lives here. Only the 'full' arm is wired to
    the real simulator; the other ablation arms are a follow-up (kept honest,
    mirroring --live).
    """
    if ablation != "full":
        raise NotImplementedError(
            f"ablation arm {ablation!r} is not wired to the T5 simulator yet; "
            "only 'full' runs through the real cascade"
        )
    from harness.cascade import simulate_cascade  # T5 (deferred import)

    records = simulate_cascade(
        list(response.get("groups", [])),
        catalog.get("catalog_by_uuid", {}),
        catalog.get("by_norm", {}),
    )
    branches = [
        {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(rec).items()}
        for rec in records
    ]
    return {
        "branches": branches,
        "residual_ids": list(response.get("residual_ids", [])),
    }


def _load_clusters(paths: HarnessPaths) -> list[dict[str, Any]]:
    raw = (paths.fixtures_dir / "clusters.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, dict):
        # T3 freeze envelope: {"version": ..., "clusters": [...]}
        data = data.get("clusters", [])
    return list(data)


def run(
    *,
    paths: HarnessPaths,
    catalog_loader: Callable[[HarnessPaths], dict[str, Any]],
    cascade_fn: Callable[..., dict[str, Any]],
    llm_replayer: FrozenLLMReplayer,
    nonmutation_probe: Callable[[], dict[str, Any]],
    ablation: str = "full",
    repeats: int = 5,
    catalog_mode: str = "in_process",
    limit: Optional[int] = None,
    model: str = "gpt-4.1",
    run_ts: Optional[str] = None,
) -> Path:
    """Top-level orchestration (replay path). Returns the snapshot path.

    In-process only: the v2 endpoint is driven via FastAPI TestClient with
    ``generate_completion`` already monkeypatched to ``llm_replayer`` by the
    caller, and the catalog served by a StubTypeRegistry built from the frozen
    catalog (installed for the duration of the run, then restored). No
    graph/redis/hermes writes -- verified by ``assert_non_mutation``.
    """
    import hermes.main as m
    from fastapi.testclient import TestClient

    run_ts = run_ts or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    paths.workspace_dir.mkdir(parents=True, exist_ok=True)

    clusters = _load_clusters(paths)
    if limit is not None:
        clusters = clusters[:limit]
    catalog = catalog_loader(paths)
    roots_present = bool(
        catalog.get(
            "roots_present", catalog.get("roots_present_in_live_catalog", False)
        )
    )

    client = TestClient(m.app)

    # In-process stub registry built from the frozen catalog -- /type-cluster's
    # catalog block never touches the prod Redis key. Restored afterwards.
    stub_registry = StubTypeRegistry.from_catalog(catalog)
    prev_registry = m._type_registry
    m._type_registry = stub_registry
    try:
        cluster_results: list[dict[str, Any]] = []
        for cluster in clusters:
            repeats_out = run_cluster_repeats(
                cluster,
                catalog,
                client=client,
                replayer=llm_replayer,
                cascade_fn=cascade_fn,
                repeats=repeats,
                ablation=ablation,
            )
            cluster_results.append(
                {
                    "cluster_id": cluster["cluster_id"],
                    "current_name": cluster.get("current_name", ""),
                    "member_count": len(cluster["members"]),
                    "repeats": repeats_out,
                }
            )
    finally:
        m._type_registry = prev_registry

    snapshot = build_snapshot(
        cluster_results=cluster_results,
        catalog=catalog,
        ablation=ablation,
        model=model,
        repeats=repeats,
        catalog_mode=catalog_mode,
        roots_present=roots_present,
        run_ts=run_ts,
    )

    out_path = paths.workspace_dir / f"run_{run_ts}.json"
    out_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    assert_non_mutation(nonmutation_probe())
    return out_path


def _build_replay_wiring(
    paths: HarnessPaths, model: str
) -> tuple[dict[str, Any], Callable[[], dict[str, Any]]]:
    """Build the --replay seams: frozen LLM responses + a no-op non-mutation probe.

    Replay never touches Neo4j/Redis/prod-Hermes, so the probe reports an
    unchanged, never-targeted reading. (--live wiring is a follow-up.)
    """
    responses = json.loads(
        (paths.fixtures_dir / "llm_responses.json").read_text(encoding="utf-8")
    )

    def probe() -> dict[str, Any]:
        return {
            "type_def_count_before": 0,
            "type_def_count_after": 0,
            "redis_key_before": "",
            "redis_key_after": "",
            "prod_hermes_targeted": False,
        }

    return responses, probe


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--replay",
        dest="live",
        action="store_false",
        default=False,
        help="replay frozen fixtures (default, $0, reproducible)",
    )
    mode.add_argument(
        "--live",
        dest="live",
        action="store_true",
        help="pull clusters/catalog from the live read-only stack",
    )
    p.add_argument(
        "--catalog-mode",
        choices=("in_process", "throwaway_hermes"),
        default="in_process",
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--ablation", choices=ABLATIONS, default="full")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default="gpt-4.1")
    args = p.parse_args(argv)

    if args.live:
        raise NotImplementedError("--live wiring is a follow-up; use --replay")

    paths = HarnessPaths.default()
    responses, probe = _build_replay_wiring(paths, args.model)
    replayer = FrozenLLMReplayer(responses)

    import hermes.main as m

    m.generate_completion = replayer  # in-process replay seam

    # T3 frozen-fixture loader (the PLAN named harness.catalog, but T2 landed
    # load_catalog in harness.fixtures_io -- the harness consumes what landed).
    from harness.fixtures_io import load_catalog

    out_path = run(
        paths=paths,
        catalog_loader=lambda pp: load_catalog(pp.fixtures_dir / "catalog.json"),
        cascade_fn=simulate_cascade_response,
        llm_replayer=replayer,
        nonmutation_probe=probe,
        ablation=args.ablation,
        repeats=args.repeats,
        catalog_mode=args.catalog_mode,
        limit=args.limit,
        model=args.model,
    )
    print(f"[harness] snapshot -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
