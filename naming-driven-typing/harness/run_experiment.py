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
