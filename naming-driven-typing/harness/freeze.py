"""Gated K-sample freeze: capture live /type-cluster completions per arm.

The graded run is always ``--replay`` over frozen fixtures (SPEC sections 7.6
and 7.7). This module owns the ONE paid step that produces those fixtures:
drive the hermes ``/type-cluster`` pass once per cluster x repeat x arm with
pinned sampling, capture every RAW completion, and write the per-arm
``fixtures/llm_responses*.json`` files in exactly the replayer shape
(``\"<cluster_id>::<repeat>\" -> {\"choices\": [{\"message\": {\"content\": ...}}]}``)
plus a ``fixtures/freeze_meta.json`` sidecar recording the model snapshot ids
and the pinned sampling parameters.

Why prompts run in-process while the LLM call goes to the deployed hermes
==========================================================================

The deployed ``/type-cluster`` endpoint returns the VALIDATED parse (groups,
residual_ids, raw_partition_ok): it carries neither the raw completion the
replayer needs (raw partition fidelity is a primary gated metric and must be
re-derivable from the frozen content) nor the model snapshot id, and its
prompt is built from the deployed registry, so the per-arm views (roots-only
for no_graft, the naive prompt for naive_llm, the no-chain override) cannot
be expressed against it. The freeze therefore drives the SAME
``/type-cluster`` handler code in-process (hermes redeployed from main is the
same code imported here) with the frozen catalog served per arm, and forwards
the inner ``generate_completion`` call over HTTP to the deployed instance
gateway (``POST {HERMES_URL}/llm``): the deployed hermes holds the LLM key
and performs the paid call, and its response carries both the raw completion
and the model snapshot id. The deployed ``/type-cluster`` route itself is
never targeted (R14: prod registry perturbation).

Operator notes (the agent never runs the paid step)
====================================================

- Gates: ``LIVE_RUN=1`` AND ``HERMES_URL`` set explicitly AND the
  ``--yes-i-will-pay`` flag after the printed cost estimate. Any miss aborts
  before the first LLM call.
- Target the DISPOSABLE reseeded stack only. The deployed ``/llm`` path
  spawns background proposal ingestion from the prompt text (entity-level
  writes on the connected stack); pause maintenance/emergence schedulers for
  the freeze window so nothing mints types mid-freeze.
- ``top_p`` is not exposed by the hermes gateway; it stays at the provider
  default and is recorded as such in the freeze meta. ``temperature`` is
  pinned to 0.0 and enforced fail-closed by the transport.
- Arms whose prompt is byte-identical to the full arm (``no_reuse``,
  ``no_gate``) share ``llm_responses.json``; the freeze covers the four
  prompt-distinct arms (``full``, ``naive_llm``, ``no_graft``, ``no_chain``).
- A failed arm writes its captured-so-far samples to
  ``<fixture>.partial.json`` so paid samples are never silently lost.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from harness.ablations import (
    NaiveLLMClient,
    arm_message_transform,
    arm_registry_factory,
    responses_filename,
)
from harness.fixtures_io import freeze_llm_responses, load_catalog
from harness.probe import LiveGateError, require_live_env
from harness.run_experiment import (
    StubTypeRegistry,
    _load_clusters,
    call_type_cluster,
)

# The prompt-distinct arms the freeze must sample. no_reuse and no_gate are
# prompt-identical to full and replay llm_responses.json (see ablations.py).
FREEZE_ARMS: tuple[str, ...] = ("full", "naive_llm", "no_graft", "no_chain")

FREEZE_META_FILENAME = "freeze_meta.json"

# Rough token model for the printed cost estimate (estimate only; the
# explicit --yes-i-will-pay acknowledgement is the real gate).
_BASE_PROMPT_TOKENS = 280
_NAIVE_BASE_PROMPT_TOKENS = 110
_TOKENS_PER_CATALOG_ENTRY = 18
_TOKENS_PER_MEMBER = 12

SendFn = Callable[..., Dict[str, Any]]


class FreezeError(RuntimeError):
    """The freeze failed fail-closed (bad completion, broken pin, non-200)."""


class CapturingTransport:
    """Async ``generate_completion`` stand-in that forwards and captures.

    Drop-in for the ``FrozenLLMReplayer`` seam surface (``for_cluster`` /
    ``set_repeat`` / async call), so the same binding pattern drives it. Each
    call forwards the (optionally arm-transformed) messages through ``send``
    and captures the RAW completion content keyed ``<cluster_id>::<repeat>``
    in exactly the replayer fixture shape. Fail-closed: a duplicate key, a
    broken temperature/top_p pin, or a completion without content raises
    ``FreezeError`` before anything is recorded.
    """

    def __init__(
        self,
        send: SendFn,
        *,
        message_transform: Optional[
            Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
        ] = None,
    ) -> None:
        self._send = send
        self._transform = message_transform
        self._cluster_id: Optional[str] = None
        self._repeat: int = 0
        self.captured: dict[str, dict[str, Any]] = {}
        self.model_ids: set[str] = set()
        self.usage_totals: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def for_cluster(self, cluster_id: str) -> "CapturingTransport":
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
            raise FreezeError("CapturingTransport: no active cluster bound")
        key = f"{self._cluster_id}::{self._repeat}"
        if key in self.captured:
            raise FreezeError(
                f"duplicate LLM call for {key!r}: exactly one call per repeat"
            )
        pinned_temperature = 0.0 if temperature is None else float(temperature)
        if pinned_temperature != 0.0:
            raise FreezeError(
                f"freeze requires pinned temperature=0.0, got {temperature!r}"
            )
        if top_p is not None and float(top_p) != 1.0:
            raise FreezeError(
                f"freeze requires provider-default top_p, got {top_p!r}"
            )
        sent_messages = list(messages)
        if self._transform is not None:
            sent_messages = self._transform(sent_messages)
        completion = self._send(
            sent_messages,
            temperature=pinned_temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )
        content = self._extract_content(completion, key)
        frozen = {"choices": [{"message": {"content": content}}]}
        self.captured[key] = frozen
        model_id = completion.get("model") if isinstance(completion, dict) else None
        if isinstance(model_id, str) and model_id:
            self.model_ids.add(model_id)
        usage = completion.get("usage") if isinstance(completion, dict) else None
        if isinstance(usage, dict):
            for field in self.usage_totals:
                value = usage.get(field)
                if isinstance(value, int):
                    self.usage_totals[field] += value
        # Return exactly what --replay will return for this key, so the
        # in-process endpoint validates the very content being frozen.
        return frozen

    @staticmethod
    def _extract_content(completion: Any, key: str) -> str:
        if not isinstance(completion, dict):
            raise FreezeError(f"{key}: completion is not an object")
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices:
            raise FreezeError(f"{key}: completion has no choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content:
            raise FreezeError(f"{key}: completion has no message content")
        return content


def live_llm_send(
    hermes_url: str, *, model: Optional[str] = None, timeout: float = 180.0
) -> SendFn:
    """HTTP send through the deployed hermes gateway (``POST /llm``).

    A fresh ``conversation_id`` per call keeps the gateway context cache from
    injecting cached context into a pinned prompt. The gateway response is
    OpenAI-shaped (``model`` snapshot id + ``choices`` + ``usage``).
    """
    base = hermes_url.rstrip("/")

    def send(
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: Optional[int],
        metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        import uuid

        import httpx  # live path only

        payload: dict[str, Any] = {
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "metadata": {
                "purpose": "naming-driven-typing-freeze",
                "conversation_id": f"ndt-freeze-{uuid.uuid4().hex}",
            },
        }
        if model:
            payload["model"] = model
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        response = httpx.post(f"{base}/llm", json=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise FreezeError("hermes /llm returned a non-object body")
        return body

    return send


def _assignable_catalog_entries(arm: str, catalog: dict[str, Any]) -> int:
    """Catalog lines the arm prompt carries (0 when the catalog is hidden)."""
    if arm in ("naive_llm", "no_graft"):
        return 0
    return sum(
        1
        for rec in catalog.get("catalog_by_uuid", {}).values()
        if not rec.get("is_root")
    )


def estimate_freeze_cost(
    clusters: list[dict[str, Any]],
    catalog: dict[str, Any],
    *,
    repeats: int,
    arms: tuple[str, ...] = FREEZE_ARMS,
) -> dict[str, Any]:
    """Pure cost estimate: clusters x K x arms, with rough token figures.

    Output tokens use the endpoint cap (``min(4096, 512 + 24 * members)``) so
    the figure is an upper bound; input tokens are a heuristic over the
    prompt base, catalog block and member lines.
    """
    member_counts = [len(c.get("members", [])) for c in clusters]
    per_arm: dict[str, dict[str, int]] = {}
    for arm in arms:
        base = _NAIVE_BASE_PROMPT_TOKENS if arm == "naive_llm" else _BASE_PROMPT_TOKENS
        entries = _assignable_catalog_entries(arm, catalog)
        input_one_pass = sum(
            base + entries * _TOKENS_PER_CATALOG_ENTRY + n * _TOKENS_PER_MEMBER
            for n in member_counts
        )
        output_one_pass = sum(min(4096, 512 + 24 * n) for n in member_counts)
        per_arm[arm] = {
            "calls": len(clusters) * repeats,
            "est_input_tokens": input_one_pass * repeats,
            "est_output_tokens_cap": output_one_pass * repeats,
        }
    return {
        "n_clusters": len(clusters),
        "repeats": repeats,
        "arms": list(arms),
        "calls": sum(v["calls"] for v in per_arm.values()),
        "est_input_tokens": sum(v["est_input_tokens"] for v in per_arm.values()),
        "est_output_tokens_cap": sum(
            v["est_output_tokens_cap"] for v in per_arm.values()
        ),
        "per_arm": per_arm,
    }


def format_cost_estimate(est: dict[str, Any]) -> str:
    """Printable cost-estimate block shown BEFORE any live LLM call."""
    arms_csv = ", ".join(est["arms"])
    return "\n".join(
        [
            "[freeze] COST ESTIMATE (live LLM spend)",
            (
                "[freeze]   clusters={n} x repeats={k} x arms={a} ({arms}) "
                "-> {calls} live calls"
            ).format(
                n=est["n_clusters"],
                k=est["repeats"],
                a=len(est["arms"]),
                arms=arms_csv,
                calls=est["calls"],
            ),
            (
                "[freeze]   est input tokens ~{i}; est output tokens "
                "(hard cap) ~{o}"
            ).format(i=est["est_input_tokens"], o=est["est_output_tokens_cap"]),
            (
                "[freeze]   shared fixtures: no_reuse and no_gate replay "
                "llm_responses.json (prompt-identical to full)"
            ),
            (
                "[freeze]   this makes PAID live LLM calls; pass "
                "--yes-i-will-pay to proceed"
            ),
        ]
    )


def _write_meta(meta: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def run_freeze(
    *,
    clusters: list[dict[str, Any]],
    catalog: dict[str, Any],
    send: SendFn,
    fixtures_dir: Path,
    repeats: int = 5,
    arms: tuple[str, ...] = FREEZE_ARMS,
    hermes_gateway: str = "",
    frozen_at: Optional[str] = None,
) -> dict[str, Any]:
    """Drive the freeze per arm and write the per-arm frozen fixtures + meta.

    Per arm: install the arm registry view over the FROZEN catalog, bind
    ``hermes.main.generate_completion`` to a ``CapturingTransport`` over
    ``send`` (with the arm message transform), then POST every whole cluster
    K times through the in-process /type-cluster (the naive arm drives its
    own prompt path via ``NaiveLLMClient``). The endpoint validates each
    completion as it is captured, so a frozen fixture is replayable by
    construction. Non-200 / bad completions abort fail-closed, dumping the
    captured-so-far samples to ``<fixture>.partial.json``.
    """
    import hermes.main as m
    from fastapi.testclient import TestClient

    fixtures_dir = Path(fixtures_dir)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    frozen_at = frozen_at or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    model_ids: set[str] = set()
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    arm_files: dict[str, str] = {}
    total_calls = 0

    for arm in arms:
        responses_file = responses_filename(arm)
        transport = CapturingTransport(
            send, message_transform=arm_message_transform(arm)
        )
        make_registry = arm_registry_factory(arm) or StubTypeRegistry.from_catalog
        client: Any = (
            NaiveLLMClient(transport) if arm == "naive_llm" else TestClient(m.app)
        )
        prev_registry = m._type_registry
        prev_generate = m.generate_completion
        m._type_registry = make_registry(catalog)
        m.generate_completion = transport
        try:
            for cluster in clusters:
                cluster_id = cluster["cluster_id"]
                members = cluster["members"]
                transport.for_cluster(cluster_id)
                for repeat in range(repeats):
                    transport.set_repeat(repeat)
                    call_type_cluster(
                        client, members, request_id=f"{cluster_id}::{repeat}"
                    )
        except Exception as err:
            partial_path = fixtures_dir / (responses_file + ".partial.json")
            if transport.captured:
                freeze_llm_responses(transport.captured, partial_path)
            raise FreezeError(
                f"freeze aborted in arm {arm!r}: {err} "
                f"(captured-so-far written to {partial_path.name})"
            ) from err
        finally:
            m._type_registry = prev_registry
            m.generate_completion = prev_generate
        freeze_llm_responses(transport.captured, fixtures_dir / responses_file)
        arm_files[arm] = responses_file
        model_ids.update(transport.model_ids)
        for field in usage_totals:
            usage_totals[field] += transport.usage_totals[field]
        total_calls += len(transport.captured)

    meta = {
        "frozen_at": frozen_at,
        "repeats": repeats,
        "n_clusters": len(clusters),
        "calls": total_calls,
        "arm_fixtures": {
            **arm_files,
            "no_reuse": responses_filename("no_reuse"),
            "no_gate": responses_filename("no_gate"),
        },
        "model_snapshot_ids": sorted(model_ids),
        "pinned_sampling": {
            "temperature": 0.0,
            "top_p": "provider-default (hermes gateway does not expose top_p)",
        },
        "usage_totals": usage_totals,
        "hermes_gateway": hermes_gateway,
    }
    _write_meta(meta, fixtures_dir / FREEZE_META_FILENAME)
    return meta


def freeze_command(
    *,
    paths: Any,
    repeats: int,
    model: Optional[str],
    limit: Optional[int],
    yes_i_will_pay: bool,
) -> int:
    """CLI flow for ``--freeze``: estimate -> acknowledge -> gates -> freeze.

    The cost estimate always prints (offline, free). Without the explicit
    ``--yes-i-will-pay`` acknowledgement, or without ``LIVE_RUN=1`` +
    ``HERMES_URL``, this returns 2 having touched nothing.
    """
    clusters = _load_clusters(paths)
    if limit is not None:
        clusters = clusters[:limit]
    catalog = load_catalog(paths.fixtures_dir / "catalog.json")
    estimate = estimate_freeze_cost(clusters, catalog, repeats=repeats)
    print(format_cost_estimate(estimate), flush=True)
    if not yes_i_will_pay:
        print(
            "[freeze] refusing to spend: re-run with --yes-i-will-pay to "
            "acknowledge the estimate above",
            flush=True,
        )
        return 2
    try:
        hermes_url = require_live_env()
    except LiveGateError as err:
        print(f"[freeze] {err}", file=sys.stderr, flush=True)
        return 2
    meta = run_freeze(
        clusters=clusters,
        catalog=catalog,
        send=live_llm_send(hermes_url, model=model),
        fixtures_dir=Path(paths.fixtures_dir),
        repeats=repeats,
        hermes_gateway=hermes_url,
    )
    snapshot_ids = ", ".join(meta["model_snapshot_ids"]) or "<none reported>"
    print(
        "[freeze] froze {calls} completions across {arms} arms; "
        "model snapshot ids: {ids}".format(
            calls=meta["calls"], arms=len(FREEZE_ARMS), ids=snapshot_ids
        ),
        flush=True,
    )
    return 0
