"""Real non-mutation probe seams for the live harness paths (issue #13).

``--replay`` keeps its no-op probe (replay never touches a stack). The live
paths capture a BEFORE state when the probe is built and an AFTER state when
the probe is called, and ``assert_non_mutation`` fires on any drift BEFORE the
run snapshot persists (the landed ordering in ``run_experiment.run``).

Probed invariants (SPEC section 6, non-mutation):

(a) Neo4j ``type_definition`` count -- read through the same read API the
    reseed/catalog path already uses (``HCGClient.get_all_type_definitions``).
(b) Prod Redis ontology snapshot key ``logos:ontology:types`` -- READ ONLY;
    compared by sha256 content hash, with an explicit sentinel for an absent
    key so a deleted key and an empty key both register as drift.
(c) Hermes TypeRegistry count -- read from the in-process
    ``hermes.main._type_registry`` when one is installed. The offline harness
    never runs hermes startup, so this is usually ``None``; that case is
    covered by (b), because the prod TypeRegistry is exactly a view of the
    Redis key probed there.

Every reader failure is wrapped in ``NonMutationViolation`` (fail closed): a
probe that cannot read the stack must abort the run rather than vouch for it,
and the landed ordering guarantees the abort lands before ``run_<ts>.json``.

The pure pieces (state snapshot, payload assembly, hashing) are unit-tested
with fakes; only ``build_live_readers`` touches real clients, and the live
callers gate it behind ``LIVE_RUN=1`` (mirroring the ``RESEED_LIVE`` idiom).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from harness.run_experiment import NonMutationViolation, PROD_REDIS_KEY

# Sentinel recorded in place of a content hash when the probed key is absent.
ABSENT_KEY = "<absent>"


class LiveGateError(RuntimeError):
    """A live-path env gate (LIVE_RUN / HERMES_URL) is not satisfied."""


def require_live_env(env: Optional[Mapping[str, str]] = None) -> str:
    """Enforce the live-path env gates; returns the deployed hermes URL.

    Mirrors the RESEED_LIVE idiom: ``LIVE_RUN=1`` must be set explicitly, and
    ``HERMES_URL`` must be set explicitly (no prod default fallback) so a live
    run can never silently target the default prod URL.
    """
    env = os.environ if env is None else env
    if env.get("LIVE_RUN") != "1":
        raise LiveGateError(
            "live path is gated: set LIVE_RUN=1 explicitly (and HERMES_URL) "
            "to run against a live stack"
        )
    hermes_url = (env.get("HERMES_URL") or "").strip()
    if not hermes_url:
        raise LiveGateError(
            "live path requires HERMES_URL set explicitly to the deployed "
            "hermes instance (prod default URLs are never assumed)"
        )
    return hermes_url


@dataclass(frozen=True)
class StackReaders:
    """Read-only stack accessors the probe snapshots (injectable for tests)."""

    type_def_count: Callable[[], int]
    redis_key_bytes: Callable[[], Optional[bytes]]
    hermes_registry_count: Optional[Callable[[], Optional[int]]] = None


def hash_key_content(value: Optional[Any]) -> str:
    """sha256 hex of the key content; ``ABSENT_KEY`` sentinel when missing."""
    if value is None:
        return ABSENT_KEY
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def snapshot_stack_state(readers: StackReaders) -> dict[str, Any]:
    """One read-only stack snapshot; any reader failure fails closed."""
    try:
        type_def_count = int(readers.type_def_count())
    except Exception as err:
        raise NonMutationViolation(
            f"non-mutation probe failed reading the Neo4j type-def count: {err}"
        ) from err
    try:
        redis_key_hash = hash_key_content(readers.redis_key_bytes())
    except Exception as err:
        raise NonMutationViolation(
            "non-mutation probe failed reading the prod Redis key "
            f"{PROD_REDIS_KEY!r}: {err}"
        ) from err
    registry_count: Optional[int] = None
    if readers.hermes_registry_count is not None:
        try:
            raw = readers.hermes_registry_count()
            registry_count = None if raw is None else int(raw)
        except Exception as err:
            raise NonMutationViolation(
                "non-mutation probe failed reading the hermes TypeRegistry "
                f"count: {err}"
            ) from err
    return {
        "type_def_count": type_def_count,
        "redis_key_hash": redis_key_hash,
        "hermes_registry_count": registry_count,
    }


def probe_payload(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    prod_hermes_targeted: bool = False,
) -> dict[str, Any]:
    """Assemble the dict shape ``assert_non_mutation`` consumes (pure)."""
    return {
        "type_def_count_before": before["type_def_count"],
        "type_def_count_after": after["type_def_count"],
        "redis_key_before": before["redis_key_hash"],
        "redis_key_after": after["redis_key_hash"],
        "hermes_registry_count_before": before["hermes_registry_count"],
        "hermes_registry_count_after": after["hermes_registry_count"],
        "prod_hermes_targeted": prod_hermes_targeted,
    }


def make_live_probe(readers: StackReaders) -> Callable[[], dict[str, Any]]:
    """Capture the BEFORE state now; the returned probe captures AFTER.

    Wire the returned callable as ``run(..., nonmutation_probe=...)``: the
    landed ordering asserts it BEFORE the run snapshot persists, so any drift
    (or any probe read failure) aborts without leaving a poisoned
    ``run_<ts>.json`` on disk.
    """
    before = snapshot_stack_state(readers)

    def probe() -> dict[str, Any]:
        return probe_payload(before, snapshot_stack_state(readers))

    return probe


def build_live_readers(
    *,
    neo4j_client: Optional[Any] = None,
    redis_client: Optional[Any] = None,
) -> StackReaders:
    """Real readers for the live path (lazy imports; env-gated by callers).

    The Neo4j reader uses the same read API the reseed/catalog path already
    uses (``get_all_type_definitions``); the Redis reader GETs the prod
    ontology snapshot key (never writes); the hermes reader sees the
    in-process registry only when hermes startup installed one (otherwise it
    reports ``None`` and the Redis key hash carries the invariant).
    """
    if neo4j_client is None:
        from logos_hcg.client import HCGClient  # live path only

        neo4j_client = HCGClient(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "logosdev"),
        )
    if redis_client is None:
        import redis  # live path only

        from logos_config import RedisConfig

        redis_client = redis.from_url(RedisConfig().url)

    client = neo4j_client
    rclient = redis_client

    def type_def_count() -> int:
        return len(client.get_all_type_definitions() or [])

    def redis_key_bytes() -> Optional[bytes]:
        return rclient.get(PROD_REDIS_KEY)

    def hermes_registry_count() -> Optional[int]:
        import hermes.main as m  # in-process registry, when installed

        registry = m._type_registry
        if registry is None:
            return None
        return len(registry.get_type_names())

    return StackReaders(
        type_def_count=type_def_count,
        redis_key_bytes=redis_key_bytes,
        hermes_registry_count=hermes_registry_count,
    )
