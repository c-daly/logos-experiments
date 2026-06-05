"""Shared fixtures for cascade simulator tests.

Hand-built enriched catalog matching the T2 schema:
    catalog_by_uuid: dict[uuid -> {uuid,name,norm_name,member_count,
                                   ancestors,chain,depth,parent_uuid,is_root}]
    by_norm: dict[norm_name -> list[uuid]]   (LIST, per T2)

Roots (entity/concept/process) are present and is_root True. A second
'vehicle' uuid exercises the by_norm multi-match tiebreak. No live stack,
no LLM, no graph writes -- pure dict fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make 'harness' importable: experiment dir is the parent of tests/.
_EXP_DIR = Path(__file__).resolve().parent.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))


def _rec(uuid, name, ancestors, member_count, parent_uuid, is_root=False):
    chain = list(ancestors) + [name]
    return {
        "uuid": uuid,
        "name": name,
        "norm_name": name,  # fixtures use already-canonical names
        "member_count": member_count,
        "ancestors": list(ancestors),  # root-first, self-excluded
        "chain": chain,                # root-first
        "depth": len(chain),
        "parent_uuid": parent_uuid,
        "is_root": is_root,
    }


@pytest.fixture
def catalog_by_uuid():
    # Roots
    recs = [
        _rec("u-entity", "entity", [], 100, None, is_root=True),
        _rec("u-concept", "concept", [], 50, None, is_root=True),
        _rec("u-process", "process", [], 40, None, is_root=True),
        # entity branch: vehicle -> car -> (sedan is the type being minted)
        _rec("u-vehicle-A", "vehicle", ["entity"], 9, "u-entity"),
        _rec("u-vehicle-B", "vehicle", ["entity"], 3, "u-entity"),  # twin
        _rec("u-car", "car", ["entity", "vehicle"], 7, "u-vehicle-A"),
        # homonym: bear-animal (entity) vs bear-process (process)
        _rec("u-bear-animal", "bear", ["entity"], 2, "u-entity"),
        _rec("u-bear-process", "bear", ["process"], 1, "u-process"),
    ]
    return {r["uuid"]: r for r in recs}


@pytest.fixture
def by_norm(catalog_by_uuid):
    out: dict[str, list[str]] = {}
    for uuid, rec in catalog_by_uuid.items():
        out.setdefault(rec["norm_name"], []).append(uuid)
    # deterministic order so list contents are stable across runs
    for k in out:
        out[k].sort()
    return out
