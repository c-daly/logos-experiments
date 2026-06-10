"""Apply an approved consolidation mapping to the live graph (W0.3b, #34).

The separate, reviewed maintenance step the proposer defers: it folds the df=1
one-off predicates named in ``mapping.csv`` into their proposed survivors by
renaming ``relation`` on the reified edge nodes (``:Node {type:'edge'}``).

**Dry-run is the default and needs no graph access** -- it previews the fold
set and the projected df=1 from the frozen ``snapshot.json``, so the change can
be inspected before anything is touched. ``--apply`` performs the mutation
(``NEO4J_PASSWORD`` required) inside one transaction, after writing a rollback
file mapping every touched edge uuid to its original relation.

Scope defaults to the safe-on-glance tiers (``high`` canonical + ``embed``
synonym); widen with ``--tiers``. The ``review`` column overrides per row:
an accept word forces a fold in, a reject/keep word forces it out.

    uv run python apply_mapping.py                      # dry-run preview (no graph)
    uv run python apply_mapping.py --graph-check        # exact live edge counts (read-only)
    NEO4J_PASSWORD=... uv run python apply_mapping.py --apply   # mutate + write rollback
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Default auto-apply scope: canonical (high), embedding synonym (embed), and
# EXACT token matches (medium that share all content tokens). Lossy token
# matches and signature-only (low) proposals are held back for explicit review
# -- they collapse specific relations into vague heads or match by coincidence.
DEFAULT_TIERS = ("high", "embed", "medium")
ACCEPT_WORDS = {"accept", "apply", "yes", "y", "approved", "ok"}
REJECT_WORDS = {"reject", "keep", "no", "n", "skip", "drop"}


def is_lossy(row: dict) -> bool:
    """A token-match proposal that dropped tokens from the one-off (the proposer
    annotates these 'lossy'); folding them can collapse a specific relation."""
    return "lossy" in (row.get("evidence") or "")


# --------------------------------------------------------------------------
# Pure selection / projection (unit-tested; no I/O)
# --------------------------------------------------------------------------


def select_folds(
    rows: list[dict],
    tiers: tuple[str, ...] = DEFAULT_TIERS,
    include_lossy: bool = False,
) -> list[tuple[str, str]]:
    """Pick (predicate -> target) folds to apply.

    A row is in if it has a target and is not vetoed. The ``review`` column
    decides first: an accept word forces the fold in (overriding tier and the
    lossy guard); a reject/keep word forces it out. For un-reviewed rows the
    row's tier must be in ``tiers`` and -- unless ``include_lossy`` -- it must
    not be a lossy token match. (Lossy matches and signature-only/low rows are
    held back for explicit review; widen scope deliberately, not by accident.)
    """
    folds = []
    for r in rows:
        target = (r.get("proposed_target") or "").strip()
        if not target:
            continue
        review = (r.get("review") or "").strip().lower()
        if review in REJECT_WORDS:
            continue
        if review in ACCEPT_WORDS:
            folds.append((r["predicate"], target))
            continue
        if review == "" and r.get("tier") in tiers:
            if is_lossy(r) and not include_lossy:
                continue
            folds.append((r["predicate"], target))
    return folds


def _resolve(target: str, fold_map: dict[str, str]) -> str:
    """Follow a chain of folds to its ultimate target (A->B, B->C => A->C)."""
    seen = {target}
    while target in fold_map and fold_map[target] not in seen:
        target = fold_map[target]
        seen.add(target)
    return target


def project_after_folds(
    folds: list[tuple[str, str]], snapshot: dict[str, dict]
) -> dict:
    """df=1 fraction before vs after applying ``folds`` (each one-off's edges
    move to its ultimate target), estimated from snapshot edge counts."""
    fold_map = dict(folds)
    final: dict[str, int] = {p: m.get("edge_count", 0) for p, m in snapshot.items()}
    edges_moved = 0
    for src, _ in folds:
        if src in final:
            moved = final.pop(src)
            tgt = _resolve(src, fold_map)
            final[tgt] = final.get(tgt, 0) + moved
            edges_moved += moved

    def df1_frac(counts: dict[str, int]) -> float:
        d = len(counts)
        return round(sum(1 for v in counts.values() if v <= 1) / d, 3) if d else 0.0

    before = {p: m.get("edge_count", 0) for p, m in snapshot.items()}
    return {
        "folds": len(folds),
        "edges_moved": edges_moved,
        "distinct_before": len(before),
        "distinct_after": len(final),
        "df1_before": df1_frac(before),
        "df1_after": df1_frac(final),
    }


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _driver():
    from neo4j import GraphDatabase

    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        sys.exit("NEO4J_PASSWORD must be set explicitly for --apply / --graph-check.")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    return GraphDatabase.driver(uri, auth=(user, password))


def graph_counts(driver, srcs: list[str]) -> dict[str, int]:
    """Exact edge count per source predicate in the live graph (read-only)."""

    def work(tx):
        rows = tx.run(
            "MATCH (e:Node {type:'edge'}) WHERE e.relation IN $srcs "
            "RETURN e.relation AS rel, count(e) AS n",
            srcs=srcs,
        )
        return {r["rel"]: r["n"] for r in rows}

    with driver.session() as s:
        return s.execute_read(work)


def apply_folds(driver, folds: list[tuple[str, str]]) -> tuple[int, list[dict]]:
    """Rename relation on edge nodes, in one transaction. Returns (edges
    changed, rollback rows) where each rollback row is {uuid, old, new}."""
    payload = [{"src": s, "tgt": t} for s, t in folds]

    def work(tx):
        rollback = [
            {"uuid": r["uuid"], "old": r["old"], "new": r["new"]}
            for r in tx.run(
                "UNWIND $folds AS f "
                "MATCH (e:Node {type:'edge'}) WHERE e.relation = f.src "
                "RETURN e.uuid AS uuid, e.relation AS old, f.tgt AS new",
                folds=payload,
            )
        ]
        changed = tx.run(
            "UNWIND $folds AS f "
            "MATCH (e:Node {type:'edge'}) WHERE e.relation = f.src "
            "SET e.relation = f.tgt RETURN count(e) AS n",
            folds=payload,
        ).single()["n"]
        return changed, rollback

    with driver.session() as s:
        return s.execute_write(work)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=str(HERE / "mapping.csv"))
    ap.add_argument("--snapshot", default=str(HERE / "snapshot.json"))
    ap.add_argument(
        "--tiers", default=",".join(DEFAULT_TIERS),
        help="comma-separated tiers to apply for un-reviewed rows "
        f"(default: {','.join(DEFAULT_TIERS)}; 'low' is weak, opt in explicitly)",
    )
    ap.add_argument(
        "--include-lossy", action="store_true",
        help="also fold lossy token matches (drop-token medium); off by default",
    )
    ap.add_argument("--graph-check", action="store_true", help="read-only live counts")
    ap.add_argument("--apply", action="store_true", help="MUTATE the live graph")
    args = ap.parse_args()

    rows = load_rows(Path(args.mapping))
    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    folds = select_folds(rows, tiers, include_lossy=args.include_lossy)
    selected = {s for s, _ in folds}
    by_tier = Counter(r["tier"] for r in rows if r["predicate"] in selected)

    # Transparency: what has a target but is deliberately held back for review.
    held = [
        r for r in rows
        if (r.get("proposed_target") or "").strip()
        and r["predicate"] not in selected
        and (r.get("review") or "").strip().lower() not in REJECT_WORDS
    ]
    held_lossy = sum(1 for r in held if is_lossy(r) and r.get("tier") in tiers)
    held_oos = Counter(
        r["tier"] for r in held if not (is_lossy(r) and r.get("tier") in tiers)
    )

    print(
        f"selected {len(folds)} folds  "
        f"tiers={list(tiers)} include_lossy={args.include_lossy}"
    )
    print(f"  included by tier: {dict(by_tier)}")
    print(
        f"  held back for review: {held_lossy} lossy-medium + "
        f"{sum(held_oos.values())} out-of-scope {dict(held_oos)}"
    )
    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    proj = project_after_folds(folds, snapshot)
    print(
        f"  projected (snapshot estimate): df=1 {proj['df1_before']} -> "
        f"{proj['df1_after']}  | distinct {proj['distinct_before']} -> "
        f"{proj['distinct_after']}  | edges moved ~{proj['edges_moved']}"
    )

    if args.graph_check:
        driver = _driver()
        try:
            counts = graph_counts(driver, [s for s, _ in folds])
        finally:
            driver.close()
        total = sum(counts.values())
        print(f"  live graph: {len(counts)}/{len(folds)} sources present, "
              f"{total} edges would be renamed")
        return

    if not args.apply:
        print("\nDRY RUN -- nothing changed. Re-run with --apply (and "
              "NEO4J_PASSWORD) to mutate the live graph.")
        return

    driver = _driver()
    try:
        changed, rollback = apply_folds(driver, folds)
    finally:
        driver.close()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rollback_path = HERE / f"rollback_{stamp}.json"
    rollback_path.write_text(json.dumps(rollback, indent=2), encoding="utf-8")
    print(f"\nAPPLIED: {changed} edges renamed across {len(folds)} folds.")
    print(f"rollback written: {rollback_path} ({len(rollback)} edges)")


if __name__ == "__main__":
    main()
