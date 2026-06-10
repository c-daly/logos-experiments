"""Post-hoc: add name-embedding evidence to an EXISTING mapping.csv.

For when the Neo4j-derived table is already committed but Neo4j isn't available
to re-run the full proposer. Reads ``mapping.csv`` (the df=1 rows) + a frozen
relation snapshot (survivors = ``edge_count > 1``) + cached embeddings, applies
the SAME ``apply_embed_fallback`` step the runner uses inline, and rewrites
``mapping.csv``. The runner (``propose_mapping.py``) does this on a fresh Neo4j
run; this reproduces just the embedding pass off the graph.

    uv run python enrich_mapping.py            # uses ./mapping.csv + ./snapshot.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from embed_evidence import load_vectors, nearest_survivors
from propose import Row, apply_embed_fallback

HERE = Path(__file__).resolve().parent
COLS = ["predicate", "df", "proposed_target", "tier", "evidence", "review"]


def load_rows(path: Path) -> list[Row]:
    rows = []
    for d in csv.DictReader(path.open()):
        rows.append(
            Row(
                d["predicate"], int(d["df"]), d["proposed_target"],
                d["tier"], d["evidence"], d.get("review", ""),
            )
        )
    return rows


def write_rows(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            w.writerow([r.predicate, r.df, r.target, r.tier, r.evidence, r.review])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=str(HERE / "mapping.csv"))
    ap.add_argument("--snapshot", default=str(HERE / "snapshot.json"))
    args = ap.parse_args()

    rows = load_rows(Path(args.mapping))
    snapshot = json.loads(Path(args.snapshot).read_text())
    one_offs = [r.predicate for r in rows]
    # Targets are genuine survivors only -- never a predicate that is itself in
    # the consolidation set (the committed table's df and the live snapshot's
    # edge_count can drift, so subtract the one-offs explicitly).
    survivors = {
        p for p, m in snapshot.items() if m.get("edge_count", 0) > 1
    } - set(one_offs)

    before = sum(1 for r in rows if r.tier != "keep")
    vectors = load_vectors(sorted(survivors | set(one_offs)))
    rows = apply_embed_fallback(rows, nearest_survivors(one_offs, survivors, vectors))
    proposals = sum(1 for r in rows if r.tier != "keep")  # rows with a fold target
    annotated = sum(1 for r in rows if r.evidence)  # incl. keep nearest-neighbour
    write_rows(rows, Path(args.mapping))

    n = len(rows)
    print("tiers:", dict(Counter(r.tier for r in rows)))
    print(
        f"proposal coverage (a fold target): {before} -> {proposals}/{n} "
        f"= {proposals / n:.1%}  (ticket gate: >=80%)"
    )
    print(
        f"rows annotated with evidence (incl. keep nearest-neighbour): "
        f"{annotated}/{n} = {annotated / n:.1%}  (review aid, not the gate)"
    )


if __name__ == "__main__":
    main()
