"""ANALYZE: drill into ONE clustering config -- per-entity assignments + confusion.

Reuses the sweep's EXACT clustering (imports sweep.py) so the labels match the
scored config. Reads a fixture.json (+ optionally results.json to auto-pick the
top-ranked node config) and writes:
  - per-domain recall (largest single-cluster share of each domain)
  - per-cluster majority domain + purity + domain mix
  - the misclustered (minority-in-cluster) entities, by name

Additive + read-only w.r.t. the sweep -- safe to run anytime against a saved
fixture, no re-ingest, does not touch capture/sweep/harness.

Usage:
  poetry run python analyze.py --fixture fixture.json --results results.json
  poetry run python analyze.py --fixture fixture.json \
      --scheme name+ctx:concat:a0.5 --algorithm kmeans --preproc raw \
      --k-mode n_domains --min 2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sweep", HERE / "sweep.py")
sw = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(sw)  # type: ignore[union-attr]


def pick_top_config(results: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in results.get("node_results", []) if r.get("ari") == r.get("ari")]
    if not rows:
        raise SystemExit("results.json has no scored node configs")
    return rows[0]


def cluster_one(
    entities: list[dict[str, Any]], cfg: dict[str, Any]
) -> tuple[list[str], list[str], list[int], int]:
    scheme = cfg.get("scheme", "name")
    schemes = {label: ents for label, ents in sw._node_feature_schemes(entities)}
    if scheme not in schemes:
        raise SystemExit(f"scheme {scheme!r} not in fixture; have {list(schemes)}")
    ents = schemes[scheme]
    embeddings = [e["embedding"] for e in ents]
    domains = [e.get("domain", "unknown") for e in ents]
    names = [e.get("name", e["uuid"]) for e in ents]
    n_domains = len({d for d in domains if d != "unknown"}) or len(set(domains))
    x = sw.preprocess(np.asarray(embeddings, dtype=float), cfg["preprocessing"])
    labels = sw._node_labels(
        cfg["algorithm"], x, embeddings, n_domains,
        cfg["min_cluster_size"], cfg["k_mode"], {},
    )
    labels = sw._apply_min_size(np.asarray(labels), cfg["min_cluster_size"])
    return names, domains, [int(v) for v in labels], n_domains


def analyze(entities: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    names, domains, labels, n_domains = cluster_one(entities, cfg)
    conf: dict[int, Counter] = defaultdict(Counter)
    members: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for nm, d, lab in zip(names, domains, labels):
        conf[lab][d] += 1
        members[lab].append((nm, d))

    cluster_rows = []
    misclustered = []
    for lab in sorted(conf):
        c = conf[lab]
        tot = sum(c.values())
        maj_dom, maj_n = c.most_common(1)[0]
        cluster_rows.append(
            {
                "cluster": lab,
                "size": tot,
                "majority_domain": maj_dom,
                "purity": round(maj_n / tot, 3),
                "domain_mix": dict(c),
            }
        )
        if lab != -1:
            for nm, d in members[lab]:
                if d != maj_dom:
                    misclustered.append(
                        {"name": nm, "true_domain": d, "cluster": lab,
                         "cluster_domain": maj_dom}
                    )

    domain_recall = {}
    for d in sorted(set(domains)):
        by_cluster = Counter(
            lab for lab, dd in zip(labels, domains) if dd == d and lab != -1
        )
        tot = sum(1 for dd in domains if dd == d)
        best = by_cluster.most_common(1)[0][1] if by_cluster else 0
        domain_recall[d] = round(best / tot, 3) if tot else 0.0

    return {
        "config": cfg,
        "n_domains": n_domains,
        "n_entities": len(names),
        "n_clusters": len([r for r in cluster_rows if r["cluster"] != -1]),
        "domain_recall": domain_recall,
        "clusters": cluster_rows,
        "misclustered": misclustered,
    }


def render(a: dict[str, Any]) -> str:
    cfg = a["config"]
    L = [
        f"# Cluster analysis -- scheme={cfg.get('scheme', 'name')} "
        f"algo={cfg['algorithm']} preproc={cfg['preprocessing']} "
        f"k={cfg['k_mode']} min={cfg['min_cluster_size']}",
        "",
        f"- entities: {a['n_entities']}  domains: {a['n_domains']}  "
        f"clusters: {a['n_clusters']}",
        f"- scored ARI: {cfg.get('ari')}  purity: {cfg.get('purity')}  "
        f"coverage: {cfg.get('coverage')}",
        "",
        "## Per-domain recall (largest single-cluster share, worst first)",
        "",
    ]
    for d, r in sorted(a["domain_recall"].items(), key=lambda kv: kv[1]):
        L.append(f"- {d}: {r}")
    L += ["", "## Clusters (majority domain + purity + mix)", "",
          "| cluster | size | majority | purity | domain_mix |",
          "|---|---|---|---|---|"]
    for c in sorted(a["clusters"], key=lambda c: -c["size"]):
        L.append(
            f"| {c['cluster']} | {c['size']} | {c['majority_domain']} | "
            f"{c['purity']} | {c['domain_mix']} |"
        )
    L += ["", f"## Misclustered entities ({len(a['misclustered'])})", ""]
    for m in a["misclustered"][:80]:
        L.append(
            f"- {m['name']}: true={m['true_domain']} -> "
            f"cluster {m['cluster']} ({m['cluster_domain']})"
        )
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--results", type=Path, help="auto-pick top node config")
    ap.add_argument("--scheme")
    ap.add_argument("--algorithm")
    ap.add_argument("--preproc")
    ap.add_argument("--k-mode", dest="k_mode")
    ap.add_argument("--min", dest="min_cluster_size", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    fx = json.loads(args.fixture.read_text())
    if args.algorithm:
        cfg = {
            "scheme": args.scheme or "name",
            "algorithm": args.algorithm,
            "preprocessing": args.preproc or "raw",
            "k_mode": args.k_mode or "n_domains",
            "min_cluster_size": args.min_cluster_size or 2,
        }
    elif args.results:
        cfg = pick_top_config(json.loads(args.results.read_text()))
        cfg.setdefault("scheme", "name")
    else:
        raise SystemExit("pass --results (auto-pick) or --algorithm ... (manual)")

    a = analyze(fx["entities"], cfg)
    out = args.out or args.fixture.parent / "analysis.md"
    out.write_text(render(a))
    out.with_suffix(".json").write_text(json.dumps(a, indent=2))
    print(f"wrote {out} and {out.with_suffix('.json')}")
    print(render(a)[:1400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
