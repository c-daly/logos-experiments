#!/usr/bin/env bash
# =============================================================================
# Real-data clustering grid search  --  "edge-embeddings-worth-it" experiment
# =============================================================================
# End-to-end, reproducible, FROM A CLEAN SLATE:
#
#   1. (re)start Hermes + Sophia with LOGOS_EMBEDDING_DIM=1536
#   2. CLEAR all three stores  (Milvus collections, Redis, Neo4j)
#   3. ingest the corpus fresh via the harness   -> workspace/round_N.json
#   4. capture a static fixture from the live graph -> sweep/fixture.json
#   5. grid-search clustering configs offline       -> sweep/results.json + REPORT.md
#
# Why clean-slate (not a live-graph snapshot): the graph drifts (cumulative
# rounds + additive-ingestion duplicates), so a snapshot is not provenance-clean
# or comparable. goal.yaml: "DB reset between runs is fine."
#
# WHY all three stores (the harness only clears Neo4j via seeder.clear()):
#   - Milvus keeps stale entity/edge vectors + type centroids
#   - Redis keeps a stale type-registry (Hermes loads it on boot)
# Both must go for a true cold start.
#
# WHY the dim pin: OpenAI text-embedding-3-small is 1536-d. If a writer
# (re)creates a Milvus collection with LOGOS_EMBEDDING_DIM unset it defaults to
# 384 and silently rejects the 1536-d vectors. So every writer (Hermes, Sophia,
# the harness) is started with LOGOS_EMBEDDING_DIM=1536.
#
# PREREQS:
#   - docker infra up: Neo4j :7687, Milvus :19530, Redis :6379
#   - OPENAI_API_KEY exported in the calling environment
#   - sophia poetry env has scikit-learn + hdbscan + umap-learn
#       (poetry run python -m pip install scikit-learn hdbscan umap-learn)
#   - hermes checked out on branch feat/505-name-cluster
#
# USAGE:  ./run_real_grid_search.sh [ROUNDS]      (default ROUNDS=3 -> 6 domains)
# =============================================================================
set -euo pipefail

ROUNDS="${1:-3}"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOPHIA_DIR="${SOPHIA_DIR:-/home/fearsidhe/projects/logos-workspace/sophia}"
HERMES_DIR="${HERMES_DIR:-/home/fearsidhe/projects/logos-workspace/hermes}"

export LOGOS_EMBEDDING_DIM="${LOGOS_EMBEDDING_DIM:-1536}"
export SOPHIA_API_TOKEN="${SOPHIA_API_TOKEN:-sophia_dev}"
# Sophia emergence naming calls Hermes /name-cluster; its FeedbackConfig default
# is :18000 (wrong) while Hermes runs on :17000 -> Connection refused -> no mints.
# Override so emergence can actually name + mint emergent types.
export SOPHIA_FEEDBACK_HERMES_URL="${SOPHIA_FEEDBACK_HERMES_URL:-http://localhost:17000}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-logos-hcg-neo4j}"
REDIS_CONTAINER="${REDIS_CONTAINER:-logos-hcg-redis}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-logosdev}"
MILVUS_HOST="${MILVUS_HOST:-localhost}"
MILVUS_PORT="${MILVUS_PORT:-19530}"

: "${OPENAI_API_KEY:?export OPENAI_API_KEY before running}"

say() { printf '\n==> %s\n' "$*"; }
run_py() { ( cd "$SOPHIA_DIR" && poetry run python "$@" ); }

wait_health() {  # name url
  local name="$1" url="$2" i
  for i in $(seq 1 40); do
    if curl -s -m 3 "$url" >/dev/null 2>&1; then echo "    $name healthy"; return 0; fi
    sleep 3
  done
  echo "    ERROR: $name never came healthy at $url" >&2; return 1
}

# ---------------------------------------------------------------------------
say "[1/5] (re)start Hermes + Sophia with LOGOS_EMBEDDING_DIM=$LOGOS_EMBEDDING_DIM"
# bracket trick in the pattern avoids pkill matching its own command line
pkill -f "[u]vicorn hermes.main"  2>/dev/null || true
pkill -f "[u]vicorn sophia.api"   2>/dev/null || true
sleep 2

( cd "$HERMES_DIR" && LOGOS_EMBEDDING_DIM="$LOGOS_EMBEDDING_DIM" \
    nohup env OPENAI_API_KEY="$OPENAI_API_KEY" \
    poetry run uvicorn hermes.main:app --host 127.0.0.1 --port 17000 \
    >/tmp/hermes.log 2>&1 & )

( cd "$SOPHIA_DIR" && LOGOS_EMBEDDING_DIM="$LOGOS_EMBEDDING_DIM" \
    SOPHIA_API_TOKEN="$SOPHIA_API_TOKEN" \
    nohup env OPENAI_API_KEY="$OPENAI_API_KEY" \
    poetry run uvicorn sophia.api.app:app --host 0.0.0.0 --port 47000 \
    >/tmp/sophia.log 2>&1 & )

wait_health hermes http://localhost:17000/health
wait_health sophia http://localhost:47000/health

# ---------------------------------------------------------------------------
say "[2/5] CLEAR all three stores (Milvus + Redis + Neo4j)"
echo "    Milvus: dropping all collections ..."
( cd "$SOPHIA_DIR" && poetry run python - "$MILVUS_HOST" "$MILVUS_PORT" <<'PY'
import sys
from pymilvus import connections, utility
connections.connect(host=sys.argv[1], port=sys.argv[2])
for c in utility.list_collections():
    utility.drop_collection(c)
    print(f"      dropped {c}")
PY
)
echo "    Redis: FLUSHALL ..."
docker exec "$REDIS_CONTAINER" redis-cli FLUSHALL
echo "    Neo4j: DETACH DELETE all ..."
docker exec "$NEO4J_CONTAINER" cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
    "MATCH (n) DETACH DELETE n;"

# Restart Hermes+Sophia so their in-memory caches / collection handles are fresh
# after the wipe (Hermes recreates hermes_embeddings at the pinned dim).
say "    re-cycling services after wipe ..."
pkill -f "[u]vicorn hermes.main" 2>/dev/null || true
pkill -f "[u]vicorn sophia.api"  2>/dev/null || true
sleep 2
( cd "$HERMES_DIR" && LOGOS_EMBEDDING_DIM="$LOGOS_EMBEDDING_DIM" \
    nohup env OPENAI_API_KEY="$OPENAI_API_KEY" \
    poetry run uvicorn hermes.main:app --host 127.0.0.1 --port 17000 \
    >/tmp/hermes.log 2>&1 & )
( cd "$SOPHIA_DIR" && LOGOS_EMBEDDING_DIM="$LOGOS_EMBEDDING_DIM" \
    SOPHIA_API_TOKEN="$SOPHIA_API_TOKEN" \
    nohup env OPENAI_API_KEY="$OPENAI_API_KEY" \
    poetry run uvicorn sophia.api.app:app --host 0.0.0.0 --port 47000 \
    >/tmp/sophia.log 2>&1 & )
wait_health hermes http://localhost:17000/health
wait_health sophia http://localhost:47000/health

# ---------------------------------------------------------------------------
say "[3/5] ingest the corpus fresh (rounds=$ROUNDS) -- harness reseeds Neo4j + ingests"
run_py "$EXP/harness/run_experiment.py" --seed-n 1 --rounds "$ROUNDS"

ROUND_JSON="$EXP/workspace/round_${ROUNDS}.json"
[ -f "$ROUND_JSON" ] || { echo "ERROR: $ROUND_JSON not produced" >&2; exit 1; }

# ---------------------------------------------------------------------------
say "[4/5] capture static fixture from the live graph ($ROUND_JSON)"
run_py "$EXP/sweep/capture.py" --round-json "$ROUND_JSON" --out "$EXP/sweep/fixture.json"

# ---------------------------------------------------------------------------
say "[5/5] grid-search clustering configs over the captured fixture"
run_py "$EXP/sweep/sweep.py" --fixture "$EXP/sweep/fixture.json" --out "$EXP/sweep/results.json"

say "DONE"
echo "    fixture : $EXP/sweep/fixture.json"
echo "    results : $EXP/sweep/results.json"
echo "    report  : $EXP/sweep/REPORT.md"
