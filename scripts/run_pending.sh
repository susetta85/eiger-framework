#!/usr/bin/env bash
# Runs the configs of one or more sweep folders that are not done yet.
# Usage: scripts/run_pending.sh FOLDER [FOLDER ...]
#   FOLDER: a sweep folder, as "experiments/<name>" or just "<name>".
#
# Differences from run_benchmark_sweep.sh, for long campaigns:
# - a config whose output_dir already has results.json is skipped, so the
#   script can be relaunched after a stop without redoing finished runs;
# - after each successful run, that run's Qdrant collection is deleted.
#   Nothing reads it afterwards (the analyses read results/ only, and every
#   run recreates its collection from scratch). Left in place, hundreds of
#   collections pile up and Qdrant eventually fails with RocksDB IO errors
#   ("While open a file for lock"), as on 17-19 November 2026;
# - if a run fails, Qdrant is restarted, the script waits until it is ready
#   and retries that run once; it stops at the second failure.
#
# Environment: QDRANT_URL (default http://localhost:6333),
#              QDRANT_CONTAINER (default eiger-qdrant).
# Output lines: "=== <config> <time>", "RETRY ...", "FAILED ...", "FINE <time>".
set -uo pipefail
cd "$(dirname "$0")/.."
[[ $# -ge 1 ]] || { echo "usage: $0 FOLDER [FOLDER ...]"; exit 2; }
if [[ -n "${VIRTUAL_ENV:-}" ]]; then PY="$VIRTUAL_ENV/bin/python"; else PY="venv-$(hostname -s)/bin/python"; fi
[[ -x "$PY" ]] || { echo "ERROR: $PY not found. Run scripts/install_locked.sh first."; exit 1; }
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-eiger-qdrant}"

yaml_field() {
  sed -n "s/^[[:space:]]*$1:[[:space:]]*//p" "$2" | head -1 | tr -d "\"'"
}

qdrant_ready() {
  local i
  for i in $(seq 1 60); do
    [[ "$(curl -s "$QDRANT_URL/readyz")" == "all shards are ready" ]] && return 0
    sleep 5
  done
  return 1
}

run_one() {
  "$PY" -m eiger run "$1" > "$2" 2>&1 < /dev/null
}

configs=()
for d in "$@"; do
  d="${d#experiments/}"; d="${d%/}"
  [[ -d "experiments/$d" ]] || { echo "ERROR: experiments/$d not found."; exit 1; }
  for y in "experiments/$d"/*.yaml; do configs+=("$y"); done
done

todo=0
for y in "${configs[@]}"; do
  out="$(yaml_field output_dir "$y")"
  [[ -f "${out%/}/results.json" ]] || todo=$((todo + 1))
done
echo "configurazioni: ${#configs[@]}, da fare: $todo"

for y in "${configs[@]}"; do
  out="$(yaml_field output_dir "$y")"
  [[ -f "${out%/}/results.json" ]] && continue
  folder="$(basename "$(dirname "$y")")"
  name="$(basename "$y" .yaml)"
  log_dir="logs/$folder"; mkdir -p "$log_dir"
  log="$log_dir/$name.log"
  echo "=== $y $(date +%H:%M:%S)"
  if ! run_one "$y" "$log"; then
    echo "RETRY $y dopo riavvio Qdrant $(date +%H:%M:%S)"
    docker restart "$QDRANT_CONTAINER" > /dev/null < /dev/null
    qdrant_ready || { echo "FAILED Qdrant non pronto"; exit 1; }
    sleep 10
    run_one "$y" "$log" || { echo "FAILED $y (vedi $log)"; tail -5 "$log"; exit 1; }
  fi
  coll="$(yaml_field collection_name "$y")"
  [[ -n "$coll" ]] && curl -s -X DELETE "$QDRANT_URL/collections/$coll" > /dev/null < /dev/null
done
echo "FINE $(date +%H:%M:%S)"
