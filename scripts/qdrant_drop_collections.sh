#!/usr/bin/env bash
# Deletes the benchmark's Qdrant collections (name starting with a prefix,
# default "eibench_"). Safe between runs: the analyses read results/ only and
# every run recreates its own collection. Refuses while an "eiger run" is
# active, since that run's collection would disappear under it.
# Usage: scripts/qdrant_drop_collections.sh [PREFIX]      (lists, asks y/N)
#        scripts/qdrant_drop_collections.sh [PREFIX] --yes
# Environment: QDRANT_URL (default http://localhost:6333).
set -uo pipefail
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
PREFIX="eibench_"; YES=0
for a in "$@"; do
  if [[ "$a" == "--yes" ]]; then YES=1; else PREFIX="$a"; fi
done
if pgrep -f "eiger run" > /dev/null; then
  echo "ERROR: an 'eiger run' is active; stop it or wait for it to finish."; exit 1
fi
names="$(curl -s "$QDRANT_URL/collections" | python3 -c '
import json, sys
prefix = sys.argv[1]
for c in json.load(sys.stdin)["result"]["collections"]:
    if c["name"].startswith(prefix):
        print(c["name"])
' "$PREFIX")" || { echo "ERROR: Qdrant not reachable at $QDRANT_URL"; exit 1; }
n=$(printf '%s' "$names" | grep -c . || true)
echo "collezioni con prefisso '$PREFIX': $n"
[[ "$n" -eq 0 ]] && exit 0
if [[ "$YES" -ne 1 ]]; then
  read -r -p "cancellarle? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "nulla cancellato"; exit 0; }
fi
printf '%s\n' "$names" | while read -r c; do
  curl -s -X DELETE "$QDRANT_URL/collections/$c" > /dev/null
done
echo "cancellate: $n"
