#!/usr/bin/env bash
# Wait for the corpus ladder run to finish, then measure the ladder's RESCUE
# rate on the four chunks that are known to be hard.
#
# The wait is sequential on purpose: the model is bistable with respect to the
# server's numeric state, so a concurrent request would perturb the very
# measurement being made.
set -u
cd "$(dirname "$0")/.."

echo "waiting for data/cache/exp_reextract_ladder.json ..."
for i in $(seq 1 720); do
  if [ -f data/cache/exp_reextract_ladder.json ]; then
    echo "corpus ladder run finished (waited $((i*10))s)"
    break
  fi
  sleep 10
done

if [ ! -f data/cache/exp_reextract_ladder.json ]; then
  echo "TIMEOUT waiting for the corpus ladder run"
  exit 1
fi

for rep in 0 1 2; do
  echo "=== hard-chunk ladder repetition $rep ==="
  EXP_LADDER_CACHE_DIR="data/cache/exp_ladder_hard_r${rep}" \
  EXP_REP="${rep}" \
    python tests/exp_ladder_hard.py
  echo "--- repetition $rep done ---"
done

echo "ALL HARD-CHUNK LADDER REPETITIONS COMPLETE"
