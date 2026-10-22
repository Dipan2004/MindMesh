#!/usr/bin/env bash
# Re-run the ladder measurements with the FIXED OllamaClient.
#
# The first ladder run measured a broken client, not the ladder: the
# 2026-10-22 refactor that added frequency_penalty/cache_tag/with_overrides
# dropped `prompt=prompt` from the request, so every call went out with an
# empty prompt and came back empty. 42/42 chunks failed in 9.6 s.
# (Fixed in src/graphrag_lite/llm/ollama_client.py; regression test in
# tests/test_ollama_client_prompt.py.)
#
# Order matters: the model is bistable with respect to the server's numeric
# state, so the runs are sequential, never concurrent.
set -u
cd "$(dirname "$0")/.."

echo "=== [1/2] corpus run through the production path WITH the ladder ==="
python tests/exp_reextract_corpus_ladder.py
echo "ladder corpus exit=$?"

echo
echo "=== [2/2] hard-chunk ladder: does the ladder RESCUE? ==="
for rep in 0 1 2; do
  echo "--- repetition $rep ---"
  EXP_LADDER_CACHE_DIR="data/cache/exp_ladder_hard_r${rep}" \
  EXP_REP="${rep}" \
    python tests/exp_ladder_hard.py
  echo "--- repetition $rep exit=$? ---"
done

echo "ALL LADDER MEASUREMENTS COMPLETE"
