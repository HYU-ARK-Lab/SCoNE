#!/usr/bin/env bash
# Controlled-noise sweep on HotpotQA: 3 gold positions x 4 noise levels, run SEPARATELY.
# Same questions (fixed --seed --n_examples) across every cell; only the context changes.
#
#   bash scripts/run_distractor_noise.sh <tag> '<neurons_json>'
# e.g.
#   bash scripts/run_distractor_noise.sh base  '[]'   # vanilla-RAG-under-noise control
#   bash scripts/run_distractor_noise.sh scone '[[27,8140],[13,2158],[21,12666],[30,3382],[30,5035]]'
#   bash scripts/run_distractor_noise.sh ircan '[[27,12776],[27,9228]]'
#
# Writes $OUTDIR/{tag}_n0.json (position is meaningless at N=0, so it runs once) and
# $OUTDIR/{tag}_{first,shuffle,last}_n{2,4,8}.json -- the names collect_noise_results.py reads.
# Run both tags, then: python scripts/collect_noise_results.py --dir "$OUTDIR"
set -euo pipefail

TAG="${1:?usage: run_distractor_noise.sh <tag> '<neurons_json>'}"
NEURONS="${2:?pass neurons list, e.g. '[]' for base}"
PY="${PY:-python}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
SPLIT="${SPLIT:-validation}"
N_EXAMPLES="${N_EXAMPLES:-1000}"
SEED="${SEED:-0}"
STRENGTH="${STRENGTH:-7.0}"
OUTDIR="${OUTDIR:-results/noise}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTDIR"

run() {  # run <n_distractors> <gold_pos> <out>
  echo "================ $TAG  N=$1  gold_pos=$2 ================"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" "$HERE/distractor_noise_eval.py" \
      --model_path "$MODEL" \
      --neurons "$NEURONS" \
      --enhance_strength "$STRENGTH" \
      --n_distractors "$1" \
      --gold_pos "$2" \
      --split "$SPLIT" \
      --n_examples "$N_EXAMPLES" \
      --seed "$SEED" \
      --out "$3"
}

# N=0: gold only -> position has no effect, one run shared by all three tables
run 0 shuffle "$OUTDIR/${TAG}_n0.json"

for POS in first shuffle last; do
  for N in 2 4 8; do
    run "$N" "$POS" "$OUTDIR/${TAG}_${POS}_n${N}.json"
  done
done

echo "== summary: $TAG =="
for F in "$OUTDIR/${TAG}_n0.json" "$OUTDIR/${TAG}_"{first,shuffle,last}"_n"{2,4,8}".json"; do
  "$PY" -c "import json,sys;s=json.load(open(sys.argv[1]))['summary'];print(f\"{sys.argv[1]}  M={s['M']:.4f}  EM={s['EM']:.4f}  F1={s['F1']:.4f}  n={s['n_examples']}\")" "$F"
done
