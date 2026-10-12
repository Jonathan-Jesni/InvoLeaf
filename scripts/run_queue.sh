#!/usr/bin/env bash
# Training queue for the W7900 cloud instances.
#
# Three people have three separate instances. The work is split BY SEED, not by model:
#
#   person 1:  SEED=42 bash scripts/run_queue.sh S C
#   person 2:  SEED=43 bash scripts/run_queue.sh S E
#   person 3:  SEED=44 bash scripts/run_queue.sh S F
#
# Splitting by seed rather than by model is deliberate. If an instance dies mid-way you
# lose error bars, which is survivable; if you had split by model you would lose an
# entire arm of the comparison and have no result at all.
#
# Phase S is self-contained: it trains all four headline models at one seed and then
# evaluates them on the same instance. That matters because evaluation needs the
# checkpoints, and a ViT-B checkpoint is ~350MB - it never leaves the box it was trained
# on. Only the small result JSONs are committed and merged.
#
# The queue is idempotent: a run whose results file already exists is skipped, so if the
# instance drops you re-run the same command and it resumes. Pull your teammates' result
# JSONs BEFORE starting, not during - a mid-run pull makes the skip logic skip work you
# are already doing.
#
# Logs go to results/logs/. Failures do not stop the queue; they are collected and
# reported at the end, so one OOM does not cost you the night.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SEED="${SEED:-42}"
BATCH="${BATCH:-64}"          # 48GB fits 64 for every model here; check_env.py confirms
LOGDIR="results/logs"
mkdir -p "$LOGDIR" results/checkpoints

FAILED=(); SKIPPED=(); RAN=()

train () {                     # train <config> <run-name> [extra --set args...]
  local cfg="$1" name="$2"; shift 2
  if [[ -f "results/${name}.json" ]]; then
    echo ">> skip   ${name}"; SKIPPED+=("$name"); return 0
  fi
  echo ">> train  ${name}  ($(date +%H:%M))"
  if python scripts/train.py --config "$cfg" \
       --set "batch_size=${BATCH}" "seed=${SEED}" "$@" \
       > "${LOGDIR}/${name}.log" 2>&1; then
    echo "   done   ${name}  ($(date +%H:%M))"; RAN+=("$name")
  else
    echo "   FAILED ${name} - see ${LOGDIR}/${name}.log"
    tail -n 15 "${LOGDIR}/${name}.log" | sed 's/^/     /'
    FAILED+=("$name")
  fi
}

evaluate () {                  # evaluate <checkpoint-stem>
  local stem="$1"
  if [[ ! -f "results/checkpoints/${stem}.pth" ]]; then
    echo ">> skip   eval ${stem} (no checkpoint on this instance)"
    SKIPPED+=("eval:${stem}"); return 0
  fi
  if [[ -f "results/eval_${stem}.json" ]]; then
    echo ">> skip   eval ${stem}"; SKIPPED+=("eval:${stem}"); return 0
  fi
  echo ">> eval   ${stem}"
  if python scripts/eval_robustness.py --ckpt "results/checkpoints/${stem}.pth" \
       > "${LOGDIR}/eval_${stem}.log" 2>&1; then
    echo "   done   eval ${stem}"; RAN+=("eval:${stem}")
  else
    echo "   FAILED eval ${stem} - see ${LOGDIR}/eval_${stem}.log"
    tail -n 15 "${LOGDIR}/eval_${stem}.log" | sed 's/^/     /'
    FAILED+=("eval:${stem}")
  fi
}

# --------------------------------------------------------------------------------- #

phase_S () {
  echo "=== Phase S - all four headline models at seed ${SEED}, then evaluate ======"
  echo "Self-contained. Train first so a crash during eval costs nothing retrainable."
  train configs/resnet50.yaml          "resnet50_seed${SEED}"
  train configs/vit_b16_baseline.yaml  "vit_b16_baseline_seed${SEED}"
  train configs/involeaf_b16.yaml      "involeaf_b16_seed${SEED}"
  train configs/agritl_vit.yaml        "agritl_vit_seed${SEED}"

  echo "--- evaluating this instance's checkpoints ---"
  evaluate "resnet50_seed${SEED}"
  evaluate "vit_b16_baseline_seed${SEED}"
  evaluate "involeaf_b16_seed${SEED}"
  evaluate "agritl_vit_seed${SEED}"
}

phase_C () {
  echo "=== Phase C - kernel-size ablation on ViT-S/16 ============================="
  echo "Answers 'does involution just duplicate attention?' either way. Fast backbone."
  train configs/vit_s16_baseline.yaml "vit_s16_baseline_seed${SEED}"
  for k in 3 5 7; do
    train configs/involeaf_s16.yaml "abl_s16_k${k}_seed${SEED}" --set "kernel_size=${k}"
  done
}

phase_E () {
  echo "=== Phase E - variant ablation ============================================="
  echo "Separates 'involution helped' from 'channel mixing mattered'."
  train configs/ablations/variant_pure.yaml     "abl_variant_pure_seed${SEED}"
  train configs/ablations/variant_parallel.yaml "abl_variant_parallel_seed${SEED}"
  evaluate "abl_variant_pure_seed${SEED}"
  evaluate "abl_variant_parallel_seed${SEED}"
}

phase_F () {
  echo "=== Phase F - efficiency benchmark ========================================="
  echo "ONE instance only. Latency is device-specific; mixing machines in one table is"
  echo "meaningless even though all three are W7900s."
  if [[ -f results/benchmark.json ]]; then
    echo ">> skip   benchmark"; SKIPPED+=("benchmark")
  elif python scripts/benchmark.py --all --train-batch "${BATCH}" \
         > "${LOGDIR}/benchmark.log" 2>&1; then
    echo "   done   benchmark"; RAN+=("benchmark")
  else
    echo "   FAILED benchmark - see ${LOGDIR}/benchmark.log"
    tail -n 15 "${LOGDIR}/benchmark.log" | sed 's/^/     /'
    FAILED+=("benchmark")
  fi
}

# --------------------------------------------------------------------------------- #

if [[ $# -eq 0 ]]; then
  cat <<'USAGE'
usage: SEED=<n> bash scripts/run_queue.sh <phase> [phase...]

  S  four headline models at SEED, then evaluate them   <- everyone runs this
  C  ViT-S baseline + kernel-size ablation              <- person 1
  E  pure / parallel variant ablation                   <- person 2
  F  efficiency benchmark (params/FLOPs/latency)        <- person 3 only

suggested split:
  person 1:  SEED=42 bash scripts/run_queue.sh S C
  person 2:  SEED=43 bash scripts/run_queue.sh S E
  person 3:  SEED=44 bash scripts/run_queue.sh S F
USAGE
  exit 1
fi

START=$(date +%s)
echo "seed=${SEED}  batch_size=${BATCH}  started $(date)"

for phase in "$@"; do
  case "$phase" in
    S|s) phase_S ;; C|c) phase_C ;; E|e) phase_E ;; F|f) phase_F ;;
    *) echo "unknown phase '${phase}' - expected S, C, E or F"; exit 1 ;;
  esac
done

ELAPSED=$(( $(date +%s) - START ))
echo
echo "=== Summary ================================================================"
printf 'elapsed  %dh%02dm\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60))
printf 'ran      %d  %s\n' "${#RAN[@]}"     "${RAN[*]-}"
printf 'skipped  %d  %s\n' "${#SKIPPED[@]}" "${SKIPPED[*]-}"
printf 'failed   %d  %s\n' "${#FAILED[@]}"  "${FAILED[*]-}"
echo
echo "Next: commit and push this instance's result JSONs so the others can merge them."
echo "  git add results/*.json && git commit -m 'results: seed ${SEED}' && git push"
echo "Checkpoints and logs stay local - they are gitignored."
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
