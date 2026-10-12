#!/usr/bin/env bash
# Ordered training queue for the W7900 cloud instance.
#
#   bash scripts/run_queue.sh A          # one phase
#   bash scripts/run_queue.sh A B C      # several, in order
#   BATCH=32 bash scripts/run_queue.sh A # override the batch size
#
# Phases are ordered by what you lose if you run out of time, not by the order the
# report presents them. Phase A alone produces a complete four-model comparison at one
# seed; phase D only improves its error bars. Never start D before B is done.
#
# The queue is idempotent: a run whose results file already exists is skipped, so if the
# instance drops you can re-run the same command and it resumes. Delete the results JSON
# to force a re-run.
#
# Logs go to results/logs/<run>.log. Failures do not stop the queue - they are collected
# and reported at the end, so one OOM does not cost you the whole night.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BATCH="${BATCH:-64}"          # 48 GB fits 64 for every model here; check_env.py confirms
SEEDS_EXTRA="${SEEDS_EXTRA:-43 44}"
LOGDIR="results/logs"
mkdir -p "$LOGDIR" results/checkpoints

FAILED=()
SKIPPED=()
RAN=()

train () {                     # train <config> <run-name> [extra args...]
  local cfg="$1" name="$2"; shift 2
  if [[ -f "results/${name}.json" ]]; then
    echo ">> skip   ${name} (results/${name}.json exists)"
    SKIPPED+=("$name"); return 0
  fi
  echo ">> train  ${name}"
  if python scripts/train.py --config "$cfg" --set "batch_size=${BATCH}" \
       "$@" > "${LOGDIR}/${name}.log" 2>&1; then
    echo "   done   ${name}"
    RAN+=("$name")
  else
    echo "   FAILED ${name} - see ${LOGDIR}/${name}.log"
    tail -n 15 "${LOGDIR}/${name}.log" | sed 's/^/     /'
    FAILED+=("$name")
  fi
}

evaluate () {                  # evaluate <checkpoint-stem>
  local stem="$1"
  if [[ ! -f "results/checkpoints/${stem}.pth" ]]; then
    echo ">> skip   eval ${stem} (no checkpoint)"; SKIPPED+=("eval:${stem}"); return 0
  fi
  if [[ -f "results/eval_${stem}.json" ]]; then
    echo ">> skip   eval ${stem} (already evaluated)"; SKIPPED+=("eval:${stem}"); return 0
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

phase_A () {
  echo "=== Phase A - the four headline models at seed 42 ==========================="
  echo "This is the minimum publishable result. Everything else is refinement."
  train configs/resnet50.yaml            resnet50_seed42
  train configs/vit_b16_baseline.yaml    vit_b16_baseline_seed42
  train configs/involeaf_b16.yaml        involeaf_b16_seed42
  train configs/agritl_vit.yaml          agritl_vit_seed42
}

phase_B () {
  echo "=== Phase B - robustness, resolution, cross-domain ========================="
  echo "Inference only on existing checkpoints. Cheap, and it is where the claim lives."
  evaluate resnet50_seed42
  evaluate vit_b16_baseline_seed42
  evaluate involeaf_b16_seed42
  evaluate agritl_vit_seed42
}

phase_C () {
  echo "=== Phase C - kernel-size ablation on ViT-S/16 ============================="
  echo "Answers the 'does involution just duplicate attention?' question either way."
  train configs/vit_s16_baseline.yaml    vit_s16_baseline_seed42
  for k in 3 5 7; do
    train configs/involeaf_s16.yaml "abl_s16_k${k}_seed42" --set "kernel_size=${k}"
  done
}

phase_D () {
  echo "=== Phase D - extra seeds for the headline models =========================="
  echo "Turns single numbers into mean +/- std. Do not start this before B is done."
  for s in $SEEDS_EXTRA; do
    train configs/resnet50.yaml          "resnet50_seed${s}"          --set "seed=${s}"
    train configs/vit_b16_baseline.yaml  "vit_b16_baseline_seed${s}"  --set "seed=${s}"
    train configs/involeaf_b16.yaml      "involeaf_b16_seed${s}"      --set "seed=${s}"
    train configs/agritl_vit.yaml        "agritl_vit_seed${s}"        --set "seed=${s}"
  done
}

phase_E () {
  echo "=== Phase E - variant ablation ============================================="
  echo "Separates 'involution helped' from 'channel mixing mattered'."
  train configs/ablations/variant_pure.yaml     abl_variant_pure_seed42
  train configs/ablations/variant_parallel.yaml abl_variant_parallel_seed42
  evaluate abl_variant_pure_seed42
  evaluate abl_variant_parallel_seed42
}

# --------------------------------------------------------------------------------- #

if [[ $# -eq 0 ]]; then
  echo "usage: bash scripts/run_queue.sh A [B C D E]"
  echo "  A  four headline models, seed 42      (start here)"
  echo "  B  robustness + cross-domain eval     (cheap, high value)"
  echo "  C  kernel-size ablation on ViT-S      (fast)"
  echo "  D  seeds 43 and 44                    (error bars)"
  echo "  E  pure / parallel variants           (nice to have)"
  exit 1
fi

START=$(date +%s)
echo "batch_size=${BATCH}  started $(date)"

for phase in "$@"; do
  case "$phase" in
    A|a) phase_A ;; B|b) phase_B ;; C|c) phase_C ;; D|d) phase_D ;; E|e) phase_E ;;
    *) echo "unknown phase '${phase}' - expected one of A B C D E"; exit 1 ;;
  esac
done

python scripts/make_tables.py > "${LOGDIR}/make_tables.log" 2>&1 \
  && echo ">> tables regenerated into results/tables.md" \
  || echo ">> make_tables.py failed - see ${LOGDIR}/make_tables.log"

echo
echo "=== Summary ================================================================"
printf 'elapsed  %s\n' "$(printf '%dh%02dm' $((($(date +%s)-START)/3600)) $(((($(date +%s)-START)%3600)/60)))"
printf 'ran      %d  %s\n' "${#RAN[@]}"     "${RAN[*]-}"
printf 'skipped  %d  %s\n' "${#SKIPPED[@]}" "${SKIPPED[*]-}"
printf 'failed   %d  %s\n' "${#FAILED[@]}"  "${FAILED[*]-}"
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
