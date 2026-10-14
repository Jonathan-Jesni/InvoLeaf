#!/usr/bin/env bash
# Training queue for time-limited cloud sessions (about 3 hours a day).
#
#   SEED=42 bash scripts/run_queue.sh S C
#
# Run the SAME command at the start of every session. It picks up where the last
# session stopped:
#   - finished runs are skipped (their results/<run>.json exists),
#   - an unfinished run resumes from its last completed epoch (<run>.last.pth),
#   - nothing new is started once the session budget is nearly spent.
#
# Set BUDGET_MIN to the minutes you actually have left when you start, with some
# margin for the session cutoff. The default of 165 assumes a fresh 3-hour session.
#
# Resuming needs results/checkpoints/ to survive between sessions. If your instance
# wipes its disk at the end of a session, resuming cannot work; check this before
# relying on it.
#
# Splitting work between people is BY SEED, not by model:
#   person 1:  SEED=42 bash scripts/run_queue.sh S C
#   person 2:  SEED=43 bash scripts/run_queue.sh S E
#   person 3:  SEED=44 bash scripts/run_queue.sh S F
# A lost seed costs error bars; a lost model would cost the whole comparison.
#
# Each model is evaluated right after it finishes training, on the same instance,
# because evaluation needs the checkpoint and checkpoints (~350MB each) never leave the
# box. Only the small result JSONs are committed and merged.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SEED="${SEED:-42}"
BATCH="${BATCH:-64}"            # 48GB W7900 fits 64 for every model; check_env.py confirms
BUDGET_MIN="${BUDGET_MIN:-165}" # minutes this session may use, with margin before cutoff
MIN_START="${MIN_START:-15}"    # never start a training run with less than this left
EVAL_MIN="${EVAL_MIN:-10}"      # rough cost of one robustness evaluation
PAUSED_EXIT=3

LOGDIR="results/logs"
mkdir -p "$LOGDIR" results/checkpoints

START=$(date +%s)
STOP=0
FAILED=(); SKIPPED=(); RAN=(); PAUSED=()

remaining () {                  # whole minutes left in this session's budget
  echo $(( BUDGET_MIN - ($(date +%s) - START) / 60 ))
}

train () {                      # train <config> <base-name> [KEY=VALUE ...]
  local cfg="$1" base="$2"; shift 2
  local name="${base}_seed${SEED}"
  [[ $STOP -eq 1 ]] && return 0
  if [[ -f "results/${name}.json" ]]; then
    echo ">> skip   ${name} (finished)"; SKIPPED+=("$name"); return 0
  fi
  local left; left=$(remaining)
  if (( left < MIN_START )); then
    echo ">> stop   ${left} min left, not starting ${name}"; STOP=1; return 0
  fi

  local how="start"
  [[ -f "results/checkpoints/${name}.last.pth" ]] && how="resume"
  echo ">> ${how}  ${name}  ($(date +%H:%M), ${left} min left)"

  # The run name is <name>_seed<SEED>, so --set name=<base> keeps the results file,
  # the checkpoint and the queue's skip check all agreeing on one filename.
  python scripts/train.py --config "$cfg" --resume --max-minutes "$left" \
    --set "name=${base}" "seed=${SEED}" "batch_size=${BATCH}" "$@" \
    >> "${LOGDIR}/${name}.log" 2>&1
  local code=$?

  if [[ $code -eq 0 ]]; then
    echo "   done   ${name}  ($(date +%H:%M))"; RAN+=("$name")
  elif [[ $code -eq $PAUSED_EXIT ]]; then
    echo "   paused ${name} - resumes next session"; PAUSED+=("$name"); STOP=1
  else
    echo "   FAILED ${name} - see ${LOGDIR}/${name}.log"
    tail -n 15 "${LOGDIR}/${name}.log" | sed 's/^/     /'
    FAILED+=("$name")
  fi
}

evaluate () {                   # evaluate <base-name>
  local name="${1}_seed${SEED}"
  [[ $STOP -eq 1 ]] && return 0
  if [[ -f "results/eval_${name}.json" ]]; then
    echo ">> skip   eval ${name} (done)"; SKIPPED+=("eval:${name}"); return 0
  fi
  # Only evaluate a finished run. The best-so-far checkpoint exists from epoch 0 on,
  # so checking for the checkpoint alone would evaluate a half-trained model and then
  # skip it forever after.
  if [[ ! -f "results/${name}.json" ]]; then
    echo ">> skip   eval ${name} (training not finished)"; return 0
  fi
  if [[ ! -f "results/checkpoints/${name}.pth" ]]; then
    echo ">> skip   eval ${name} (checkpoint is not on this instance)"
    SKIPPED+=("eval:${name}"); return 0
  fi
  local left; left=$(remaining)
  if (( left < EVAL_MIN )); then
    echo ">> stop   ${left} min left, not starting eval ${name}"; STOP=1; return 0
  fi
  echo ">> eval   ${name}  ($(date +%H:%M))"
  if python scripts/eval_robustness.py --ckpt "results/checkpoints/${name}.pth" \
       > "${LOGDIR}/eval_${name}.log" 2>&1; then
    echo "   done   eval ${name}"; RAN+=("eval:${name}")
  else
    echo "   FAILED eval ${name} - see ${LOGDIR}/eval_${name}.log"
    tail -n 15 "${LOGDIR}/eval_${name}.log" | sed 's/^/     /'
    FAILED+=("eval:${name}")
  fi
}

# --------------------------------------------------------------------------------- #

phase_S () {
  echo "=== Phase S - four headline models at seed ${SEED}, each evaluated on finish ==="
  train configs/resnet50.yaml          resnet50;          evaluate resnet50
  train configs/vit_b16_baseline.yaml  vit_b16_baseline;  evaluate vit_b16_baseline
  train configs/involeaf_b16.yaml      involeaf_b16;      evaluate involeaf_b16
  train configs/agritl_vit.yaml        agritl_vit;        evaluate agritl_vit
}

phase_C () {
  echo "=== Phase C - kernel-size ablation on ViT-S/16 ============================="
  train configs/vit_s16_baseline.yaml vit_s16_baseline
  for k in 3 5 7; do
    train configs/involeaf_s16.yaml "abl_s16_k${k}" "kernel_size=${k}"
  done
}

phase_E () {
  echo "=== Phase E - variant ablation ============================================="
  train configs/ablations/variant_pure.yaml     abl_variant_pure;     evaluate abl_variant_pure
  train configs/ablations/variant_parallel.yaml abl_variant_parallel; evaluate abl_variant_parallel
}

phase_F () {
  echo "=== Phase F - efficiency benchmark (one instance only) ====================="
  [[ $STOP -eq 1 ]] && return 0
  if [[ -f results/benchmark.json ]]; then
    echo ">> skip   benchmark (done)"; SKIPPED+=("benchmark"); return 0
  fi
  if (( $(remaining) < MIN_START )); then
    echo ">> stop   not enough time left for the benchmark"; STOP=1; return 0
  fi
  echo ">> bench  params / FLOPs / latency / memory"
  if python scripts/benchmark.py --all --train-batch "${BATCH}" \
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
usage: SEED=<n> [BUDGET_MIN=<minutes>] bash scripts/run_queue.sh <phase> [phase...]

  S  four headline models at SEED, each evaluated when it finishes
  C  ViT-S baseline + kernel-size ablation
  E  pure / parallel variant ablation
  F  efficiency benchmark (one person only)

Run the same command at the start of every session; it resumes where it stopped.
USAGE
  exit 1
fi

echo "seed=${SEED}  batch=${BATCH}  budget=${BUDGET_MIN} min  started $(date)"

for phase in "$@"; do
  case "$phase" in
    S|s) phase_S ;; C|c) phase_C ;; E|e) phase_E ;; F|f) phase_F ;;
    *) echo "unknown phase '${phase}' - expected S, C, E or F"; exit 1 ;;
  esac
done

ELAPSED=$(( $(date +%s) - START ))
echo
echo "=== Summary ================================================================"
printf 'elapsed  %dh%02dm of a %d min budget\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) "$BUDGET_MIN"
printf 'ran      %d  %s\n' "${#RAN[@]}"     "${RAN[*]-}"
printf 'paused   %d  %s\n' "${#PAUSED[@]}"  "${PAUSED[*]-}"
printf 'skipped  %d  %s\n' "${#SKIPPED[@]}" "${SKIPPED[*]-}"
printf 'failed   %d  %s\n' "${#FAILED[@]}"  "${FAILED[*]-}"
echo
if [[ $STOP -eq 1 ]]; then
  echo "Session budget reached. Next session, run exactly the same command again."
else
  echo "Queue complete for these phases."
fi
echo "Push finished results now so teammates can see them:"
echo "  git add results/*.json && git commit -m 'results: seed ${SEED}' && git push"
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
