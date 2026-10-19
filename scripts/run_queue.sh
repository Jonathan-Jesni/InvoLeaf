#!/usr/bin/env bash
# Training queue for time-limited cloud sessions (about 3 hours a day).
#
#   SEED=42 bash scripts/run_queue.sh S C
#
# Run the SAME command at the start of every session. It works out what is left:
#   - runs that are trained AND evaluated are skipped,
#   - a run is only started if its estimated time fits in what is left of the budget;
#     one that does not fit is deferred, and later shorter runs fill the time instead,
#   - every run is evaluated straight after it trains, on the same machine.
#
# Set BUDGET_MIN to the minutes you actually have left when you start, minus ~15 for
# pushing results before the cutoff. The default of 165 assumes a fresh 3-hour session.
#
# Why everything big stays on local disk
# --------------------------------------
# On the cloud instances /workspace is a network disk whose reads and writes stall for
# minutes at a time under load. Saving a 1 GB resume checkpoint there froze one run for
# 15 minutes, and one ViT-B epoch that normally takes 39 s took 22 minutes reading
# images. So at the start of a session the images are copied to the machine's own disk
# (LOCAL_DIR), and checkpoints are written there too. Only the small result JSONs and
# logs go to /workspace.
#
# The cost is that local disk is wiped when the session ends, so a run cannot carry
# over to the next session. That is why runs are only started when they fit: every run
# here takes well under one session. A run that does not finish in time simply starts
# again next session; its partial state is never trusted.
#
# Splitting work between people is BY SEED, not by model:
#   person 1:  SEED=42 bash scripts/run_queue.sh S C
#   person 2:  SEED=43 bash scripts/run_queue.sh S E
#   person 3:  SEED=44 bash scripts/run_queue.sh S F
# A lost seed costs error bars; a lost model would cost the whole comparison.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SEED="${SEED:-42}"
BATCH="${BATCH:-64}"            # 48GB W7900 fits 64 for every model; check_env.py confirms
BUDGET_MIN="${BUDGET_MIN:-165}" # minutes this session may use, with margin before cutoff
# Never start an evaluation with less than this left. With images on local disk a full
# evaluation takes under a minute; a run whose evaluation is skipped loses its checkpoint
# at session end and has to be retrained, so this errs towards running it.
EVAL_MIN="${EVAL_MIN:-3}"
LOCAL_DIR="${LOCAL_DIR:-/root/involeaf_local}"   # the machine's own (non-network) disk
STAGE="${STAGE:-1}"             # 1 = copy images to LOCAL_DIR first; 0 = read in place
PAUSED_EXIT=3

LOGDIR="results/logs"
CKPT_DIR="${LOCAL_DIR:?}/checkpoints"
mkdir -p "$LOGDIR" "$CKPT_DIR"

START=$(date +%s)
STOP=0
FAILED=(); SKIPPED=(); RAN=(); PAUSED=(); DEFERRED=()

remaining () {                  # whole minutes left in this session's budget
  echo $(( BUDGET_MIN - ($(date +%s) - START) / 60 ))
}

# Conservative minutes for one run INCLUDING start-up tuning and its evaluation, from
# the timings measured on the W7900: ResNet ~12 min, ViT-B ~14, InvoLeaf ~45 (unfold),
# AgriTL ~25, ViT-S ~6, InvoLeaf-S K=3/5/7 ~9/14/21, pure ~41, parallel ~48, plus about
# a minute of evaluation each. Override one with e.g. EST_agritl_vit=30.
estimate () {
  local var="EST_$1"
  if [[ -n "${!var:-}" ]]; then echo "${!var}"; return; fi
  case "$1" in
    resnet50)             echo 20 ;;
    vit_b16_baseline)     echo 22 ;;
    involeaf_b16)         echo 60 ;;
    agritl_vit)           echo 40 ;;
    vit_s16_baseline)     echo 10 ;;
    abl_s16_k3)           echo 15 ;;
    abl_s16_k5)           echo 20 ;;
    abl_s16_k7)           echo 28 ;;
    abl_variant_pure)     echo 48 ;;
    abl_variant_parallel) echo 55 ;;
    benchmark)            echo 15 ;;
    abl_placement_every3|abl_placement_last4) echo 30 ;;
    abl_placement_last6)  echo 35 ;;
    abl_kernel_k3)        echo 35 ;;
    maize_resnet50|maize_vit_b16_baseline) echo 8 ;;
    maize_agritl_vit)     echo 12 ;;
    maize_involeaf_b16)   echo 15 ;;
    abl_*)                echo 55 ;;
    *)                    echo 45 ;;
  esac
}

stage_data () {                 # copy the images to local disk once per session
  [[ "$STAGE" == "1" ]] || { echo ">> data   STAGE=0, reading images from ./data"; return 0; }
  local dst="${LOCAL_DIR:?}/data"
  if [[ -f "${dst}/.staged" ]]; then
    echo ">> data   already on local disk (${dst})"
  else
    echo ">> data   copying images to local disk (${dst}), one-time per session ..."
    rm -rf "$dst" && mkdir -p "$dst"
    local t0; t0=$(date +%s)
    if ( cd data && tar --exclude=.git -cf - PlantVillage-Dataset PlantDoc-Dataset ) \
         | tar -C "$dst" -xf -; then
      touch "${dst}/.staged"
      echo "   copied $(find "$dst" -type f ! -name .staged | wc -l) files" \
           "in $(( $(date +%s) - t0 ))s"
    else
      echo "   copy FAILED - training will read images from the network disk instead"
      return 0
    fi
  fi
  # involeaf/utils/paths.py redirects every data/ root to this copy.
  export INVOLEAF_DATA_DIR="$dst"
}

train () {                      # train <config> <base-name> [KEY=VALUE ...]
  local cfg="$1" base="$2"; shift 2
  local name="${base}_seed${SEED}"
  [[ $STOP -eq 1 ]] && return 0

  if [[ -f "results/${name}.json" ]]; then
    if [[ -f "results/eval_${name}.json" || -f "${CKPT_DIR}/${name}.pth" ]]; then
      echo ">> skip   ${name} (finished)"; SKIPPED+=("$name"); return 0
    fi
    # Trained in an earlier session but never evaluated, and the checkpoint went with
    # that session's local disk. Without retraining it would be skipped forever and
    # never get robustness or PlantDoc numbers.
    echo ">> redo   ${name}: trained earlier but never evaluated, and its checkpoint is gone"
    mv "results/${name}.json" "results/${name}.json.stale"
  fi

  local left need
  left=$(remaining); need=$(estimate "$base")
  if (( left < need )); then
    echo ">> later  ${name}: needs ~${need} min, ${left} left - trying shorter runs"
    DEFERRED+=("$name"); return 0
  fi

  echo ">> start  ${name}  ($(date +%H:%M), ${left} min left, needs ~${need})"
  # The run name is <base>_seed<SEED>, so --set name=<base> keeps the results file,
  # the checkpoint and the skip check above all agreeing on one filename.
  python scripts/train.py --config "$cfg" --resume --max-minutes "$left" \
    --set "name=${base}" "seed=${SEED}" "batch_size=${BATCH}" \
          "checkpoint_dir=${CKPT_DIR}" "$@" \
    >> "${LOGDIR}/${name}.log" 2>&1
  local code=$?

  if [[ $code -eq 0 ]]; then
    echo "   done   ${name}  ($(date +%H:%M))"; RAN+=("$name")
  elif [[ $code -eq $PAUSED_EXIT ]]; then
    # Only happens if a run badly overruns its estimate. Its state is on local disk,
    # so it restarts from scratch next session.
    echo "   paused ${name} - ran over its estimate, restarts next session"
    PAUSED+=("$name"); STOP=1
  else
    echo "   FAILED ${name} - see ${LOGDIR}/${name}.log"
    tail -n 15 "${LOGDIR}/${name}.log" | sed 's/^/     /'
    FAILED+=("$name")
  fi
}

evaluate () {                   # evaluate <base-name> [eval_robustness.py args ...]
  local name="${1}_seed${SEED}"; shift
  [[ $STOP -eq 1 ]] && return 0
  if [[ -f "results/eval_${name}.json" ]]; then
    echo ">> skip   eval ${name} (done)"; SKIPPED+=("eval:${name}"); return 0
  fi
  # Only evaluate a finished run: the best-so-far checkpoint exists from epoch 0 on.
  [[ -f "results/${name}.json" ]] || return 0

  local ckpt="${CKPT_DIR}/${name}.pth"
  [[ -f "$ckpt" ]] || ckpt="results/checkpoints/${name}.pth"   # older layout
  if [[ ! -f "$ckpt" ]]; then
    echo ">> skip   eval ${name} (checkpoint is not on this machine)"
    SKIPPED+=("eval:${name}"); return 0
  fi
  local left; left=$(remaining)
  if (( left < EVAL_MIN )); then
    echo ">> stop   ${left} min left, not starting eval ${name}"; STOP=1; return 0
  fi
  echo ">> eval   ${name}  ($(date +%H:%M))"
  if python scripts/eval_robustness.py --ckpt "$ckpt" "$@" \
       > "${LOGDIR}/eval_${name}.log" 2>&1; then
    echo "   done   eval ${name}"; RAN+=("eval:${name}")
  else
    echo "   FAILED eval ${name} - see ${LOGDIR}/eval_${name}.log"
    tail -n 15 "${LOGDIR}/eval_${name}.log" | sed 's/^/     /'
    FAILED+=("eval:${name}")
  fi
}

run () {                        # run <config> <base-name> [KEY=VALUE ...]: train, then eval
  train "$@"
  evaluate "$2"
}

# --------------------------------------------------------------------------------- #

phase_S () {
  echo "=== Phase S - four headline models at seed ${SEED} ==="
  run configs/resnet50.yaml          resnet50
  run configs/vit_b16_baseline.yaml  vit_b16_baseline
  run configs/involeaf_b16.yaml      involeaf_b16
  run configs/agritl_vit.yaml        agritl_vit
}

phase_C () {
  echo "=== Phase C - kernel-size ablation on ViT-S/16 ==="
  # Evaluated as well as trained: clean accuracy is saturated near 100%, so whether a
  # smaller kernel helps can only show up in the robustness and PlantDoc numbers.
  run configs/vit_s16_baseline.yaml vit_s16_baseline
  for k in 3 5 7; do
    run configs/involeaf_s16.yaml "abl_s16_k${k}" "kernel_size=${k}"
  done
}

phase_E () {
  echo "=== Phase E - variant ablation ==="
  run configs/ablations/variant_pure.yaml     abl_variant_pure
  run configs/ablations/variant_parallel.yaml abl_variant_parallel
}

phase_P () {
  echo "=== Phase P - where involution goes (ViT-B) ==="
  # Planned before any results. first11 drops the last block's involution, which never
  # trains (no gradient reaches it; see configs/ablations/placement_first11.yaml).
  run configs/ablations/placement_every3.yaml  abl_placement_every3
  run configs/ablations/placement_last4.yaml   abl_placement_last4
  run configs/ablations/placement_last6.yaml   abl_placement_last6
  run configs/ablations/placement_first11.yaml abl_placement_first11
}

phase_D () {
  echo "=== Phase D - design choices of the operator (ViT-B) ==="
  # First: is InvoLeaf's colour/field gap involution, or its FFN training at the
  # new-module rate (10x the baseline MLP's)? See configs/ablations/ffn_lr_backbone.yaml.
  run configs/ablations/ffn_lr_backbone.yaml abl_ffn_lr_backbone
  run configs/ablations/no_delta_init.yaml  abl_no_delta_init
  run configs/ablations/kernel_k3.yaml      abl_kernel_k3
  run configs/ablations/norm_bn.yaml        abl_norm_bn
  run configs/ablations/ffninit_random.yaml abl_ffninit_random
  run configs/ablations/groups_gc64.yaml    abl_groups_gc64
  run configs/ablations/groups_gc768.yaml   abl_groups_gc768
}

phase_M () {
  echo "=== Phase M - second crop: PlantVillage maize, four headline models ==="
  # Same protocol on a different crop (AgriTL-ViT reports 97.2% on maize). PlantDoc is
  # tomato-only, so the cross-domain test is skipped: its labels are tomato classes.
  local m
  for m in resnet50 vit_b16_baseline involeaf_b16 agritl_vit; do
    train "configs/${m}.yaml" "maize_${m}" "split_file=splits/plantvillage_maize.json"
    evaluate "maize_${m}" --skip-cross-domain
  done
}

phase_F () {
  echo "=== Phase F - efficiency benchmark (one machine only) ==="
  [[ $STOP -eq 1 ]] && return 0
  if [[ -f results/benchmark.json ]]; then
    echo ">> skip   benchmark (done)"; SKIPPED+=("benchmark"); return 0
  fi
  local left need; left=$(remaining); need=$(estimate benchmark)
  if (( left < need )); then
    echo ">> later  benchmark: needs ~${need} min, ${left} left"; DEFERRED+=("benchmark")
    return 0
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
  C  ViT-S baseline + kernel-size ablation, each evaluated
  E  pure / parallel variant ablation, each evaluated
  F  efficiency benchmark (one person only)
  P  placement ablation on ViT-B (every3, last4, last6, first11)
  D  design ablations on ViT-B (FFN at the pretrained LR, no delta init, K=3,
     BatchNorm, random FFN init, group size 64 and 768)
  M  maize: ResNet-50, ViT-B, InvoLeaf, AgriTL-ViT on a second crop

Run the same command at the start of every session; it picks up what is left.
USAGE
  exit 1
fi

echo "seed=${SEED}  batch=${BATCH}  budget=${BUDGET_MIN} min  local=${LOCAL_DIR}  started $(date)"
stage_data

for phase in "$@"; do
  case "$phase" in
    S|s) phase_S ;; C|c) phase_C ;; E|e) phase_E ;; F|f) phase_F ;;
    P|p) phase_P ;; D|d) phase_D ;; M|m) phase_M ;;
    *) echo "unknown phase '${phase}' - expected S, C, E, F, P, D or M"; exit 1 ;;
  esac
done

ELAPSED=$(( $(date +%s) - START ))
echo
echo "=== Summary ================================================================"
printf 'elapsed   %dh%02dm of a %d min budget\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) "$BUDGET_MIN"
printf 'ran       %d  %s\n' "${#RAN[@]}"      "${RAN[*]-}"
printf 'deferred  %d  %s\n' "${#DEFERRED[@]}" "${DEFERRED[*]-}"
printf 'paused    %d  %s\n' "${#PAUSED[@]}"   "${PAUSED[*]-}"
printf 'skipped   %d  %s\n' "${#SKIPPED[@]}"  "${SKIPPED[*]-}"
printf 'failed    %d  %s\n' "${#FAILED[@]}"   "${FAILED[*]-}"
echo
if (( ${#DEFERRED[@]} + ${#PAUSED[@]} > 0 )); then
  echo "Some runs did not fit this session. Next session, run exactly the same command."
else
  echo "Nothing left for these phases."
fi
echo "Push finished results now:"
echo "  git add results/*seed${SEED}*.json && git commit -m 'results: seed ${SEED}' && git push"
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
