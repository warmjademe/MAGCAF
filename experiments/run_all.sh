#!/usr/bin/env bash
# Pilot-first orchestration for Option-A suite (4 baselines + MAGCAF-Net).
#
#  pilot    1 seed x (4 baselines + MAGCAF full + 4 MAGCAF ablation variants)
#           ~2h total. MUST-PASS gate before committing to Phase B.
#
#  phaseB   2 ADDITIONAL seeds x (4 baselines + MAGCAF full) for mean±std table.
#           Only run if `pilot` shows MAGCAF > best baseline by >=2pp acc.
#
#  phaseC   MAGCAF x 5 losses x 3 seeds (P3 class-imbalance ablation).
#  phaseD   re-uses `pilot` ablation + 2 extra seeds for mean±std.
#
#  aggregate    `python experiments/aggregate.py --runs runs --group <g>`
set -euo pipefail

PY="${PY:-/home/qyb/miniconda3/envs/discover_artifical_intelligence/bin/python}"
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUNS="${RUNS:-$ROOT/runs}"

cd "$ROOT"
mkdir -p "$RUNS"

PILOT_SEED=42
PHASE_B_SEEDS=(123 2024)         # 2 more seeds after pilot's seed 42
ALL_SEEDS=(42 123 2024)

run_once() {
  # $1 group   $2 model   $3 loss   $4 seed   $5 run-name   $6 extra args
  local group="$1"; local model="$2"; local loss="$3"; local seed="$4"
  local runname="$5"; local extra="${6:-}"
  local rdir="$RUNS/$group/$runname"
  if [ -f "$rdir/metrics.json" ]; then
    echo "[skip] $group/$runname"; return
  fi
  echo ">>> $group / $runname"
  $PY -m train.train_single \
      --model "$model" --loss "$loss" --seed "$seed" \
      --run-dir "$rdir" $extra 2>&1 | tee "$RUNS/$group/${runname}.log"
}

pilot() {
  # 3 baselines (seed 42)
  run_once pilot timesformer ce        $PILOT_SEED "timesformer__ce__s42"
  run_once pilot videomae    ce        $PILOT_SEED "videomae__ce__s42"
  run_once pilot resnet_tcn  ce        $PILOT_SEED "resnet_tcn__ce__s42"

  # MAGCAF full + 4 ablation variants (seed 42)
  run_once pilot magcaf      cb_focal  $PILOT_SEED "magcaf_full__s42"
  run_once pilot magcaf      cb_focal  $PILOT_SEED "magcaf_no_omega__s42"            "--ablate no_omega"
  run_once pilot magcaf      cb_focal  $PILOT_SEED "magcaf_no_magcaf__s42"           "--ablate no_magcaf"
  run_once pilot magcaf      cb_focal  $PILOT_SEED "magcaf_no_uw__s42"               "--ablate no_uw"
  run_once pilot magcaf      cb_focal  $PILOT_SEED "magcaf_no_magcaf_no_omega__s42"  "--ablate no_magcaf_no_omega"

  # Anchor: original Fusion-Attn-LSTM with CE and with CB-Focal (same signature)
  run_once pilot fusion_attn_lstm ce        $PILOT_SEED "fal_ce__s42"
  run_once pilot fusion_attn_lstm cb_focal  $PILOT_SEED "fal_cb_focal__s42"
}

phaseB() {
  for seed in "${PHASE_B_SEEDS[@]}"; do
    run_once phaseB timesformer ce       "$seed" "timesformer__ce__s${seed}"
    run_once phaseB videomae    ce       "$seed" "videomae__ce__s${seed}"
    run_once phaseB resnet_tcn  ce       "$seed" "resnet_tcn__ce__s${seed}"
    run_once phaseB magcaf      cb_focal "$seed" "magcaf_full__s${seed}"
  done
}

phaseC() {
  for seed in "${ALL_SEEDS[@]}"; do
    for loss in ce weighted_ce focal cb_focal ldam; do
      run_once phaseC magcaf "$loss" "$seed" "magcaf__${loss}__s${seed}"
    done
  done
}

phaseD() {
  for seed in "${PHASE_B_SEEDS[@]}"; do
    run_once phaseD magcaf cb_focal "$seed" "magcaf_full__s${seed}"
    run_once phaseD magcaf cb_focal "$seed" "magcaf_no_omega__s${seed}"           "--ablate no_omega"
    run_once phaseD magcaf cb_focal "$seed" "magcaf_no_magcaf__s${seed}"          "--ablate no_magcaf"
    run_once phaseD magcaf cb_focal "$seed" "magcaf_no_uw__s${seed}"              "--ablate no_uw"
  done
}

case "${1:-}" in
  pilot)  pilot ;;
  phaseB) phaseB ;;
  phaseC) phaseC ;;
  phaseD) phaseD ;;
  all)    pilot; phaseB; phaseC; phaseD ;;
  *) echo "usage: $0 {pilot|phaseB|phaseC|phaseD|all}"; exit 1 ;;
esac
