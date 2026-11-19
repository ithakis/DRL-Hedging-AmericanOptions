#!/bin/bash
# Warm-start pilot (thread "warm start from the tuned rule"): residual on top of the tuned rule (b).
# Same markets / split / selection as the retrain block B (review/retrain/settings.py); rule (b) parameters
# are the VaR95-tuned ones on validation paths (review/results/retrain/baselines_B.json).
#   PYTHONPATH=. bash review/warmstart/run_pilot.sh "res05 res02" "1" 6   # variants, seeds, parallel runs
set -u
VARIANT=${1:-res05}; SEEDS=${2:-"1 2"}; P=${3:-6}
MARKETS=${MARKETS:-../DRL-Hedging-retrain/markets}
OUT=logs_torch/warmstart; mkdir -p $OUT/_stdout
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for V in $VARIANT; do VARIANT=$V; case $V in
  res05) EXTRA="--residual_scale 0.5 --actor_freeze_episodes 2000 --val_at_start" ;;
  res02) EXTRA="--residual_scale 0.2 --actor_freeze_episodes 2000 --val_at_start" ;;
esac
for seed in $SEEDS; do for c in 0.005 0.01 0.02; do
  case $c in 0.005) R=0.3,0.3,0.3;; 0.01) R=0.3,0.3,0.5;; 0.02) R=0.2,0.3,1.0;; esac
  name=HestonP_var_spread${c}_seed${seed}_${VARIANT}
  [ -f $OUT/$name/results.json ] && { echo "skip $name" >&2; continue; }
  ARGS=$(python -c "from review.retrain.settings import train_args; print(' '.join(train_args('B', $c, $seed, out='$OUT')))")
  ARGS=${ARGS/--run_name HestonP_var_spread${c}_seed${seed}/--run_name $name}
  echo "$name $ARGS --market_dir $MARKETS --warm_rule $R --n_train_paths 40000 $EXTRA"
done; done; done | xargs -P "$P" -L 1 bash -c 'echo "$(date +%H:%M) start $0"; \
  python train_d4pg.py "$@" > "logs_torch/warmstart/_stdout/$0.log" 2>&1; echo "$(date +%H:%M) end $0 (exit $?)"'
