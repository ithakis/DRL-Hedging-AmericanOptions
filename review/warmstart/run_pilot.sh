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
  # pilot 2: slower, conservative actor (pilot 1 drifted away from the rule as soon as the actor learned)
  res02lr) EXTRA="--residual_scale 0.2 --actor_lr 1e-5 --actor_freeze_episodes 2000 --val_at_start --train_episodes 20000" ;;
  res05l2) EXTRA="--residual_scale 0.5 --actor_lr 1e-5 --residual_l2 2 --actor_freeze_episodes 2000 --val_at_start --train_episodes 20000" ;;
  # option 1: learned gate (trade the rule amount or not), train_gate.py
  gate0) EXTRA="--gate_warmup_episodes 2000 --margin 0 --train_episodes 20000" ;;
  gate05) EXTRA="--gate_warmup_episodes 2000 --margin 0.5 --train_episodes 20000" ;;
  # batch A (no warm start): training speed / stability recipe, 20k episodes, validation every 500
  A1) EXTRA="--n_envs 16 --tau 0.005 --reward_scale 0.1 --ema_decay 0.999 --lr 6e-4 --actor_lr 3e-4 --train_episodes 20000 --val_every 500" ;;
  A2) EXTRA="--n_envs 16 --tau 0.005 --reward_scale 0.1 --ema_decay 0.999 --lr 6e-4 --actor_lr 3e-4 --batch_size 128 --samples_per_insert 64 --train_episodes 20000 --val_every 500" ;;
  # batch B on the PR #3 recipe (40k episodes), one change each; L80 = PR #3 recipe for 80k episodes (paths reused twice)
  B1) EXTRA="--extra_obs" ;;
  B2) EXTRA="--train_obj cvar" ;;
  B3) EXTRA="--tail_quantiles" ;;
  L80) EXTRA="--train_episodes 80000" ;;
  # batch D pilot: expected target over the QE step (9 nodes, passes the fidelity test) vs 1-step sampled ablation
  K9) EXTRA="--kernel_mx 3 --table_dir ../DRL-Hedging-retrain/pricing_tables --n_envs 16 --train_episodes 20000 --val_every 500" ;;
  S1B1) EXTRA="--extra_obs --n_step 1 --n_envs 16 --train_episodes 20000 --val_every 500" ;;
  K9B1) EXTRA="--extra_obs --kernel_mx 3 --table_dir ../DRL-Hedging-retrain/pricing_tables --n_envs 16 --train_episodes 12000 --val_every 500" ;;
  S1) EXTRA="--n_step 1 --n_envs 16 --train_episodes 20000 --val_every 500" ;;
  *) echo "unknown variant $V" >&2; exit 1 ;;
esac
for seed in $SEEDS; do for c in 0.005 0.01 0.02; do
  case $c in 0.005) R=0.3,0.3,0.3;; 0.01) R=0.3,0.3,0.5;; 0.02) R=0.2,0.3,1.0;; esac
  name=HestonP_var_spread${c}_seed${seed}_${VARIANT}
  [ -f $OUT/$name/results.json ] && { echo "skip $name" >&2; continue; }
  ARGS=$(python -c "from review.retrain.settings import train_args; print(' '.join(train_args('B', $c, $seed, out='$OUT')))")
  ARGS=${ARGS/--run_name HestonP_var_spread${c}_seed${seed}/--run_name $name}
  SCRIPT=train_d4pg.py; case $V in gate*) SCRIPT=train_gate.py;; esac
  WR="--warm_rule $R"; case $V in A*|B*) WR="";; L*|K*|S*) WR="";; esac
  echo "$name $SCRIPT $ARGS --market_dir $MARKETS $WR --n_train_paths 40000 $EXTRA"
done; done; done | xargs -P "$P" -L 1 bash -c 'echo "$(date +%H:%M) start $0"; \
  python "$@" > "logs_torch/warmstart/_stdout/$0.log" 2>&1; echo "$(date +%H:%M) end $0 (exit $?)"'
