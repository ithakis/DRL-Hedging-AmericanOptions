#!/bin/bash
# Pilot: swing v64 recipe on block B (Heston P, r=2%, mu=3%, lambda_v=0), VaR95, 3 costs x seeds 1-2,
# same flags as review/retrain/settings.py block B; 6 runs in parallel, 1 thread each.
set -u
SEEDS=${SEEDS:-"1 2"}
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p logs_torch/swing/_stdout
for seed in $SEEDS; do for c in 0.005 0.01 0.02; do
  name=HestonP_swing_var_spread${c}_seed${seed}
  [ -f logs_torch/swing/$name/results.json ] && continue
  echo "$name --process Heston --r 0.02 --mu 0.03 --lambda_v 0 --init_vol 0.3 --kappa 1.0 --theta 0.09 --volvol 0.3 --rho -0.7 --obj_func var --train_episodes 40000 --market_seed 1234 --val_seed 2025 --test_seed 654 --n_val 8192 --n_test 8192 --torch_threads 1 --spread $c --seed $seed --out logs_torch/swing --run_name $name"
done; done | xargs -P ${P:-6} -L 1 bash -c 'echo "$(date +%H:%M) start $0"; python train_swing.py "$@" > logs_torch/swing/_stdout/$0.log 2>&1; echo "$(date +%H:%M) end $0 (exit $?)"'
echo "pilot finished $(date)"
