#!/bin/bash
# Improvement screens on the FINAL set-up (validation only: --skip_test, the test sets stay closed).
#   PYTHONPATH=. bash review/improve/run_screen.sh "ARM1 ARM2" "SEEDS" "COSTS" [P]
# ARM names map to extra flags in review/improve/arms.py. Runs go to logs_torch/improve/<run>_<ARM>.
set -u
ARMS=$1; SEEDS=$2; COSTS_=$3; P=${4:-6}
export EPISODES=${EPISODES:-8000} OUT=${OUT:-logs_torch/improve}
mkdir -p $OUT/_stdout
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python review/improve/arms.py "$ARMS" "$SEEDS" "$COSTS_" | xargs -P "$P" -L 1 bash -c 'echo "$(date +%H:%M) start $0"; python train_d4pg.py "$@" > "$OUT/_stdout/$0.log" 2>&1; echo "$(date +%H:%M) end $0 (exit $?)"'
echo "screen finished $(date)"
