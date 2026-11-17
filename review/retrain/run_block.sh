#!/bin/bash
# Train all runs of one block, P runs at a time (each run uses 1 thread).
#   PYTHONPATH=. bash review/retrain/run_block.sh A 4
# Finished runs (results.json exists) are skipped, so the script can be restarted after a stop.
set -u
BLOCK=${1:?block A or B}; P=${2:-4}
mkdir -p logs_torch/retrain/_stdout
python - "$BLOCK" <<'PY' > "logs_torch/retrain/_jobs_$BLOCK.txt"
import sys, shlex
from review.retrain.settings import BLOCKS, COSTS, train_args, run_name
b = sys.argv[1]
# seed-major order: all 3 costs of seed 1 first, so partial results cover every cost
for seed in BLOCKS[b]["seeds"]:
    for c in COSTS:
        print(run_name(b, c, seed) + "\t" + " ".join(shlex.quote(x) for x in train_args(b, c, seed)))
PY
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
while IFS=$'\t' read -r name args; do
  if [ -f "logs_torch/retrain/$name/results.json" ]; then echo "skip $name (done)" >&2; continue; fi
  printf '%s\t%s\n' "$name" "$args"
done < "logs_torch/retrain/_jobs_$BLOCK.txt" | \
  xargs -P "$P" -L 1 bash -c 'echo "$(date +%H:%M) start $0"; \
    python train_d4pg.py "$@" > "logs_torch/retrain/_stdout/$0.log" 2>&1; \
    echo "$(date +%H:%M) end $0 (exit $?)"'
echo "block $BLOCK finished $(date)"
