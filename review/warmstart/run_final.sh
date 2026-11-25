#!/bin/bash
# FINAL run (validation-only choices): pre-commitment agent on the adi_trading market (consistent tau/252 clock),
# rule (b) re-tuned on that market's validation paths (review/results/warmstart/richer_rule_params_adi_trading_<cost>.json),
# b = 0.6 x that rule's validation VaR95, kernel ATM fix, 20k episodes, fresh test seed 4242, seeds 1-10.
#   PYTHONPATH=. bash review/warmstart/run_final.sh [SEEDS] [P]
set -u
#   screens (validation only): TAG=scr_b45 EXTRA="--precommit_b_factor 0.45" EPISODES=12000 OUT=logs_torch/screen bash ...
SEEDS=${1:-"1 2 3 4 5 6 7 8 9 10"}; P=${2:-6}
export TAG=${TAG:-FINAL} EXTRA=${EXTRA:-} EPISODES=${EPISODES:-20000} OUT=${OUT:-logs_torch/final}
mkdir -p $OUT/_stdout
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python - "$SEEDS" <<'PY' | xargs -P "$P" -L 1 bash -c 'echo "$(date +%H:%M) start $0"; python train_d4pg.py "$@" > "$OUT/_stdout/$0.log" 2>&1; echo "$(date +%H:%M) end $0 (exit $?)"'
import json, os, sys
from pathlib import Path
from review.retrain.settings import COSTS, train_args, run_name
seeds = [int(s) for s in sys.argv[1].split()]
TAG, OUT, EP = os.environ["TAG"], os.environ["OUT"], os.environ["EPISODES"]
extra = os.environ["EXTRA"].split()
bf = 0.6
if "--precommit_b_factor" in extra:  # screen option: b as a factor of the rule's validation VaR
    i = extra.index("--precommit_b_factor"); bf = float(extra[i + 1]); del extra[i:i + 2]
for s in seeds:
    for c in COSTS:
        rb = json.load(open(f"review/results/warmstart/richer_rule_params_adi_trading_{c}.json"))[str(c)]["rule_b"]
        name = run_name("B", c, s) + "_" + TAG
        if Path(f"{OUT}/{name}/results.json").exists():
            continue
        args = train_args("B", c, s, out=OUT)
        args[args.index("--run_name") + 1] = name
        args += ["--market_dir", "../DRL-Hedging-retrain/markets", "--n_train_paths", "40000",
                 "--pricing_backend", "adi_trading", "--table_dir", "../DRL-Hedging-retrain/pricing_tables",
                 "--extra_obs", "--kernel_mx", "3", "--kernel_atm_fix", "--n_envs", "16",
                 "--train_episodes", EP, "--val_every", "500", "--test_seed", "4242",
                 "--warm_rule", f"{rb['a']},{rb['b']},{rb['w']}", "--residual_scale", "0.5",
                 "--actor_freeze_episodes", "1000", "--band", str(rb["w"]),
                 "--actor_sizes", "64,64,64", "--critic_sizes", "64,64,64", "--activation", "swish3",
                 "--lr", "6e-4", "--actor_lr", "3e-4", "--ema_decay", "0.999", "--tau", "0.003",
                 "--noise_mode", "pre", "--sigma", "1.0", "--sigma_floor", "0.2",
                 "--train_obj", "mean", "--precommit_b", f"{round(bf * rb['VaR95'], 2)}"] + extra
        print(name + " " + " ".join(args))
PY
echo "final run finished $(date)"
