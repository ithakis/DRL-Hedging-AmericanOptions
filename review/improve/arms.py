"""Arms of the improvement screens: FINAL flags (review/warmstart/run_final.sh) + one change per arm.
Prints one line per run: '<run_name> <train_d4pg args>'. Factors: b = BF x rule val VaR95, band = BANDF x rule w."""
import json
import os
import sys
from pathlib import Path

from review.retrain.settings import run_name, train_args

ARMS = {
    "ctrl": [],
    "b45": ["BF", "0.45"],
    "b30": ["BF", "0.3"],
    "rin": ["--rule_obs", "--rule_frac_obs"],
    "band05": ["BANDF", "0.5"],
    "upd2": ["--samples_per_insert", "64"],
    # batch 2
    "mc": ["--critic_type", "mean"],
    "kap1": ["--precommit_kappa", "1.0"],
    "lam05": ["--precommit_lambda", "0.05"],
    "lam15": ["--precommit_lambda", "0.15"],
    "aoc": ["--actor_on_current"],
    "dqn": ["--dqda_norm", "--dqda_clip", "3"],
    "sq05": ["--squash", "0.5"],
    # batch 3
    "arr2": ["--kernel_arrivals", "2"],
}


def lines(arms, seeds, costs, out, episodes):
    for s in seeds:  # seed-major order: a stopped screen still has complete paired comparisons
        for arm in arms:
            extra = list(ARMS[arm])
            bf, bandf = 0.6, 1.0
            if "BF" in extra:
                i = extra.index("BF"); bf = float(extra[i + 1]); del extra[i:i + 2]
            if "BANDF" in extra:
                i = extra.index("BANDF"); bandf = float(extra[i + 1]); del extra[i:i + 2]
            for c in costs:
                rb = json.load(open(f"review/results/warmstart/richer_rule_params_adi_trading_{c}.json"))[str(c)]["rule_b"]
                name = run_name("B", c, s) + "_" + arm
                if Path(f"{out}/{name}/results.json").exists():
                    continue
                args = train_args("B", c, s, out=out)
                args[args.index("--run_name") + 1] = name
                args += ["--market_dir", "../DRL-Hedging-retrain/markets", "--n_train_paths", "40000",
                         "--pricing_backend", "adi_trading", "--table_dir", "../DRL-Hedging-retrain/pricing_tables",
                         "--extra_obs", "--kernel_mx", "3", "--kernel_atm_fix", "--n_envs", "16",
                         "--train_episodes", str(episodes), "--val_every", "500", "--skip_test",
                         "--warm_rule", f"{rb['a']},{rb['b']},{rb['w']}", "--residual_scale", "0.5",
                         "--actor_freeze_episodes", "1000", "--band", str(round(bandf * rb["w"], 4)),
                         "--actor_sizes", "64,64,64", "--critic_sizes", "64,64,64", "--activation", "swish3",
                         "--lr", "6e-4", "--actor_lr", "3e-4", "--ema_decay", "0.999", "--tau", "0.003",
                         "--noise_mode", "pre", "--sigma", "1.0", "--sigma_floor", "0.2",
                         "--train_obj", "mean", "--precommit_b", f"{round(bf * rb['VaR95'], 2)}"] + extra
                print(name + " " + " ".join(args))


if __name__ == "__main__":
    lines(sys.argv[1].split(), [int(s) for s in sys.argv[2].split()], [float(c) for c in sys.argv[3].split()],
          os.environ.get("OUT", "logs_torch/improve"), int(os.environ.get("EPISODES", "8000")))
