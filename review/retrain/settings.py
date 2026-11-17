"""Run settings for the revision retrain (thread 3, Sven's scope). One place for all flags.

Block A (gate): Black-Scholes / GBM, VaR95, 3 option costs, 5 agent seeds.
Block B (main): Heston under P with r = 2%, mu = 3%, lambda_v = 0 (P and Q variance dynamics equal),
                VaR95, 3 option costs, 10 agent seeds.
Robustness: block-B agent at 2% cost on 5 parameters x 2 bumps (outer values of "Robustness Testing.sh").
"""
COSTS = [0.005, 0.01, 0.02]
EPISODES = 40_000

COMMON = ["--obj_func", "var", "--train_episodes", str(EPISODES),
          "--market_seed", "1234", "--val_seed", "2025", "--test_seed", "654",
          "--n_val", "8192", "--n_test", "8192", "--torch_threads", "1"]

BLOCKS = {
    "A": dict(flags=["--process", "GBM", "--init_vol", "0.3", "--r", "0", "--mu", "0"], seeds=range(1, 6),
              tag="BS"),
    "B": dict(flags=["--process", "Heston", "--r", "0.02", "--mu", "0.03", "--lambda_v", "0",
                     "--init_vol", "0.3", "--kappa", "1.0", "--theta", "0.09", "--volvol", "0.3",
                     "--rho", "-0.7"], seeds=range(1, 11), tag="HestonP"),
}

# robustness bumps (parameter flag, value); init_vol is the initial vol (v0 = init_vol^2)
BUMPS = [("--init_vol", 0.15), ("--init_vol", 0.45), ("--volvol", 0.15), ("--volvol", 0.45),
         ("--rho", -0.85), ("--rho", -0.55), ("--theta", 0.045), ("--theta", 0.135),
         ("--kappa", 0.65), ("--kappa", 1.35)]


def run_name(block, spread, seed):
    return f"{BLOCKS[block]['tag']}_var_spread{spread}_seed{seed}"


def train_args(block, spread, seed, out="logs_torch/retrain"):
    b = BLOCKS[block]
    return b["flags"] + COMMON + ["--spread", str(spread), "--seed", str(seed), "--out", out,
                                  "--run_name", run_name(block, spread, seed)]


def with_bump(flags, flag, value):
    f = list(flags)
    f[f.index(flag) + 1] = str(value)
    return f
