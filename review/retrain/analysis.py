"""Analysis for the revision retrain: baselines, main table, delta costs, robustness, early exercise.

All numbers come from the new pipeline (cached markets of train_d4pg.py, trained policy_best.npz files).
Paths: validation seed 2025 (tuning / selection), test seed 654 (reporting), 8,192 paths each.

    PYTHONPATH=. python review/retrain/analysis.py rules --block A        # tune rules, test baselines
    PYTHONPATH=. python review/retrain/analysis.py main --block A         # RL table (+ gate check for A)
    PYTHONPATH=. python review/retrain/analysis.py deltacost --block B    # 0/1/2/5 bp after the fact
    PYTHONPATH=. python review/retrain/analysis.py robustness             # OOD table, block-B 2% agent
    PYTHONPATH=. python review/retrain/analysis.py early                  # early-exercise diagnostic
    PYTHONPATH=. python review/retrain/analysis.py export                 # weights/ + mapping.csv

Definitions (same as experiments/heston_rule_baselines.py and torch_d4pg/metrics.py):
loss = -P&L per episode, VaR95 = 95% quantile of the loss, CVaR95 = mean loss at or beyond VaR95.
Standard errors: bootstrap over test paths (1,000 resamples); gains use paired resamples.
"""

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import evaluate_policy_numpy as npe  # noqa: E402
import train_d4pg as T  # noqa: E402
from review.retrain.settings import BLOCKS, BUMPS, COSTS, run_name, with_bump  # noqa: E402

LOGS = ROOT / "logs_torch" / "retrain"
OUT = ROOT / "review" / "results" / "retrain"
DELTA_BP = [0, 1, 2, 5]
A_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 2)
W_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
N_BOOT = 1000
SVEN_BS_RULE_VAR = {0.005: 5.16, 0.01: 8.55, 0.02: 13.57}  # Sven's tuned gamma-fraction rule, his test set


# ----------------------------------------------------------------------------- markets / rollout
def block_args(block, extra=()):
    return T.get_args(BLOCKS[block]["flags"] + ["--obj_func", "var", "--spread", "0.0"] + list(extra))


def markets(block, which=("val", "test"), extra=()):
    a = block_args(block, extra)
    seeds = dict(val=(a.val_seed, a.n_val), test=(a.test_seed, a.n_test))
    return {w: T.get_market(a, *seeds[w]) for w in which}


def rollout(m, decide, vega_obs):
    """Vectorised episode. decide(t, obs, n_g, n_v, lo, hi) -> contracts x of the new hedge option.

    Returns dict: pnl before option cost, option-cost base (sum |x price|), delta-trade base
    (sum |stock position change| S_t), gamma / vega hedge ratios (mean of x/n_G, x/n_V).
    The action range [lo, hi] is the RL agent's range (vega part only when vega_obs).
    """
    m = {k: np.asarray(v) for k, v in m.items()}
    S = m["S"]
    P, T1 = S.shape
    n = np.zeros((P, T1))
    pnl, ob, db, pos_prev = np.zeros(P), np.zeros(P), np.zeros(P), np.zeros(P)
    rg, rv = [], []
    for t in range(T1 - 1):
        act = ~m["h_inactive"][:, t, :]
        G = m["l_gamma"][:, t] + np.sum(n * m["h_gamma"][:, t, :] * act, axis=1)
        V = m["l_vega"][:, t] + np.sum(n * m["h_vega"][:, t, :] * act, axis=1)
        g_h, v_h = m["h_gamma"][:, t, t], m["h_vega"][:, t, t]
        with np.errstate(divide="ignore", invalid="ignore"):
            n_g = np.nan_to_num(-G / g_h, nan=0.0, posinf=np.inf, neginf=-np.inf)
            n_v = np.nan_to_num(-V / v_h, nan=0.0, posinf=np.inf, neginf=-np.inf)
        lo, hi = np.minimum(0.0, n_g), np.maximum(0.0, n_g)
        if vega_obs:
            lo, hi = np.minimum(lo, n_v), np.maximum(hi, n_v)
        obs = np.stack([S[:, t], G, g_h] + ([V, v_h] if vega_obs else []), axis=1)
        x = decide(t, obs, n_g, n_v, lo, hi)
        n[:, t] = x
        ok_g, ok_v = np.abs(n_g) > 1e-8, np.abs(n_v) > 1e-8
        rg.append(np.where(ok_g, x / np.where(ok_g, n_g, 1.0), np.nan))
        rv.append(np.where(ok_v, x / np.where(ok_v, n_v, 1.0), np.nan))
        ob += np.abs(x * m["h_price"][:, t, t])
        act_next = ~m["h_inactive"][:, t + 1, :]
        pos = -(m["l_delta"][:, t] + np.sum(n * m["h_delta"][:, t, :] * act, axis=1))
        db += np.abs(pos - pos_prev) * S[:, t]
        pos_prev = pos
        pnl += m["l_next"][:, t] - m["l_price"][:, t]
        pnl += np.sum(n * (m["h_price"][:, t + 1, :] - m["h_price"][:, t, :]) * act_next, axis=1)
        pnl += pos * (S[:, t + 1] - S[:, t])
    return dict(pnl=pnl, ob=ob, db=db, gamma_ratio=float(np.nanmean(rg)), vega_ratio=float(np.nanmean(rv)))


def net(res, spread, bp=0):
    return res["pnl"] - spread * res["ob"] - bp * 1e-4 * res["db"]


def rule_decider(kind, a=0.0, b=0.0, w=0.0):
    """Delta (no option trade), DG (n_G), DV (n_V), fraction rule clip(a n_G + b n_V) with band w."""
    def f(t, obs, n_g, n_v, lo, hi):
        if kind == "delta":
            return np.zeros_like(n_g)
        if kind == "DG":
            return n_g
        if kind == "DV":
            return n_v
        lo2 = np.minimum(0, np.minimum(n_g, n_v))
        hi2 = np.maximum(0, np.maximum(n_g, n_v))
        x = np.clip(a * n_g + b * n_v, lo2, hi2)
        return np.where(np.abs(x) < w, 0.0, x)
    return f


def policy_decider(npz):
    pol = npe.NumpyPolicy(npz)

    def f(t, obs, n_g, n_v, lo, hi):
        u = np.clip(pol(obs.astype(np.float32).astype(float)), 0.0, 1.0)
        return lo + u * (hi - lo)
    return f, pol.meta


# ----------------------------------------------------------------------------- metrics
def var_cvar(pnl):
    return npe.var_cvar(pnl)


def summary(res, spread, bp=0, n_boot=N_BOOT):
    p = net(res, spread, bp)
    s = npe.summarize(p, spread * res["ob"], n_boot=n_boot)
    s = {k: float(v) for k, v in s.items()}
    s.update(gamma_ratio=res["gamma_ratio"], vega_ratio=res["vega_ratio"])
    return s


def paired_gain(p_bench, p_x, metric="VaR95", n_boot=N_BOOT, seed=7):
    """bench - x (positive = x is better) and its paired bootstrap SE."""
    P = len(p_x)
    idx = np.random.default_rng(seed).integers(0, P, (n_boot, P))
    k = 0 if metric == "VaR95" else 1
    g = var_cvar(p_bench)[k] - var_cvar(p_x)[k]
    gb = var_cvar(p_bench[idx])[k] - var_cvar(p_x[idx])[k]
    return float(g), float(gb.std(ddof=1))


def fmt(v, se):
    return f"{v:.2f} ({se:.2f})"


# ----------------------------------------------------------------------------- rule tuning
def tune_rules(val_res_fn, spread, bp, gamma_only):
    """Grid search on validation. Returns {(kind, metric): (a, b, w)} for metric in VaR95/CVaR95.

    val_res_fn(a, b) -> rollout result without band; the band is applied by re-running only when w > 0,
    so we cache rollouts per (a, b, w).
    """
    best = {}
    b_grid = [0.0] if gamma_only else A_GRID
    for a, b, w in itertools.product(A_GRID, b_grid, W_GRID):
        p = net(val_res_fn(a, b, w), spread, bp)
        v, c = var_cvar(p)
        for kind in (["a"] if w == 0 else []) + ["b"]:
            for metric, val in (("VaR95", v), ("CVaR95", c)):
                key = (kind, metric)
                if key not in best or val < best[key][0] - 1e-12:
                    best[key] = (float(val), float(a), float(b), float(w))
    return {k: v[1:] for k, v in best.items()}


class RolloutCache:
    def __init__(self, m, vega_obs):
        self.m, self.vega_obs, self.c = m, vega_obs, {}

    def rule(self, kind, a=0.0, b=0.0, w=0.0):
        key = (kind, round(a, 4), round(b, 4), round(w, 4))
        if key not in self.c:
            self.c[key] = rollout(self.m, rule_decider(kind, a, b, w), self.vega_obs)
        return self.c[key]

    def frac(self, a, b, w):
        return self.rule("rule", a, b, w)


# ----------------------------------------------------------------------------- commands
def cmd_rules(args):
    """Tune rules on validation (per cost, per bp), evaluate all baselines on test. Saves baselines.json."""
    blk = args.block
    gamma_only = blk == "A"  # Black-Scholes: no vega risk, one-parameter gamma-fraction rule (+ band)
    vega_obs = blk == "B"
    mk = markets(blk)
    val, test = RolloutCache(mk["val"], vega_obs), RolloutCache(mk["test"], vega_obs)
    out = {}
    for spread in COSTS:
        for bp in (DELTA_BP if blk == "B" else [0]):
            params = tune_rules(val.frac, spread, bp, gamma_only)
            row = {}
            for name in ["delta", "DG"] + ([] if gamma_only else ["DV"]):
                row[name] = summary(test.rule(name), spread, bp)
            for (kind, metric), (a, b, w) in params.items():
                row[f"rule_{kind}_{metric}"] = dict(a=a, b=b, w=w, **summary(test.frac(a, b, w), spread, bp))
            out[f"{spread}_{bp}"] = row
            print(spread, bp, {k: round(v["VaR95"], 2) for k, v in row.items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"baselines_{blk}.json", "w"), indent=2)


def runs_of(block, spread):
    rows = []
    for seed in BLOCKS[block]["seeds"]:
        d = LOGS / run_name(block, spread, seed)
        if (d / "results.json").exists():
            r = json.load(open(d / "results.json"))
            rows.append(dict(seed=seed, dir=d, val_VaR95=r["val_score"], selected_episode=r["selected_episode"],
                             test_VaR95=r["test"]["VaR95"], sec_per_1000=r["seconds_per_1000_episodes"]))
    rows.sort(key=lambda r: r["val_VaR95"])
    return rows


def pick(rows):
    sel = rows[0]
    med = rows[(len(rows) - 1) // 2]  # lower median of the validation scores
    return sel, med


def cmd_main(args):
    blk = args.block
    vega_obs = blk == "B"
    base = json.load(open(OUT / f"baselines_{blk}.json"))
    test_m = markets(blk, ("test",))["test"]
    test = RolloutCache(test_m, vega_obs)
    table, seeds_rows = [], []
    for spread in COSTS:
        rows = runs_of(blk, spread)
        if not rows:
            continue
        for r in rows:
            seeds_rows.append(dict(block=blk, spread=spread, **{k: v for k, v in r.items() if k != "dir"}))
        sel, med = pick(rows)
        b0 = base[f"{spread}_0"]
        rule_b = b0["rule_b_VaR95"]
        p_dg = net(test.rule("DG"), spread)
        p_rule = net(test.frac(rule_b["a"], rule_b["b"], rule_b["w"]), spread)
        for label, r in (("selected", sel), ("median", med)):
            f, _ = policy_decider(r["dir"] / "policy_best.npz")
            res = rollout(test_m, f, vega_obs)
            np.savez_compressed(OUT / f"rl_test_{blk}_{spread}_{label}.npz", **{k: v for k, v in res.items()
                                                                              if isinstance(v, np.ndarray)})
            s = summary(res, spread)
            p = net(res, spread)
            g_dg = paired_gain(p_dg, p)
            g_rule = paired_gain(p_rule, p)
            table.append(dict(block=blk, model=BLOCKS[blk]["tag"], spread=spread, agent=label, seed=r["seed"],
                              n_seeds=len(rows), val_VaR95=r["val_VaR95"], **s,
                              gain_vs_DG=g_dg[0], gain_vs_DG_se=g_dg[1],
                              gain_vs_rule_b=g_rule[0], gain_vs_rule_b_se=g_rule[1],
                              DG_VaR95=b0["DG"]["VaR95"], DG_VaR95_se=b0["DG"]["VaR95_se"],
                              delta_VaR95=b0["delta"]["VaR95"], delta_VaR95_se=b0["delta"]["VaR95_se"],
                              DV_VaR95=b0.get("DV", {}).get("VaR95", np.nan),
                              DV_VaR95_se=b0.get("DV", {}).get("VaR95_se", np.nan),
                              rule_a_VaR95=b0["rule_a_VaR95"]["VaR95"], rule_a_VaR95_se=b0["rule_a_VaR95"]["VaR95_se"],
                              rule_b_VaR95=rule_b["VaR95"], rule_b_VaR95_se=rule_b["VaR95_se"],
                              rule_b_params=f"a={rule_b['a']}, b={rule_b['b']}, w={rule_b['w']}"))
    df = pd.DataFrame(table)
    df.to_csv(OUT / f"main_{blk}.csv", index=False)
    pd.DataFrame(seeds_rows).to_csv(OUT / f"seeds_{blk}.csv", index=False)
    md = [f"# Main table, block {blk} ({BLOCKS[blk]['tag']}, VaR95 agents), test paths seed 654, 8,192 paths",
          "", "VaR95 of the episode loss, standard errors in brackets. Gain = baseline - RL (positive = RL better), "
          "paired bootstrap SE.", "",
          "| Cost | Agent | Seed | RL VaR95 | CVaR95 | Mean P&L | E[cost] | Gamma ratio | Vega ratio | Delta | DG | DV | "
          "Rule (a) | Rule (b) | Gain vs DG | Gain vs rule (b) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r.spread:.1%} | {r.agent} | {r.seed} | {fmt(r.VaR95, r.VaR95_se)} | {fmt(r.CVaR95, r.CVaR95_se)} | "
                  f"{fmt(r.mean_pnl, r.mean_pnl_se)} | {r.mean_cost:.2f} | {r.gamma_ratio:.2f} | {r.vega_ratio:.2f} | "
                  f"{fmt(r.delta_VaR95, r.delta_VaR95_se)} | {fmt(r.DG_VaR95, r.DG_VaR95_se)} | "
                  f"{'-' if np.isnan(r.DV_VaR95) else fmt(r.DV_VaR95, r.DV_VaR95_se)} | "
                  f"{fmt(r.rule_a_VaR95, r.rule_a_VaR95_se)} | {fmt(r.rule_b_VaR95, r.rule_b_VaR95_se)} | "
                  f"{fmt(r.gain_vs_DG, r.gain_vs_DG_se)} | {fmt(r.gain_vs_rule_b, r.gain_vs_rule_b_se)} |")
    if blk == "A":
        md += ["", "## Gate: selected BS agent vs tuned gamma-fraction rule (a) on the same test paths", "",
               "| Cost | RL VaR95 | Rule (a) VaR95 (ours) | RL - rule (paired SE) | Pass | Sven's rule VaR (his paths) |",
               "|---|---|---|---|---|---|"]
        for _, r in df[df.agent == "selected"].iterrows():
            rule_a = base[f"{r.spread}_0"]["rule_a_VaR95"]
            p_ra = net(test.frac(rule_a["a"], 0.0, 0.0), r.spread)
            p_rl = np.load(OUT / f"rl_test_A_{r.spread}_selected.npz")
            p_rl = p_rl["pnl"] - r.spread * p_rl["ob"]
            g, se = paired_gain(p_ra, p_rl)  # rule - RL
            ok = -g <= 2 * se
            md.append(f"| {r.spread:.1%} | {r.VaR95:.2f} | {rule_a['VaR95']:.2f} (a={rule_a['a']}) | "
                      f"{-g:+.2f} ({se:.2f}) | {'yes' if ok else 'NO'} | {SVEN_BS_RULE_VAR[r.spread]:.2f} |")
        md += ["", "Pass = RL VaR95 is not worse than the rule by more than 2 paired standard errors."]
    (OUT / f"main_{blk}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def cmd_deltacost(args):
    blk = args.block
    vega_obs = blk == "B"
    base = json.load(open(OUT / f"baselines_{blk}.json"))
    test_m = markets(blk, ("test",))["test"]
    test = RolloutCache(test_m, vega_obs)
    rows = []
    for spread in COSTS:
        z = np.load(OUT / f"rl_test_{blk}_{spread}_selected.npz")
        res = {k: z[k] for k in z.files}
        res.update(gamma_ratio=np.nan, vega_ratio=np.nan)
        for bp in DELTA_BP:
            b0 = base[f"{spread}_{bp}"]
            p_rl = net(res, spread, bp)
            s = summary(res, spread, bp)
            row = dict(spread=spread, delta_bp=bp, RL_VaR95=s["VaR95"], RL_VaR95_se=s["VaR95_se"],
                       RL_CVaR95=s["CVaR95"], RL_CVaR95_se=s["CVaR95_se"])
            for name, rb in (("DG", None), ("DV", None), ("rule_b", b0["rule_b_VaR95"])):
                p = net(test.rule(name) if rb is None else test.frac(rb["a"], rb["b"], rb["w"]), spread, bp)
                v, c = var_cvar(p)
                g, se = paired_gain(p, p_rl)
                row.update({f"{name}_VaR95": float(v), f"{name}_CVaR95": float(c),
                            f"gain_vs_{name}": g, f"gain_vs_{name}_se": se})
            row["rule_b_params"] = f"a={b0['rule_b_VaR95']['a']}, b={b0['rule_b_VaR95']['b']}, w={b0['rule_b_VaR95']['w']}"
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"deltacost_{blk}.csv", index=False)
    md = ["# Delta-hedging cost charged after the fact (no retraining), selected VaR95 agents, test paths", "",
          "Delta cost = bp x |change in stock position| x S_t. Rule (b) re-tuned on validation for each cell. "
          "Gain = baseline VaR95 - RL VaR95 (paired SE).", "",
          "| Cost | Delta bp | RL VaR95 | DG | DV | Rule (b) | Gain vs DG | Gain vs DV | Gain vs rule (b) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r.spread:.1%} | {r.delta_bp} | {fmt(r.RL_VaR95, r.RL_VaR95_se)} | {r.DG_VaR95:.2f} | "
                  f"{r.DV_VaR95:.2f} | {r.rule_b_VaR95:.2f} | {fmt(r.gain_vs_DG, r.gain_vs_DG_se)} | "
                  f"{fmt(r.gain_vs_DV, r.gain_vs_DV_se)} | {fmt(r.gain_vs_rule_b, r.gain_vs_rule_b_se)} |")
    (OUT / f"deltacost_{blk}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def cmd_robustness(args):
    spread = 0.02
    sel, _ = pick(runs_of("B", spread))
    f, _ = policy_decider(sel["dir"] / "policy_best.npz")
    base = json.load(open(OUT / "baselines_B.json"))[f"{spread}_0"]
    ra, rb = base["rule_a_VaR95"], base["rule_b_VaR95"]
    rows = []
    for flag, v in [(None, None)] + BUMPS:
        a = T.get_args((BLOCKS["B"]["flags"] if flag is None else with_bump(BLOCKS["B"]["flags"], flag, v))
                       + ["--obj_func", "var", "--spread", "0.0"])
        m = T.get_market(a, a.test_seed, a.n_test)
        c = RolloutCache(m, True)
        rl = rollout(m, f, True)
        row = dict(parameter="base" if flag is None else flag.lstrip("-"), value=v, n_paths=a.n_test,
                   test_seed=a.test_seed)
        for name, res in (("RL", rl), ("delta", c.rule("delta")), ("DG", c.rule("DG")), ("DV", c.rule("DV")),
                          ("rule_a", c.frac(ra["a"], ra["b"], 0.0)), ("rule_b", c.frac(rb["a"], rb["b"], rb["w"]))):
            s = summary(res, spread)
            for k in ["VaR95", "VaR95_se", "CVaR95", "CVaR95_se", "mean_pnl", "mean_pnl_se", "mean_cost"]:
                row[f"{name}_{k}"] = s[k]
        p_rl = net(rl, spread)
        for name, res in (("DG", c.rule("DG")), ("rule_b", c.frac(rb["a"], rb["b"], rb["w"]))):
            g, se = paired_gain(net(res, spread), p_rl)
            row[f"gain_vs_{name}"], row[f"gain_vs_{name}_se"] = g, se
        rows.append(row)
        print(row["parameter"], v, round(row["RL_VaR95"], 2), round(row["DG_VaR95"], 2), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "robustness.csv", index=False)
    md = ["# Out-of-distribution generalisation (Heston, r = 2%, 2% option cost)", "",
          f"RL = selected VaR95 agent (seed {sel['seed']}) trained at the base parameters, not retrained. "
          "Every row: new test paths (seed 654, 8,192 paths) and all prices/Greeks at the bumped parameters. "
          f"Rules keep their base tuning (rule (a): a={ra['a']}, b={ra['b']}; rule (b): a={rb['a']}, b={rb['b']}, "
          f"w={rb['w']}). VaR95 of the episode loss (SE).", "",
          "| Parameter | Value | RL | Delta | DG | DV | Rule (a) | Rule (b) | RL CVaR95 | RL E[cost] | "
          "Gain vs DG | Gain vs rule (b) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r.parameter} | {'' if r.value is None or pd.isna(r.value) else r.value} | "
                  + " | ".join(fmt(r[f'{n}_VaR95'], r[f'{n}_VaR95_se']) for n in
                               ["RL", "delta", "DG", "DV", "rule_a", "rule_b"])
                  + f" | {fmt(r.RL_CVaR95, r.RL_CVaR95_se)} | {r.RL_mean_cost:.2f} | "
                  f"{fmt(r.gain_vs_DG, r.gain_vs_DG_se)} | {fmt(r.gain_vs_rule_b, r.gain_vs_rule_b_se)} |")
    (OUT / "robustness.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def cmd_early(args):
    """How often the held hedge puts are in the early-exercise region, by days to expiry; premium Am - Eu."""
    import QuantLib as ql  # noqa: F401
    rows = []
    for label, extra in (("r=2% (main)", []), ("r=0 (contrast)", ["--r", "0", "--mu", "0"])):
        a = block_args("B", extra)
        m = {k: np.asarray(v) for k, v in T.get_market(a, a.test_seed, a.n_test).items()}
        S, cs = m["S"], a.contract_size
        P, T1 = S.shape
        for i in range(T1 - 1):  # hedge option issued on day i, strike = S_i (ATM)
            K = S[:, i]
            for t in range(i, T1):
                active = ~m["h_inactive"][:, t, i]
                if not active.any():
                    continue
                ttm = a.hed_ttm - (t - i)
                if ttm <= 0:
                    continue
                price = m["h_price"][:, t, i] / cs
                intr = np.maximum(K - S[:, t], 0.0)
                inreg = active & (intr > 0) & (price - intr < 5e-4)
                rows.append(dict(setting=label, issue_day=i, day=t, ttm=ttm, n_active=int(active.sum()),
                                 n_exercise_region=int(inreg.sum()),
                                 mean_moneyness=float(np.mean(S[active, t] / K[active]))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "early_exercise_by_day.csv", index=False)
    g = df.groupby(["setting", "ttm"])[["n_active", "n_exercise_region"]].sum().reset_index()
    g["share"] = g.n_exercise_region / g.n_active
    tot = df.groupby("setting")[["n_active", "n_exercise_region"]].sum()
    tot["share"] = tot.n_exercise_region / tot.n_active
    prem = early_premium()
    prem.to_csv(OUT / "early_exercise_premium.csv", index=False)
    md = ["# Early-exercise diagnostic (Heston test paths, seed 654, 8,192 paths)", "",
          "A held hedge put (strike = spot at issue) is in the exercise region when it is in the money and "
          "its model price is within 5e-4 of intrinsic value. The environment never exercises; it marks the "
          "put to model value, which equals intrinsic value in the region.", "",
          "| Setting | Option-days | In exercise region | Share |", "|---|---|---|---|"]
    for s, r in tot.iterrows():
        md.append(f"| {s} | {int(r.n_active)} | {int(r.n_exercise_region)} | {r.share:.3%} |")
    md += ["", "By days to expiry (only rows with any option in the region):", "",
           "| Setting | Days to expiry | Option-days | In region | Share |", "|---|---|---|---|---|"]
    for _, r in g[g.n_exercise_region > 0].iterrows():
        md.append(f"| {r.setting} | {r.ttm} | {int(r.n_active)} | {int(r.n_exercise_region)} | {r.share:.3%} |")
    md += ["", "Early-exercise premium (American - European, per share, K = 10, v = 0.09, QuantLib FD vs analytic):",
           "", "| r | Days to expiry | S/K | American | European | Premium | Premium / American |",
           "|---|---|---|---|---|---|---|"]
    for _, r in prem.iterrows():
        md.append(f"| {r.r:.0%} | {r.ttm} | {r.moneyness:.2f} | {r.american:.4f} | {r.european:.4f} | "
                  f"{r.premium:.4f} | {r.rel:.2%} |")
    (OUT / "early_exercise.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def early_premium():
    import QuantLib as ql
    from environment.utils import Utils
    rows = []
    for r in (0.0, 0.02):
        for ttm in (5, 15, 30, 60):
            for mny in (0.8, 0.9, 1.0, 1.1):
                S0, K = 10.0 * mny, 10.0
                T_ = ttm / 252
                am = Utils._American_Option_Heston(S0, K, T_, r, 0.0, 0.09, 1.0, 0.09, 0.3, -0.7,
                                                   N_gridpoints=200, Type="put", exercise="American")[0]
                eu = _heston_european(S0, K, T_, r, 0.09, 1.0, 0.09, 0.3, -0.7)
                rows.append(dict(r=r, ttm=ttm, moneyness=mny, american=am, european=eu, premium=am - eu,
                                 rel=(am - eu) / am if am > 0 else 0.0))
    return pd.DataFrame(rows)


def _heston_european(S0, K, T_, r, v0, kappa, theta, sigma, rho):
    import QuantLib as ql
    today = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    days = max(1, round(T_ * 365))
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, K), ql.EuropeanExercise(today + days))
    rts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    qts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    proc = ql.HestonProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(S0)), v0, kappa, theta, sigma, rho)
    opt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc)))
    return opt.NPV()


def cmd_export(args):
    """Copy every trained policy to weights/ and write mapping.csv (table entry -> selected / median seed)."""
    W = ROOT / "review" / "results" / "retrain" / "weights"
    W.mkdir(parents=True, exist_ok=True)
    rows = []
    for blk in BLOCKS:
        for spread in COSTS:
            runs = runs_of(blk, spread)
            if not runs:
                continue
            sel, med = pick(runs)
            for r in runs:
                name = run_name(blk, spread, r["seed"]) + ".npz"
                shutil.copy(r["dir"] / "policy_best.npz", W / name)
                rows.append(dict(model=BLOCKS[blk]["tag"], objective="VaR95",
                                 option_cost=spread, seed=r["seed"], file=f"weights/{name}",
                                 selected=r is sel, median=r is med, val_VaR95=r["val_VaR95"],
                                 test_VaR95=r["test_VaR95"], selected_episode=r["selected_episode"],
                                 robustness_agent=(blk == "B" and spread == 0.02 and r is sel)))
    pd.DataFrame(rows).to_csv(W.parent / "mapping.csv", index=False)
    print(f"{len(rows)} policies -> {W}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rules", "main", "deltacost", "robustness", "early", "export"])
    ap.add_argument("--block", choices=["A", "B"], default="B")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{a.cmd}"](a)
