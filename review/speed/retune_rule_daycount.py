"""Re-tune rule (b) (a, b, w) on the VALIDATION market built with --pricing_backend adi_trading (day-count check).

Same grid and selection as review/retrain/analysis.py (cmd_rules, block B, bp 0, VaR95). Validation only.
    PYTHONPATH=. python review/speed/retune_rule_daycount.py --markets ../DRL-Hedging-retrain/markets \
        --table_dir ../DRL-Hedging-retrain/pricing_tables
Writes review/results/speed/daycount_rule_b.json.
"""
import argparse
import json
from pathlib import Path

from review.retrain import analysis as A
from review.retrain.settings import COSTS

ap = argparse.ArgumentParser()
ap.add_argument("--markets", default="../DRL-Hedging-retrain/markets")
ap.add_argument("--table_dir", default="../DRL-Hedging-retrain/pricing_tables")
args = ap.parse_args()
a = A.block_args("B", ["--market_dir", args.markets, "--pricing_backend", "adi_trading", "--table_dir", args.table_dir])
val = A.RolloutCache(A.T.get_market(a, a.val_seed, a.n_val), True)
old = json.load(open("review/results/retrain/baselines_B.json"))
out = {}
for c in COSTS:
    best = A.tune_rules(val.frac, c, 0, False)[("b", "VaR95")]
    ob = old[f"{c}_0"]["rule_b_VaR95"]
    v_new = float(A.var_cvar(A.net(val.frac(*best), c, 0))[0])
    v_old = float(A.var_cvar(A.net(val.frac(ob["a"], ob["b"], ob["w"]), c, 0))[0])
    out[str(c)] = dict(a=best[0], b=best[1], w=best[2], val_VaR95=v_new,
                       old_params=dict(a=ob["a"], b=ob["b"], w=ob["w"]), val_VaR95_old_params=v_old)
    print(c, out[str(c)], flush=True)
Path("review/results/speed").mkdir(parents=True, exist_ok=True)
json.dump(out, open("review/results/speed/daycount_rule_b.json", "w"), indent=2)
