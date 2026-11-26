"""Sven item 2: size of the day-count bug on the baselines. Same test paths (seed 4242: identical S and V), option
prices and Greeks from (a) the QuantLib-clock table (maturity round(tau*365/252)/365, as in the original submission's
pricer) and (b) the adi_trading table (tau/252). VaR95 / CVaR95 of DG, Delta and rule (b)."""
import json
import numpy as np
import pandas as pd
from review.retrain import analysis as A
from review.retrain.settings import COSTS

rows = []
mk = {}
for be in ["table", "adi_trading"]:
    a = A.block_args("B", ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", be,
                           "--table_dir", "../DRL-Hedging-retrain/pricing_tables"])
    mk[be] = A.T.get_market(a, 4242, 8192)
print("identical S paths:", np.array_equal(np.asarray(mk["table"]["S"]), np.asarray(mk["adi_trading"]["S"])),
      "identical V paths:", np.array_equal(np.asarray(mk["table"]["V"]), np.asarray(mk["adi_trading"]["V"])))
OLD = {0.005: dict(a=0.3, b=0.3, w=0.3), 0.01: dict(a=0.3, b=0.3, w=0.5), 0.02: dict(a=0.2, b=0.3, w=1.0)}  # tuned on the old-clock market
for c in COSTS:
    new = json.load(open(f"review/results/warmstart/richer_rule_params_adi_trading_{c}.json"))[str(c)]["rule_b"]
    ob = OLD[c]
    for be, m in mk.items():
        for name, dec in [("Delta", A.rule_decider("delta")), ("DG", A.rule_decider("DG")),
                          (f"rule (b) {new['a']},{new['b']},{new['w']}", A.rule_decider("rule", new["a"], new["b"], new["w"])),
                          (f"rule (b) {ob['a']},{ob['b']},{ob['w']}", A.rule_decider("rule", ob["a"], ob["b"], ob["w"]))]:
            v, cv = A.var_cvar(A.net(A.rollout(m, dec, True), c))
            rows.append(dict(cost=c, policy=name, backend=be, VaR95=float(v), CVaR95=float(cv)))
df = pd.DataFrame(rows)
t = df.pivot_table(index=["cost", "policy"], columns="backend", values="VaR95").round(2)
t["diff"] = (t["adi_trading"] - t["table"]).round(2)
print(t.to_string())
df.to_csv("review/results/final/daycount_effect.csv", index=False)
