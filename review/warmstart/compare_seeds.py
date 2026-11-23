"""Seed-paired comparison of set-ups (validation and test VaR95, gain vs rule (b)), from pilot.csv."""
import sys
import pandas as pd
RULE_VAL = {0.005: 24.59, 0.01: 27.87, 0.02: 32.80}
vs = sys.argv[1].split(",")
anchor = sys.argv[2] if len(sys.argv) > 2 else "KBresband"
d = pd.read_csv("review/results/warmstart/pilot.csv")
d = d[d.variant.isin(vs + [anchor])]
rows = []
for (v, c), g in d.groupby(["variant", "spread"]):
    a = d[(d.variant == anchor) & (d.spread == c)].set_index("seed")
    g = g.set_index("seed")
    common = g.index.intersection(a.index)
    sel = g.loc[g.val_VaR95.idxmin()]
    rows.append(dict(variant=v, cost=c, seeds=len(g), mean_val=g.val_VaR95.mean(), rule_val=RULE_VAL[c],
                     val_vs_anchor=(g.val_VaR95[common] - a.val_VaR95[common]).mean(),
                     mean_test=g.test_VaR95.mean(), mean_gain=g.gain_vs_rule_b.mean(),
                     sel_seed=int(sel.name), sel_gain=sel.gain_vs_rule_b, sel_gain_se=sel.gain_vs_rule_b_se,
                     sec_per_1000=g.sec_per_1000.mean()))
print(pd.DataFrame(rows).round(2).to_string(index=False))
