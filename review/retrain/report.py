"""Write review/retrain_report.md: new results (this pipeline) next to the OLD submitted-paper numbers.

Old numbers are copied from the submitted manuscript (Tables gbmtable and hestontableWdrift). They are
shown only here, in columns labelled OLD, and never mixed into the result tables of the revision.

    PYTHONPATH=. python review/retrain/report.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "review" / "results" / "retrain"

# OLD paper numbers (submitted version). VaR95 rows; agents selected on the evaluation paths (in-sample).
OLD = {
    "BS": dict(delta=24.71, DG={0.005: 5.51, 0.01: 9.92, 0.02: 18.93}, RL={0.005: 5.38, 0.01: 8.60, 0.02: 14.08},
               RL_cost={0.005: 2.88, 0.01: 4.42, 0.02: 6.09}),
    "HestonP": dict(delta=54.54, DG={0.005: 31.68, 0.01: 35.03, 0.02: 41.96},
                    DV={0.005: 38.93, 0.01: 44.79, 0.02: 57.43}, RL={0.005: 25.12, 0.01: 28.55, 0.02: 34.78},
                    RL_cost={0.005: 3.44, 0.01: 5.61, 0.02: 9.59},
                    gamma_ratio={0.005: 0.40, 0.01: 0.26, 0.02: 0.24}, vega_ratio={0.005: 0.30, 0.01: 0.24, 0.02: 0.18}),
}


def section(model, blk, title):
    f = RES / f"main_{blk}.csv"
    if not f.exists():
        return [f"## {title}", "", f"(main_{blk}.csv not found yet)", ""]
    df = pd.read_csv(f)
    o = OLD[model]
    out = [f"## {title}", "",
           "VaR95 of the episode loss. NEW = this pipeline, test paths (seed 654, 8,192), agent selected on "
           "validation paths (seed 2025), standard errors in brackets. OLD = submitted paper (agent picked on "
           "the same paths it was reported on, no standard errors).", "",
           "| Cost | NEW RL selected | NEW RL median | OLD RL | NEW DG | OLD DG | NEW Delta | OLD Delta | "
           "NEW rule (b) | NEW RL E[cost] | OLD RL E[cost] |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(df.spread.unique()):
        s = df[(df.spread == c) & (df.agent == "selected")].iloc[0]
        m = df[(df.spread == c) & (df.agent == "median")].iloc[0]
        out.append(f"| {c:.1%} | {s.VaR95:.2f} ({s.VaR95_se:.2f}) | {m.VaR95:.2f} ({m.VaR95_se:.2f}) | {o['RL'][c]:.2f} | "
                   f"{s.DG_VaR95:.2f} ({s.DG_VaR95_se:.2f}) | {o['DG'][c]:.2f} | {s.delta_VaR95:.2f} | {o['delta']:.2f} | "
                   f"{s.rule_b_VaR95:.2f} ({s.rule_b_VaR95_se:.2f}) | {s.mean_cost:.2f} | {o['RL_cost'][c]:.2f} |")
    out.append("")
    out.append("| Cost | NEW gain of RL vs DG | OLD gain vs DG | NEW gain of RL vs rule (b) |")
    out.append("|---|---|---|---|")
    for c in sorted(df.spread.unique()):
        s = df[(df.spread == c) & (df.agent == "selected")].iloc[0]
        out.append(f"| {c:.1%} | {s.gain_vs_DG:.2f} ({s.gain_vs_DG_se:.2f}) = {s.gain_vs_DG / s.DG_VaR95:.1%} | "
                   f"{o['DG'][c] - o['RL'][c]:.2f} = {(o['DG'][c] - o['RL'][c]) / o['DG'][c]:.1%} | "
                   f"{s.gain_vs_rule_b:.2f} ({s.gain_vs_rule_b_se:.2f}) |")
    out.append("")
    return out


def main():
    md = ["# Retrained agents vs the submitted paper (OLD numbers clearly labelled)", "",
          "All NEW numbers come from the PyTorch D4PG-QR pipeline (branch v2-torch-d4pg) and the driver in "
          "`review/retrain/`. OLD numbers are copied from the submitted manuscript and are not reproducible "
          "(weights lost). Differences in protocol: NEW selects on separate validation paths and reports on test "
          "paths with bootstrap standard errors; NEW Heston prices come from the precomputed table (central "
          "0.5 vol-point vega bump, fixes the vega spikes of the original forward bump); NEW Heston scenario is "
          "r = 2%, mu = 3%, lambda_v = 0 (the OLD runs also had lambda_v = 0 in the code); the actor input is "
          "scaled (median/IQR), which the old acme agent did through layer norm only.", ""]
    md += section("BS", "A", "Black-Scholes (sigma 0.3, r = 0), VaR95 agents, 5 seeds")
    md += section("HestonP", "B", "Heston, r = 2%, mu = 3%, lambda_v = 0, VaR95 agents, 10 seeds")
    for name in ["main_A.md", "deltacost_B.md", "robustness.md", "early_exercise.md"]:
        f = RES / name
        if f.exists():
            md += [f"## From `{name}`", "", f.read_text().split("\n", 1)[1].strip(), ""]
    (ROOT / "review" / "retrain_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
