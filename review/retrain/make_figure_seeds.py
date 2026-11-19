"""Validation VaR95 during training for the 10 Heston agents at 2% cost (figure fig:cvar2 of the paper).

    PYTHONPATH=. python review/retrain/make_figure_seeds.py   # -> review/results/paper/figures/heston_seeds_validation.pdf

Reads the stored validation curves (review/results/retrain/runs/*/validation.csv), the selected agent of
analysis.py and the test VaR95 of Delta-Gamma and rule (b) from baselines_B.json. No simulation.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from review.retrain.analysis import OUT, pick, runs_of  # noqa: E402

SPREAD = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "review" / "results" / "paper" / "figures" / "heston_seeds_validation.pdf"))
    a = ap.parse_args()
    runs = runs_of("B", SPREAD)
    sel, _ = pick(runs)
    base = json.load(open(OUT / "baselines_B.json"))[f"{SPREAD}_0"]

    plt.style.use("default")
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    first = True
    for r in sorted(runs, key=lambda r: r["seed"]):
        if r is sel:
            continue
        v = pd.read_csv(r["dir"] / "validation.csv")
        ax.plot(v.episode / 1000, v.val_VaR95, c="gray", lw=0.8, alpha=0.8, label="other seeds" if first else None)
        first = False
    v = pd.read_csv(sel["dir"] / "validation.csv")
    ax.plot(v.episode / 1000, v.val_VaR95, c="tab:red", lw=1.6, label=f"selected agent (seed {sel['seed']})")
    best = v.loc[v.episode == sel["selected_episode"]].iloc[0]
    ax.plot(best.episode / 1000, best.val_VaR95, "o", c="tab:red", ms=4)
    ax.axhline(base["DG"]["VaR95"], c="k", ls="--", lw=0.8, label="Delta-Gamma (test)")
    ax.axhline(base["rule_b_VaR95"]["VaR95"], c="g", ls=":", lw=1.0, label="rule (b) (test)")
    ax.set_xlabel("training episodes (thousands)")
    ax.set_ylabel("VaR$_{95\\%}$ of the loss")
    ax.set_xlim(0, v.episode.max() / 1000 + 0.5)
    ax.set_ylim(30, 60)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print("selected run:", sel["policy"], "saved", a.out)


if __name__ == "__main__":
    main()
