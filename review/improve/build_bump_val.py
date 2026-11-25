"""Validation markets (seed 2025) for the robustness screen bumps (not the test-grid values)."""
from review.retrain import analysis as A
from review.retrain.settings import BLOCKS, with_bump
extra = ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", "adi_trading",
         "--table_dir", "../DRL-Hedging-retrain/pricing_tables", "--obj_func", "var", "--spread", "0.0"]
for f, v in [("--init_vol", 0.18), ("--init_vol", 0.42), ("--volvol", 0.38)]:
    m = A.T.get_market(A.T.get_args(with_bump(BLOCKS["B"]["flags"], f, v) + extra), 2025, 8192)
    print(f, v, m["S"].shape, flush=True)
