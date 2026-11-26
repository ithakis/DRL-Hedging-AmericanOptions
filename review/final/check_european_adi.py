"""Validate the European variant of the ADI table against QuantLib's analytic Heston price (maturity tau/252)."""
import argparse
import numpy as np
import QuantLib as ql
from environment.heston_table import HestonAmericanPutTable, HestonTableConfig
from review.retrain.settings import train_args
import train_d4pg as T

args = T.get_args(train_args("B", 0.02, 1, out="/tmp/_conv") + ["--pricing_backend", "adi_trading"])
u = T.make_utils(argparse.Namespace(**{**vars(args), "spread": 0.0}), args.market_seed, 8)
cfg = HestonTableConfig.from_utils(u)
eu = HestonAmericanPutTable.load_or_build_adi(cfg, table_dir="../DRL-Hedging-retrain/pricing_tables", quantlib_maturity=False, european=True)
am = HestonAmericanPutTable.load_or_build_adi(cfg, table_dir="../DRL-Hedging-retrain/pricing_tables", quantlib_maturity=False)


def analytic(m, tau, vol):
    """Analytic Heston (QuantLib) at maturity tau/252 exactly: time change to a 365-day clock so that tau trading
    days are tau calendar days (rates, kappa, variances and xi scaled by c = 365/252, since tau/252 years = (tau/365) / c)."""
    today = ql.Date(1, 1, 2025); ql.Settings.instance().evaluationDate = today
    dc, c = ql.Actual365Fixed(), 365 / 252
    rts = ql.YieldTermStructureHandle(ql.FlatForward(today, cfg.r * c, dc))
    qts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    proc = ql.HestonProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(m)), vol ** 2 * c, cfg.kappa * c, cfg.theta * c,
                            cfg.volvol * c, cfg.rho)
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, 1.0), ql.EuropeanExercise(today + int(tau)))
    opt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc)))
    return opt.NPV()


rng = np.random.default_rng(3)
err, prem = [], []
for _ in range(200):
    m, tau, vol = rng.uniform(0.75, 1.3), int(rng.integers(1, 61)), rng.uniform(0.12, 0.6)
    pe = eu.evaluate(np.array([m]), np.array([1.0]), np.array([tau]), np.array([vol]))[0][0]
    pa = am.evaluate(np.array([m]), np.array([1.0]), np.array([tau]), np.array([vol]))[0][0]
    err.append(pe - analytic(m, tau, vol)); prem.append(pa - pe)
err, prem = np.abs(err), np.array(prem)
print(f"European ADI vs analytic Heston, 200 points: max |err| {err.max():.2e}, mean {err.mean():.2e} (per unit strike)")
print(f"American - European (ADI): min {prem.min():.2e}, max {prem.max():.2e}, mean {prem.mean():.2e}")
for m, tau, vol in [(1.0, 30, 0.3), (1.0, 5, 0.3), (0.9, 30, 0.3), (1.1, 60, 0.2)]:
    f = lambda t: t.evaluate(np.array([m]), np.array([1.0]), np.array([tau]), np.array([vol]))[0][0]
    print(m, tau, vol, "eu ADI", round(f(eu), 5), "am ADI", round(f(am), 5), "analytic", round(analytic(m, tau, vol), 5))
