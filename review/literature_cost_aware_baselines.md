# Literature notes: cost-aware hedging baselines (for the referee reply)

Date: 2026-11-17. Goal: understand current best practice for cost-aware baselines in (deep / RL) hedging, so the
reply to the referee and the new Heston experiments are aligned with the literature.
Search: web search and the arXiv pages of the papers below. The paper-search (OpenAlex) plugin was not enabled in this
session and the shared OpenAlex quota was used up. Classic references (marked *) come from my own knowledge and were
not re-opened here, so please check the details before citing them.

## 1. Closest work: Cao, Chen, Farghadani, Hull, Poulos, Wang, Yuan (2023)

"Gamma and vega hedging using deep distributional reinforcement learning", Frontiers in AI 6:1129370
([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9992725/), [arXiv 2205.05614](https://arxiv.org/pdf/2205.05614v1)).
Our environment follows their setup closely:
- Poisson(1/day) client options, 60-day, random long/short, S0 = 10, ATM 30-day hedge option, daily delta hedge
  with free stock trades, costs 0.5/1/2% of option value. SV case = SABR (vol 30%, vol-of-vol 0.3, rho -0.7).
- Baselines: delta, **full** delta-gamma, **full** delta-vega. No tuned or cost-aware rule.
- Objectives: mean + 1.645 std, VaR95, CVaR95 (same as ours).
- Important for us: the RL agent hedged **about 57% of gamma** (1% cost), and in the SV case a **vega hedge ratio of
  0.32-0.37**. Our tuned rule under Heston uses a = 0.2-0.4 of gamma and b = 0.2-0.3 of vega. So a large part of
  what the RL agent learns there looks like a partial-hedge rule. That is exactly the referee's point, and a
  tuned fraction rule is the natural "strong baseline" for this setup.

## 2. Theory: under proportional costs the optimal policy is a no-trade band

- Leland (1985)*, Hodges & Neuberger (1989)*, Whalley & Wilmott (1997)* "An asymptotic analysis of an optimal
  hedging model for option pricing with transaction costs": with proportional costs the utility-optimal hedge
  does not trade inside a band around the frictionless hedge. Whalley-Wilmott give the small-cost band width,
  h ~ (3 lambda S Gamma^2 / (2 gamma))^(1/3). Zakamouline (2006)* compares these rules numerically
  ([EFMA 2005 paper](https://www.efmaefm.org/0efmameetings/EFMA%20ANNUAL%20MEETINGS/2005-Milan/papers/284-zakamouline_paper.pdf)).
- Imaki, Imajo, Ito, Minami, Nakagawa, "No-Transaction Band Network" ([arXiv 2103.01775](https://arxiv.org/pdf/2103.01775)):
  they prove the band structure is optimal for a wide class of utilities and payoffs. A network that only outputs the
  band converges much faster than plain deep hedging (Buehler et al. 2019*) and is close to Whalley-Wilmott for
  small costs.
- Arzel & Lehdili (2026), "Bridging Stochastic Control and Deep Hedging: Structural Priors for No-Transaction Band
  Networks" ([arXiv 2603.29994](https://arxiv.org/html/2603.29994)): they put the Whalley-Wilmott band in as a prior,
  benchmark against the stochastic-control (HJB) solution, and test several cost levels.

Takeaway: "fraction + no-trade band" is the right family of cost-aware benchmarks. The Whalley-Wilmott formula
itself is for costs on the **underlying** in Black-Scholes. In our setup the costs are on the **options** and
delta trades are free, so the formula does not apply directly. A band tuned on validation paths, as we did, is
the empirical version of the same idea.

## 3. Recent empirical work: tuned classical rules are hard to beat

- Kumar (2026), deep hedging on Bitcoin options ([arXiv 2608.29025](https://arxiv.org/html/2608.29025v1)): a
  Whalley-Wilmott band beats all deep hedging networks out of sample (P&L gap about 17-21 per episode,
  p = 0.004). It also uses block-bootstrap CIs on the test set.
- Zernikov (2026), "What does deep hedging actually learn?" ([arXiv 2605.21696](https://arxiv.org/html/2605.21696v1)):
  the RL agent mainly learns a systematic delta haircut (it hedges less than BS delta). A small symbolic formula
  distilled from the policy keeps most of the gain. Benchmarks include a scalar haircut (= fraction) rule.
  Walk-forward train / select / test and a two-stage bootstrap.
- Armstrong & Tatlow (2024), "Deep Gamma Hedging" ([arXiv 2409.13567](https://arxiv.org/html/2409.13567v1)):
  two-instrument deep hedging beats delta and gamma hedging when there are costs. Plain gamma hedging is best
  without costs, and its main value is robustness to model uncertainty.
- Pickard & Lawryshyn (2023), review of DRL for option hedging, Mathematics 11(24):4943
  ([DOI 10.3390/math11244943](https://www.citedrive.com/en/discovery/deep-reinforcement-learning-for-dynamic-stock-option-hedging-a-review)):
  most DRL papers compare only with plain delta hedging, which is a known weakness of the field.
- Other context: Cao, Chen, Hull, Poulos (2021) deep hedging with RL ([arXiv 2103.16409](https://arxiv.org/pdf/2103.16409v1.pdf));
  Kolm & Ritter (2019) ([SSRN 3281235](https://papers.ssrn.com/abstract=3281235)); Mikkilä & Kanniainen (2023),
  "Empirical deep hedging", Quantitative Finance 23(1):111-122
  ([RePEc](https://econpapers.repec.org/RePEc:taf:quantf:v:23:y:2023:i:1:p:111-122)).

## 4. What this means for our revision

1. The referee's comment is in line with the literature. Full gamma / vega flattening is a weak benchmark under
   costs. A tuned partial hedge (fraction) and a no-trade band are now the expected baselines.
2. What we already do matches current practice: rules tuned on validation paths, reported on a fresh test set
   with bootstrap standard errors, several cost levels, VaR and CVaR.
3. Our Heston result (tuned rule about 20% better than full Delta-Gamma, even at 0.5% cost) fits Cao et al.:
   their RL agent also hedged only a fraction of gamma and vega. So the paper should compare the retrained RL
   agent with the tuned rule on the same test paths (seed 654). If RL is about equal to the rule, we should say so
   honestly. A possible framing: RL finds a good state-dependent partial hedge without a hand-tuned grid. This is
   the same message as Zernikov (2026) and the no-transaction band papers.
4. Possible small additions (ask Alexander first, this is new scope):
   - report the RL agent's effective hedge ratios (traded contracts / n_G and / n_V), to compare with a and b;
   - cite Whalley-Wilmott / Imaki et al. when we motivate the band rule.
