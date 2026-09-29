# Quantitative_Trading_Projects
Summary and repo of all collective work and research done for futures trading and trading based research. 

Intraday Futures Research Framework (MES / MNQ)

A quantitative research framework for testing intraday trading ideas on Micro E-mini S&P 500 (MES) and Micro E-mini Nasdaq-100 (MNQ) futures.

The framework is built around one rule: an idea is not an edge until it survives a statistical test designed to kill it. Most trading ideas look profitable in a backtest because of luck, overfitting, or unrealistic costs. This project exists to catch those problems before any money is at risk.

Key findings
Finding	Evidence
Most popular intraday signals do not hold up.	18 pre-registered hypotheses tested, 15 rejected: moving-average families, Ichimoku, momentum filters, opening-range breakouts, level-break acceptance, and multi-timeframe VWAP confluence
MES and MNQ mean-revert intraday.	98% of 64 variance-ratio cells below 1.0 (p = 0.0033). This one result explained why ten earlier trend-following strategies failed
A working acceptance/rejection filter exists, but the textbook version does not.	The conventional moving-average version was a coin flip (49.9%). A reworked discriminator held in 78% of 188 cells (p = 0.0025)
Anchored-VWAP σ bands mostly measure time of day.	75% of band touches happen before 09:50 CT, so the signal largely encodes how much of the session has elapsed
Published results can flip once real costs are added.	Reproduced the headline result of a published SSRN paper, then showed it reverses after adding bid-ask spread and measured slippage

All results are from historical backtests and simulations. None of them are live trading returns.

How it works
Hypothesis  ──►  Pre-registration  ──►  Backtest with real costs  ──►  Statistical tests  ──►  Verdict
(written down     (hashed and saved       (level-2 slippage,            (permutation, overfit,     (only "EDGE"
 before testing)   before the run)         commissions)                  variance ratio)            is tradeable)
1. Pre-registration

Before any test runs, the hypothesis, the rule that would disprove it, and the metric used to judge it are written down, hashed, and saved. This stops the results from quietly changing the question after the fact.

2. Realistic cost modeling
Round-trip costs come from measured level-2 order-book slippage (average fill price minus the best quote, by order size), not an assumed constant.
Commissions are traced back to each broker's published fee schedule.
3. Statistical validation
Monte Carlo permutation testing. Shuffles bars in groups so the normal intraday volatility pattern is preserved, then checks whether the real result beats the shuffled ones. A vectorized rewrite made this about 100× faster than the original loop.
Search-breadth correction. When a family of 100–1,300 parameter settings is tested, the test accounts for how many settings were tried, so the best one doesn't win just by luck.
Overfitting detection. Combinatorially symmetric cross-validation (CSCV) estimates the probability of backtest overfitting (PBO).
Lo–MacKinlay variance ratio test. Heteroskedasticity-robust test of whether prices trend or mean-revert at a given horizon.
4. Verdict system

Every idea ends in one of four states. The code will not label any configuration as tradeable unless the evidence reaches EDGE.

5. Data discipline

Data is split into train, validate, and holdout periods, and a log records which periods each experiment has seen. The holdout data has not been used yet.

Evaluation account simulator

A separate module models futures evaluation-account rules (trailing vs. static drawdown, real-time vs. end-of-day breach checks, consistency rules, payout paths, and fees) and uses Monte Carlo simulation to estimate:

Probability of passing
Time to pass
Chance of reaching a payout

…as functions of win rate, reward-to-risk, position size, and withdrawal policy.

The simulator was checked against the closed-form result P(pass) = L / (T + L), where T is the profit target and L is the loss limit. The match held across an 8× range of volatility, which confirms the math is volatility-independent as theory predicts.

Market profile engine

Builds Time Price Opportunity (TPO) profiles for the 08:30–15:00 CT session:

Value area high / low (70%) and point of control
Composite value areas and untouched prior levels
Initial balance
Tech stack
Python: backtesting, statistics, simulation
Parquet: cached tick and bar data
Java (MotiveWave SDK): custom studies that stream bid/ask, delta, and footprint data into Python
Data: MES/MNQ tick history (2020–2026) and level-2 order book data
Repository layout

```
src/quant/                    core library
  permutation.py              grouped bar permutation, vectorised (~100x the reference loop)
  mcpt.py                     Monte Carlo permutation test driver
  pbo.py                      CSCV / probability of backtest overfitting
  variance_ratio.py           Lo-MacKinlay variance ratios, computed per session segment
  information.py              forward-outcome evaluation and family p-values
  hypothesis.py               pre-registration records, hashed before the run
  verdict.py                  four-state verdict; refuses a config below EDGE
  protocol.py                 research protocol enforcement
  costs.py                    commissions, fees, measured level-2 book-walk slippage
  prop.py                     evaluation and funded-account simulators
  data_splits.py              declared train / validate / holdout boundaries
  splits.py                   walk-forward planning
  bar_integrity.py            bar and session sanity checks
  diagnostics.py              run diagnostics
  register.py                 run registration
  strategy.py                 shared indicator and data-loading helpers
  viz.py, viz_strategy.py     plotting

src/quant/strategies/         strategy and indicator families
  indicator_menu.py           indicator array x six signal definitions
  vwap_bands.py               anchored VWAP with volume-weighted sigma bands
  opening_range_reclaim.py    opening-range break, retest and reclaim
  levels.py                   structural levels, with a random-level control
  ma_pairs.py                 moving-average pair family
  ichimoku.py                 Ichimoku line family
  kama_ema.py                 adaptive-average crosses
  cross_filter.py             cross conditioners
  cross_frequency.py          cross frequency analysis
  path_quality.py             path-quality conditioners
  ema_cross.py                baseline EMA cross
  ema_cross_sep.py            separated EMA cross variant

scripts/                      40 runnable experiments and reports, including:
  direction_screen.py         is direction predictable at 5-20 bars?
  direction_ceiling.py        the bound on every linear filter (every moving average) at once
  indicator_array.py          indicator families x signal definitions, with negative controls
  variance_ratio_survey.py    trend vs mean reversion by horizon
  vwap_band_race.py           VWAP band reversion vs extension
  vwap_band_by_session.py     the same, bucketed by time of day
  orb_reclaim_test.py         opening-range break, retest and reclaim
  prop_validate.py            simulator checked against P(pass) = L/(T+L)
  prop_spec_feasibility.py    pass rate, time to pass and payout attainment surfaces
  prop_two_day_pass.py        how fast an evaluation can be passed, and at what size
  replicate_youngblood.py     reproduces a published SSRN result, then adds real costs
  build_volume_profiles.py    TPO / volume profile engine (POC, VAH, VAL) from raw ticks
  build_ofi_bars.py           order-flow imbalance bars from level-2 quotes

tests/                        15 modules, 169 tests
pine/                         TradingView studies (measurement only, no entry signals)
research/hypotheses.jsonl     the pre-registration ledger
```

Market data is not included: it is large, vendor licensed and not redistributable.
Set `QUANT_DATA_ROOT` to point at your own copy.

```bash
pip install -e .
export QUANT_DATA_ROOT=/path/to/data
pytest -q
```

Tests that require market data skip cleanly when it is absent.

What I learned
A strategy that looks great in a backtest is usually luck. The only fair test is one that tries hard to prove the idea wrong.
Trading costs decide whether small edges survive. Assumed costs are often too optimistic, so measure them from real order-book data.
Understanding why markets behave a certain way (for example, intraday mean reversion) is more useful than tuning parameters until a backtest looks good.
Contact

Nikita Pavlovich · Finance, Capital Markets & Investments, University of Illinois Chicago (May 2027) linkedin.com/in/nikita-pavlovich · npavlovich88@gmail.com
