#!/usr/bin/env python3
"""
Combinatorially Symmetric Cross-Validation (CSCV) and the Probability of Backtest
Overfitting (PBO).

Source: Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of Backtest Overfitting",
Journal of Computational Finance 20(4), 2017.  Open copy:
https://escholarship.org/uc/item/4w1110bb  and  https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf

WHY THIS SITS NEXT TO mcpt.py -- they answer two different questions and we need both:

  MCPT asks:  "could NOISE have produced a backtest this good?"
              null = the market has no exploitable structure.
  PBO asks:   "given that I tried N configurations and kept the best, how likely is it that
               the winner is below-median out of sample?"
              null = my SELECTION procedure is the thing generating the result.

A strategy can pass MCPT (the family has real edge) and still fail PBO (I cannot pick which
member of the family will work next year).  For our MNQ work, where the exit grid is 45 cells
and the walk-forward optimiser visibly flipped from stop2/tp6 to stop6/tp1 partway through the
sample, PBO is the more dangerous of the two tests and the one to run first.

ALGORITHM (as stated in the paper)
  1. Build M, a T x N matrix: one column per trial (configuration), one row per time
     observation, entries = that trial's per-period P&L or return.  All columns must be
     observed over the SAME index -- that is what makes the ranks comparable.
  2. Split the T rows into S disjoint submatrices of equal length T/S, S even, preserving
     time order within each block.
  3. For each of the C(S, S/2) combinations c: the chosen blocks form the in-sample set J_c,
     the complement forms the out-of-sample set J_c-bar.
  4. n*(c) = argmax over columns of the performance metric on J_c.
  5. w_c = relative rank of column n*(c) among the N out-of-sample metrics, in (0, 1).
  6. logit  lambda_c = log( w_c / (1 - w_c) ).  lambda_c <= 0 means the in-sample winner
     landed at or below the out-of-sample median.
  7. PBO = P(lambda_c <= 0) = (# of c with lambda_c <= 0) / C(S, S/2).

  Also reported, per the paper:
    * performance degradation: OLS of OOS metric on IS metric across combinations.  A
      negative slope is the signature of overfitting -- looking better in sample predicts
      doing WORSE out of sample.
    * probability of loss: fraction of combinations where the in-sample winner's OOS metric
      is below zero.

THE THRESHOLD IS THE AUTHORS', NOT OURS
  Section 3.1, verbatim: "a customary approach would be to reject models for which PBO is
  estimated to be greater than 0.05."  That is strict.  Use it; do not invent a looser one
  after seeing a result.

FIVE LIMITATIONS THE PAPER STATES, ALL OF WHICH BITE US
  1. FILE DRAWER.  "Hiding trials will lead to an underestimation of the overfit."  Every
     configuration we actually tried belongs in M -- not just the 45 exit cells, but the
     timeframes, the regime thresholds, the overnight variants, the negative-RR profiles.
     Feed it 45 columns when we really tried 300 and the PBO that comes back is too low.
     Their rule of thumb: "backtest as many theoretically reasonable strategy configurations
     as possible."  Padding with configurations designed to fail is the opposite cheat and
     biases it the other way.
  2. GUIDED SEARCH.  If an optimiser used earlier iterations to choose later ones, M's columns
     must be the CONVERGED outcome of each search, not the intermediate steps.
  3. IT DOES NOT CHECK CORRECTNESS.  "this procedure does nothing to evaluate the correctness
     of a backtest.  If the backtest is flawed due to bad assumptions, such as incorrect
     transaction costs or using data not available at the moment of making a decision, our
     approach will be making an assessment based on flawed information."  PBO does not replace
     the cost and look-ahead discipline in CLAUDE.md -- it assumes it.
  4. STRUCTURAL BREAKS OUTSIDE T ARE INVISIBLE.  Our MNQ file is 2021-2026.  A regime that
     ended before it starts cannot be detected.
  5. A PLATEAU INFLATES PBO.  "it is entirely possible that all the N strategies have high but
     similar Sharpe ratios.  Since none of the strategies is clearly better than the rest, PBO
     will be high."  This is a real tension with [[Parameter Plateaus]]: the flatness
     neurotrader wants us to look for is the same flatness that drives PBO up.  A high PBO on a
     flat grid means "I cannot tell these apart", not necessarily "none of them works".  Read
     PBO together with the MCPT p-value, never alone.

AND ONE PROHIBITION
  "we must warn the reader against applying CSCV to guide the search for an optimal strategy.
  That would constitute a gross misuse of our method ... PBO should not be the objective
  function on which such selection relies."  Same Strathern quote neurotrader uses: when a
  measure becomes a target it ceases to be a good measure.  Run PBO to judge a finished search.
  Never optimise against it.
"""

import itertools

import numpy as np


# ------------------------------------------------------------------ metrics
def sharpe(x, axis=0):
    """Per-period Sharpe, no risk-free rate and no annualisation -- the paper's default and
    the only form that is comparable across blocks of equal length."""
    sd = x.std(axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, x.mean(axis=axis) / sd, np.nan)


def profit_factor(x, axis=0):
    """Matches the objective mcpt.py optimises, so PBO and MCPT rank the same thing."""
    w = np.where(x > 0, x, 0.0).sum(axis=axis)
    l = -np.where(x < 0, x, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(l > 0, w / l, np.nan)


# ------------------------------------------------------------------ CSCV
def cscv(M, S=16, metric=sharpe):
    """M: (T, N) per-period P&L, one column per configuration. S: number of blocks (even).

    Returns a dict with pbo, the lambda distribution, the IS/OOS metric pairs, the
    degradation regression, and how often the in-sample winner lost money out of sample."""
    M = np.asarray(M, dtype=float)
    T, N = M.shape
    if N < 2:
        raise ValueError(
            "PBO needs at least 2 configurations -- it measures selection, "
            "and with one candidate there is nothing to select"
        )
    if S % 2:
        raise ValueError("S must be even")
    if T < S * 2:
        raise ValueError(f"T={T} too short for S={S} blocks")

    # Algorithm 2.3, step two: "partition M across rows, into an even number S of disjoint
    # submatrices of EQUAL dimensions ... of order (T/S x N)".  Equal, not near-equal, so the
    # tail rows that do not divide evenly are dropped rather than distributed.
    blk = T // S
    M = M[: blk * S]
    cut = [np.arange(s * blk, (s + 1) * blk) for s in range(S)]
    combos = list(itertools.combinations(range(S), S // 2))

    lam, is_m, oos_m, winners, oos_all = [], [], [], [], []
    for c in combos:
        ins = np.concatenate([cut[i] for i in c])  # J,  training, in order
        oos = np.concatenate([cut[i] for i in range(S) if i not in c])  # J-bar, testing, in order
        r_is = metric(M[ins])
        r_oos = metric(M[oos])
        if not np.isfinite(r_is).any():
            continue
        n_star = int(np.nanargmax(r_is))  # step (e): best performing strategy IS
        finite = np.isfinite(r_oos)
        if finite.sum() < 2:
            continue
        # step (f): w_bar_c = rank of the IS winner among the N OOS metrics, / (N + 1)
        order = np.argsort(np.argsort(np.where(finite, r_oos, -np.inf)))
        w = (order[n_star] + 1) / (finite.sum() + 1)
        w = min(max(w, 1e-9), 1 - 1e-9)
        lam.append(np.log(w / (1 - w)))  # step (g): logit
        is_m.append(r_is[n_star])
        oos_m.append(r_oos[n_star])
        oos_all.append(np.nanmean(r_oos))  # for stochastic dominance (sec 3.3)
        winners.append(n_star)

    lam, is_m, oos_m = np.array(lam), np.array(is_m), np.array(oos_m)
    oos_all = np.array(oos_all)
    ok = np.isfinite(is_m) & np.isfinite(oos_m)
    slope, intercept = np.polyfit(is_m[ok], oos_m[ok], 1) if ok.sum() > 2 else (np.nan, np.nan)
    uniq, cnt = np.unique(winners, return_counts=True)

    # Section 3.3: does selecting the IS-best beat picking one of the N at random?
    # First-order dominance of R_n* over Mean(R) requires P[R_n* >= x] >= P[mean >= x] for all x.
    grid = np.unique(np.concatenate([oos_m[np.isfinite(oos_m)], oos_all[np.isfinite(oos_all)]]))
    if len(grid):
        f_sel = np.array([(oos_m >= x).mean() for x in grid])
        f_rnd = np.array([(oos_all >= x).mean() for x in grid])
        fosd = bool(np.all(f_sel >= f_rnd - 1e-12) and np.any(f_sel > f_rnd + 1e-12))
        sd2 = float(
            np.trapezoid(
                (oos_all[:, None] <= grid).mean(0) - (oos_m[:, None] <= grid).mean(0), grid
            )
        )
    else:
        fosd, sd2 = False, np.nan

    return dict(
        pbo=float((lam <= 0).mean()),
        reject_at_005=bool((lam <= 0).mean() > 0.05),  # the paper's own customary threshold
        n_combinations=len(lam),
        S=S,
        N_trials=N,
        T=T,
        lambda_median=float(np.median(lam)),
        degradation_slope=float(slope),
        degradation_intercept=float(intercept),
        prob_oos_loss=float((oos_m < 0).mean()),
        oos_metric_median=float(np.median(oos_m)),
        is_metric_median=float(np.median(is_m)),
        first_order_dominance=fosd,
        sd2_integral=sd2,
        winner_counts={
            int(u): int(c) for u, c in sorted(zip(uniq, cnt, strict=True), key=lambda t: -t[1])
        },
        lambdas=lam.tolist(),
        is_metric=is_m.tolist(),
        oos_metric=oos_m.tolist(),
    )


def report(res, names=None):
    lines = [
        f"CSCV / PBO   {res['N_trials']} configurations, {res['T']} periods, "
        f"S={res['S']} blocks, {res['n_combinations']} combinations",
        f"  PBO                        {res['pbo']:.3f}   "
        f"(fraction of splits where the in-sample winner was at or below the OOS median)",
        f"  median logit lambda        {res['lambda_median']:+.3f}",
        f"  performance degradation    OOS = {res['degradation_slope']:+.3f} * IS "
        f"{res['degradation_intercept']:+.3f}   (negative slope = overfit signature)",
        f"  P(OOS metric < 0)          {res['prob_oos_loss']:.3f}",
        f"  median IS / OOS metric     {res['is_metric_median']:+.4f} / "
        f"{res['oos_metric_median']:+.4f}",
        "  most-selected configurations:",
    ]
    for k, v in list(res["winner_counts"].items())[:6]:
        nm = names[k] if names else f"#{k}"
        lines.append(f"    {nm:40s} chosen in {v} of {res['n_combinations']} splits")
    return "\n".join(lines)


if __name__ == "__main__":
    # self-check: with pure noise, selection carries no information, so PBO should sit near 0.5
    rng = np.random.default_rng(0)
    print(report(cscv(rng.normal(size=(2000, 40)), S=12)))
    print()
    # and with one genuinely superior column, PBO should collapse toward 0
    M = rng.normal(size=(2000, 40))
    M[:, 7] += 0.12
    print(report(cscv(M, S=12)))
