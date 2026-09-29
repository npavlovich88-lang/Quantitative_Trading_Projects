"""Does an event carry directional information? The shared core of the information tests.

WHY THIS EXISTS AS A MODULE

The Ichimoku study (2026-09-24) asked whether a price/line cross predicts anything, and answered
it without simulating a single trade. That turned out to be the right shape for the question:
an event that carries no directional information cannot be rescued by an exit rule, so the cheap
test comes first and the expensive one only runs on survivors. The machinery is not specific to
Ichimoku, so it lives here rather than being copied per study.

THE STATISTIC

For an event type E and a horizon h:

    t = ( mean(forward return | E fired long) - mean(forward return | E fired short) ) / SE

Forward return runs close[i] -> close[i+h] and is divided by ATR(14) shifted one bar, so that a
2020 move and a 2026 move are comparable. Positive t means the event predicts continuation,
negative means reversion.

WHY A DIFFERENCE OF TWO CONDITIONAL MEANS

Unconditional drift is present in both groups and cancels in the difference. Comparing one side
against zero would mostly measure the fact that equity indices went up.

WHY THE MAXIMUM, AND WHY A PERMUTATION

Studies of this shape evaluate hundreds or thousands of cells at once. Reporting the best cell's
own p-value is the most common way a backtest lies. So `family_pvalues` computes two things:

  max-statistic     max |t| over the WHOLE family, against the max |t| over the same family
                    recomputed on each permuted market. Powerful against one strong cell.
  count-statistic   how many cells beat their OWN null 95th percentile, against how many the
                    permuted markets manage. Powerful against many weak cells.

They answer different questions and can disagree; that disagreement is informative, not a bug.

A NOTE ON CALIBRATION, LEARNED THE HARD WAY

Do not compare t against 1.96. When the long and short events interleave in time -- which they
do for any crossover -- the two groups' forward windows overlap each other, the two means are
positively correlated, and the Welch standard error OVERSTATES the variance of their difference.
Measured on the Ichimoku family, the per-cell null 95th percentile of |t| had a median of 1.377,
falling with horizon. A t-table is wrong per cell and far too lenient across a family at the
same time. Only the permutation distribution is calibrated.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .strategy import true_range

ATR_N = 14
DEFAULT_HORIZONS = (1, 3, 6, 12, 26, 78)


def atr_shifted(df: pd.DataFrame, n: int = ATR_N) -> np.ndarray:
    """ATR(n) shifted one bar. The R-unit must not know the current bar's own range."""
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    return pd.Series(true_range(h, lo, c)).ewm(alpha=1 / n, adjust=False).mean().shift(1).to_numpy()


@dataclasses.dataclass(frozen=True)
class Cells:
    """Per-cell results, shape (n_events, n_horizons)."""

    t: np.ndarray  # the statistic
    d: np.ndarray  # the effect, in ATR units
    n_long: np.ndarray
    n_short: np.ndarray


def evaluate(
    close: np.ndarray,
    atr: np.ndarray,
    masks: list[tuple[np.ndarray, np.ndarray]],
    base: np.ndarray,
    horizons=DEFAULT_HORIZONS,
    min_events: int = 100,
    direction: np.ndarray | None = None,
) -> Cells:
    """t-statistics for every (event type, horizon) cell on one series.

    `masks` is one (long_event, short_event) boolean pair per event type. `base` is the
    eligibility mask (session window, warm-up, finite ATR) and is ANDed into every event.

    The group sums go through one float32 matmul per horizon rather than 2N boolean-index
    reductions. That is what makes a 3,000-cell family x 1,000 permutations affordable; callers
    should check it against an exact float64 recomputation once per series (`verify`).
    """
    n = len(close)
    n_ev = len(masks)
    # `direction` flips the forward return per bar, so a test run inside a SHORT state reports a
    # positive t when the conditioner helps. Without it, every bearish cell would read backwards
    # and the family maximum would be taken over a mix of two sign conventions.
    sgn = np.ones(n) if direction is None else np.asarray(direction, dtype=float)
    M = np.zeros((2 * n_ev, n), dtype=np.float32)
    for j, (lg, sh) in enumerate(masks):
        M[2 * j] = lg & base
        M[2 * j + 1] = sh & base

    shape = (n_ev, len(horizons))
    t_out = np.full(shape, np.nan)
    d_out = np.full(shape, np.nan)
    nl = np.zeros(shape, dtype=np.int64)
    ns = np.zeros(shape, dtype=np.int64)

    for hi, hor in enumerate(horizons):
        fr = np.full(n, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            fr[: n - hor] = (close[hor:] - close[: n - hor]) / atr[: n - hor] * sgn[: n - hor]
        ok = np.isfinite(fr)
        f0 = np.where(ok, fr, 0.0).astype(np.float32)
        X = np.empty((n, 3), dtype=np.float32)
        X[:, 0] = f0
        X[:, 1] = f0 * f0
        X[:, 2] = ok
        R = (M @ X).astype(np.float64)  # (2*n_ev, 3): sum, sum of squares, count

        cnt = R[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = R[:, 0] / cnt
            var = np.maximum(R[:, 1] / cnt - mean**2, 0.0)
        mu_l, mu_s = mean[0::2], mean[1::2]
        v_l, v_s = var[0::2], var[1::2]
        cl, cs = cnt[0::2], cnt[1::2]
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(v_l / np.maximum(cl - 1, 1) + v_s / np.maximum(cs - 1, 1))
            d = mu_l - mu_s
            t = np.where(se > 0, d / se, np.nan)
        enough = (cl >= min_events) & (cs >= min_events)
        t_out[:, hi] = np.where(enough, t, np.nan)
        d_out[:, hi] = np.where(enough, d, np.nan)
        nl[:, hi] = cl.astype(np.int64)
        ns[:, hi] = cs.astype(np.int64)

    return Cells(t=t_out, d=d_out, n_long=nl, n_short=ns)


def verify(
    close: np.ndarray,
    atr: np.ndarray,
    masks: list[tuple[np.ndarray, np.ndarray]],
    base: np.ndarray,
    cells: Cells,
    horizons=DEFAULT_HORIZONS,
    min_events: int = 100,
    n_check: int = 6,
    seed: int = 0,
) -> float:
    """Largest disagreement between the float32 matmul path and an exact float64 recomputation.

    A fast statistic that is subtly wrong is worse than a slow one, so this runs once per series
    and the caller aborts if it is large.
    """
    n = len(close)
    worst = 0.0
    rng = np.random.default_rng(seed)
    for j in rng.choice(len(masks), size=min(n_check, len(masks)), replace=False):
        lg, sh = masks[j]
        lg, sh = lg & base, sh & base
        for hi, hor in enumerate(horizons):
            fr = np.full(n, np.nan)
            fr[: n - hor] = (close[hor:] - close[: n - hor]) / atr[: n - hor]
            ok = np.isfinite(fr)
            a, b = fr[lg & ok], fr[sh & ok]
            if len(a) < min_events or len(b) < min_events:
                continue
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            ref = cells.t[j, hi]
            if np.isfinite(ref) and se > 0:
                worst = max(worst, abs((a.mean() - b.mean()) / se - ref))
    return worst


def family_pvalues(real: np.ndarray, null: np.ndarray) -> dict:
    """The two family-level tests, plus everything needed to report them honestly.

    `real` is the full array of t-statistics over the family; `null` is the same array recomputed
    for each permutation, with the permutation as axis 0. Shapes must agree after axis 0.
    """
    n_perm = null.shape[0]
    finite = np.isfinite(real)
    with np.errstate(invalid="ignore"):
        null_max = np.nanmax(np.abs(null.reshape(n_perm, -1)), axis=1)
        obs_max = float(np.nanmax(np.abs(real)))
        p_max = (1 + int((null_max >= obs_max).sum())) / (1 + n_perm)

        # Per-cell threshold from the cell's OWN null, never from a t-table.
        thr = np.nanpercentile(np.abs(null), 95, axis=0)
        over = np.where(finite, np.abs(real) > thr, False)
        cnt_real = int(over.sum())
        cnt_null = np.array(
            [int(np.where(finite, np.abs(null[p]) > thr, False).sum()) for p in range(n_perm)]
        )
        p_count = (1 + int((cnt_null >= cnt_real).sum())) / (1 + n_perm)

        p_cell = np.where(
            finite, (1 + (np.abs(null) >= np.abs(real)[None]).sum(axis=0)) / (1 + n_perm), np.nan
        )

    return dict(
        permutations=n_perm,
        live_cells=int(finite.sum()),
        observed_max_t=obs_max,
        null_max_median=float(np.median(null_max)),
        null_max_p95=float(np.percentile(null_max, 95)),
        null_max_max=float(null_max.max()),
        p_max=p_max,
        cells_over_own_null=cnt_real,
        null_count_median=float(np.median(cnt_null)),
        null_count_p95=float(np.percentile(cnt_null, 95)),
        p_count=p_count,
        per_cell_threshold_median=float(np.nanmedian(thr[finite])),
        null_max_draws=null_max,
        null_count_draws=cnt_null,
        p_cell=p_cell,
        cell_threshold=thr,
    )


def winner_support(real: np.ndarray, families: np.ndarray | None = None) -> dict:
    """Is the best cell a plateau or a lone spike, and does the other market agree?

    `real` is (symbol, timeframe, config, horizon). Returns the two checks that the 2026-09-24
    HMA/TEMA/VWAP run showed are worth more than a marginal family p-value:

      neighbour_ratio   the winner's |t| divided by the median |t| of the other configurations
                        of its own sub-family, same symbol, timeframe and horizon. A lone spike
                        has a large ratio over a flat field.
      cross_symbol_t    the SAME configuration, timeframe and horizon on the other symbol.

    Note what is deliberately NOT here: support along the horizon axis. Forward returns at h=3, 6
    and 12 are nested windows over the same bars, so neighbouring horizons are correlated by
    construction and a horizon "plateau" is an artifact of the measure, not evidence. Only the
    parameter axis and the second market carry information.
    """
    si, ti, ci, hi = np.unravel_index(np.nanargmax(np.abs(real)), real.shape)
    w = float(abs(real[si, ti, ci, hi]))

    sel = np.ones(real.shape[2], bool) if families is None else (families == families[ci])
    sel = sel.copy()
    sel[ci] = False
    peers = np.abs(real[si, ti, sel, hi])
    peer_med = float(np.nanmedian(peers)) if np.isfinite(peers).any() else float("nan")

    others = [s for s in range(real.shape[0]) if s != si]
    cross = [float(real[s, ti, ci, hi]) for s in others]
    cross_finite = [x for x in cross if np.isfinite(x)]

    return dict(
        index=(int(si), int(ti), int(ci), int(hi)),
        winner_t=float(real[si, ti, ci, hi]),
        peer_median_abs_t=peer_med,
        neighbour_ratio=(w / peer_med if peer_med and np.isfinite(peer_med) else float("inf")),
        cross_symbol_t=cross_finite,
        cross_symbol_agrees=bool(
            cross_finite
            and all(np.sign(x) == np.sign(real[si, ti, ci, hi]) for x in cross_finite)
            and max(abs(x) for x in cross_finite) >= 1.5
        ),
    )


OUTCOMES = ("ret", "mfe", "mae")


def forward_outcomes(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    direction: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Forward return, best excursion and worst excursion over the next `horizon` bars.

    WHY ALL THREE, AND WHY THIS SIGN CONVENTION

    Every test in this project up to 2026-09-24 compared MEAN forward returns. A trade with a
    stop and a target does not care about the mean; it cares about whether the favourable
    excursion arrives before the adverse one. Those come apart -- a filter can leave the mean
    untouched while turning ten small losses and one large win into eleven medium ones.

    All three are ATR-normalised, direction-adjusted, and oriented so that HIGHER IS BETTER for
    the trader, including `mae`, which is returned NEGATED. That matters: it means one family
    maximum can be taken across all three without mixing sign conventions.

    Excursions look at bars t+1..t+h. Bar t's own high and low are excluded because the entry
    is at that bar's close and its range has already happened.
    """
    d = np.ones(len(close)) if direction is None else np.asarray(direction, dtype=float)
    fmax = pd.Series(high).rolling(horizon).max().shift(-horizon).to_numpy()
    fmin = pd.Series(low).rolling(horizon).min().shift(-horizon).to_numpy()
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = close[horizon:] - close[: n - horizon]
    up, dn = fmax - close, close - fmin
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = fwd * d / atr
        mfe = np.where(d > 0, up, dn) / atr
        mae = -np.where(d > 0, dn, up) / atr  # negated: higher means a smaller drawdown
    return {"ret": ret, "mfe": mfe, "mae": mae}


def evaluate_outcomes(
    outcome: np.ndarray,
    masks: list[tuple[np.ndarray, np.ndarray]],
    base: np.ndarray,
    min_events: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Welch t of (mean outcome | first mask) - (mean outcome | second mask), per mask pair.

    Same float32 matmul as `evaluate`, but the outcome is supplied rather than derived from
    closes, so excursion labels go through the identical code path as returns.
    """
    n = len(outcome)
    M = np.zeros((2 * len(masks), n), dtype=np.float32)
    for j, (a, b) in enumerate(masks):
        M[2 * j] = a & base
        M[2 * j + 1] = b & base
    ok = np.isfinite(outcome)
    f0 = np.where(ok, outcome, 0.0).astype(np.float32)
    X = np.empty((n, 3), dtype=np.float32)
    X[:, 0] = f0
    X[:, 1] = f0 * f0
    X[:, 2] = ok
    R = (M @ X).astype(np.float64)
    cnt = R[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = R[:, 0] / cnt
        var = np.maximum(R[:, 1] / cnt - mean**2, 0.0)
    m1, m2 = mean[0::2], mean[1::2]
    v1, v2 = var[0::2], var[1::2]
    c1, c2 = cnt[0::2], cnt[1::2]
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(v1 / np.maximum(c1 - 1, 1) + v2 / np.maximum(c2 - 1, 1))
        diff = m1 - m2
        t = np.where(se > 0, diff / se, np.nan)
    enough = (c1 >= min_events) & (c2 >= min_events)
    return (
        np.where(enough, t, np.nan),
        np.where(enough, diff, np.nan),
        c1.astype(np.int64),
        c2.astype(np.int64),
    )
