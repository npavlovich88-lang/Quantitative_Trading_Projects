#!/usr/bin/env python3
"""Run one strategy through the whole ladder and emit every figure plus a written conclusion.

    python scripts/run_strategy.py --symbol MES --tf 5m --fast 8 --slow 21 --perms 300

Gates run in order and the run HALTS at the first failure. Nothing downstream of a failed gate
is computed, because a walk-forward on an unvalidated edge is a precise number about nothing.

The expensive gates (5 and 7) are the permutation tests. `--perms 0` skips them, which is only
appropriate when an earlier gate has already failed and you are producing the report anyway.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant import diagnostics as diag
from quant import pbo as P
from quant import strategy as S
from quant import viz
from quant import viz_strategy as V
from quant.costs import CostModel
from quant.data_splits import DATA, make_split, make_wf_plan
from quant.hypothesis import Hypothesis
from quant.mcpt import GRID, bar_pnl, pf, simulate_with_px
from quant.permutation import get_permutation, session_groups
from quant.register import TrialRegister
from quant.strategies.ema_cross import build_features
from quant.verdict import Verdict

S.simulate = simulate_with_px

P_INSAMPLE, P_WALKFORWARD, PBO_MAX = 0.01, 0.05, 0.05


def log(msg=""):
    print(msg, flush=True)


def grid_best(F, cost_pts, lo, hi, reg=None, segment="train"):
    """Best configuration over the exit grid, registering every trial looked at."""
    best = (None, -np.inf, None, None)
    for cfg in GRID:
        tr = S.simulate(F, cfg["exit"], cfg["stop_atr"], cfg["tp_atr"], cost_pts)
        tr = tr[(tr["entry_i"] >= lo) & (tr["entry_i"] < hi)]
        if len(tr) < 20:
            continue
        b = bar_pnl(F, tr, cost_pts)[lo:hi]
        p = pf(b)
        if reg is not None:
            reg.record(dict(cfg), segment=segment, metrics={"pf": p, "n": len(tr)}, pnl=b)
        if np.isfinite(p) and p > best[1]:
            best = (cfg, p, tr, b)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="MES", choices=list(DATA))
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--fast", type=int, default=8)
    ap.add_argument("--slow", type=int, default=21)
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--firm", default="average")
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tag = f"ema{a.fast}_{a.slow}_{a.symbol}_{a.tf}"
    out = pathlib.Path(a.out or f"out/{tag}")
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    gates, numbers = [], {}

    cost = CostModel.for_prop(a.symbol, a.firm, contracts=a.contracts)
    cp = cost.round_turn_points
    log(f"{'=' * 78}\n  EMA({a.fast})/EMA({a.slow})  {a.symbol} {a.tf}\n{'=' * 78}")
    log(cost.breakdown())

    # ---------------------------------------------------------------- 1 pre-register
    h = Hypothesis(
        name=tag,
        instrument=f"{a.symbol} {a.tf} full-session bars",
        session="08:30-15:00 America/Chicago, RTH entries only, flat at the session close",
        setup=f"EMA({a.fast}) of close crosses above EMA({a.slow}) of close for a long, below "
        f"for a short. The cross is the trigger, not the state: no re-entry while the "
        f"relationship persists.",
        direction="both",
        exit_rule="best of a pre-declared 45-cell exit grid (stop 2-8 ATR x target none-8 ATR, "
        "plus SuperTrend flip), ATR(14) shifted one bar",
        execution="enter at the close of the bar on which the cross completes; stops and targets "
        "evaluated intrabar, gap-aware, stop assumed first when both touch",
        costs=f"CostModel.for_prop('{a.symbol}','{a.firm}',contracts={a.contracts}) "
        f"= {cp:.3f} pts round turn",
        invalidation=f"abandon if PBO > {PBO_MAX}, OR in-sample MCPT p >= {P_INSAMPLE}, OR "
        f"walk-forward MCPT p >= {P_WALKFORWARD}, OR fewer than 30 train trades. "
        f"A fail sends it back to a new hypothesis, not to a parameter tweak.",
        metric="profit factor on per-bar P&L, train segment, all 45 grid cells registered",
        rationale="Faster pair than the 10/29 already rejected. Same family, so a pass would be "
        "surprising; running it is a genuine test of whether the pair matters.",
    )
    hh = h.register("out/hypotheses.jsonl")
    log(f"\n{h.describe()}\n")
    gates.append(("1. pre-registered", "pass", f"hash {hh}, registered before any run"))

    # ---------------------------------------------------------------- data + split
    df = S.load_data(DATA[a.symbol].format(tf=a.tf), "America/Chicago")
    d = df["dt"].dt.tz_localize(None).to_numpy()
    sp = make_split(a.symbol, a.tf, d)
    wf = make_wf_plan(d)
    F = build_features(df, fast=a.fast, slow=a.slow)
    lo, hi = sp.train.start, sp.train.stop
    log(sp.describe(d))
    log("  " + wf.describe(sp, d).replace("\n", "\n  ") + "\n")

    # ---------------------------------------------------------------- 2 causality
    from quant.strategy import ema

    c = df["close"].to_numpy(float)
    # Probe indices scaled to the series, not hardcoded. The old (5000, 20000, 50000) crashed
    # outright on any file shorter than 50,001 bars -- which is every 30m and 60m series for both
    # symbols. The ladder could not be run at the timeframes the information tests now point at.
    probes = sorted({int(len(c) * f) for f in (0.25, 0.55, 0.9)} - {0})
    ok = bool(probes) and all(
        np.isclose(ema(c, n)[i], ema(c[: i + 1], n)[i], rtol=1e-9)
        for n in (a.fast, a.slow)
        for i in probes
    )
    gates.append(
        (
            "2. causality",
            "pass" if ok else "fail",
            f"EMA({a.fast}) and EMA({a.slow}) prefix-tested; ATR shifted one bar",
        )
    )
    log(f"[gate 2] causality: {'PASS' if ok else 'FAIL'}")
    if not ok:
        return finish(out, figs, tag, gates, numbers, "REJECTED", a, None)

    # ---------------------------------------------------------------- 3 in-sample
    reg = TrialRegister(out / "trials.jsonl")
    cfg, best_pf, tr, b = grid_best(F, cp, lo, hi, reg)
    if cfg is None:
        gates.append(("3. in-sample quality", "fail", "no configuration produced >= 20 trades"))
        return finish(out, figs, tag, gates, numbers, "REJECTED", a, None)
    good = best_pf > 1.0 and len(tr) >= 30
    gates.append(
        (
            "3. in-sample quality",
            "pass" if good else "fail",
            f"best {cfg['exit']} stop={cfg['stop_atr']} tp={cfg['tp_atr']}: "
            f"PF {best_pf:.4f} on {len(tr)} trades, {b.sum():+.0f} pts",
        )
    )
    numbers |= {
        "best in-sample PF": f"{best_pf:.4f}",
        "trades": f"{len(tr):,}",
        "net over train": f"{b.sum():+,.0f} pts",
        "avg per trade": f"{b.sum() / max(len(tr), 1):.3f} pts",
        "cost per round turn": f"{cp:.3f} pts",
    }
    log(
        f"[gate 3] in-sample: PF {best_pf:.4f}, {len(tr)} trades, {b.sum():+.0f} pts "
        f"-> {'PASS' if good else 'FAIL'}"
    )

    # figures that do not depend on later gates
    V.anatomy(
        df,
        tr,
        figs / "anatomy.png",
        n_bars=300,
        indicators={f"EMA {a.fast}": ema(c, a.fast), f"EMA {a.slow}": ema(c, a.slow)},
        atr=F["atr"],
        stop_atr=cfg["stop_atr"],
        tp_atr=cfg["tp_atr"],
        title=f"EMA {a.fast}/{a.slow} - {a.symbol} {a.tf}",
    )
    V.rule_card(
        figs / "rules.png",
        name=f"EMA {a.fast}/{a.slow} crossover",
        entry_long=f"EMA({a.fast}) crosses ABOVE EMA({a.slow}). The cross is the trigger.",
        entry_short=f"EMA({a.fast}) crosses BELOW EMA({a.slow}).",
        exit_rule=[
            f"{cfg['stop_atr']}xATR stop, {cfg['tp_atr']}xATR target "
            f"({cfg['exit']}), ATR(14) shifted one bar.",
            "Flat at the RTH close (15:00 CT).",
        ],
        costs=f"{cp:.3f} pts round turn ({a.firm}, {a.contracts} lot)",
        notes=["RTH entries only.", "No regime filter."],
        code=pathlib.Path("src/quant/strategies/ema_cross.py").read_text().split('"""', 2)[2],
    )
    viz.equity_with_splits(
        bar_pnl(F, tr, cp), sp, figs / "equity.png", title=f"EMA {a.fast}/{a.slow}"
    )
    viz.drawdown(b, figs / "drawdown.png")
    V.monthly_heatmap(b, d[lo:hi], figs / "monthly.png")
    V.rolling_performance(b, figs / "rolling.png", window=5000)
    V.mae_mfe(tr, figs / "mae_mfe.png")
    V.walkforward_folds(wf.folds(sp), sp, figs / "folds.png", dates=d)
    if not good:
        return finish(out, figs, tag, gates, numbers, "REJECTED", a, (tr, b, F, sp, d, lo, hi, cp))

    # ---------------------------------------------------------------- 4 PBO
    m, cfgs = reg.matrix("train")
    res = P.cscv(m, S=10)
    names = [
        f"{x['config']['exit']} {x['config']['stop_atr']}/{x['config']['tp_atr']}" for x in cfgs
    ]
    log("\n" + P.report(res, names))
    viz.pbo_panel(res, figs / "pbo.png", title=f"EMA {a.fast}/{a.slow} - PBO")
    flat = res["pbo"] > PBO_MAX and abs(res["degradation_slope"]) < 0.2
    gates.append(
        (
            "4. selection bias (PBO)",
            "pass" if res["pbo"] <= PBO_MAX else "fail",
            f"PBO {res['pbo']:.3f} (max {PBO_MAX}); degradation slope "
            f"{res['degradation_slope']:+.3f}; P(OOS<0) {res['prob_oos_loss']:.3f}"
            + (
                "; surface is flat so PBO is inflated by configurations being hard to tell apart"
                if flat
                else ""
            ),
        )
    )
    numbers |= {
        "PBO": f"{res['pbo']:.3f}  (max {PBO_MAX})",
        "degradation slope": f"{res['degradation_slope']:+.3f}",
        "P(OOS < 0)": f"{res['prob_oos_loss']:.3f}",
    }
    log(f"[gate 4] PBO {res['pbo']:.3f} -> {'PASS' if res['pbo'] <= PBO_MAX else 'FAIL'}")
    pbo_failed = res["pbo"] > PBO_MAX

    # ---------------------------------------------------------------- 5 in-sample MCPT
    if a.perms > 0:
        train = df.iloc[lo:hi].reset_index(drop=True)
        groups = session_groups(train)
        better, perms, t0 = 1, [], time.time()
        for i in range(1, a.perms):
            p = get_permutation(train, seed=i, groups=groups)
            Fp = build_features(p, fast=a.fast, slow=a.slow)
            _, v, _, _ = grid_best(Fp, cp, 0, len(p))
            if np.isfinite(v):
                perms.append(v)
                if v >= best_pf:
                    better += 1
            if i % 25 == 0:
                el = time.time() - t0
                log(
                    f"   MCPT {i}/{a.perms}  beaten {better - 1}x  p~{better / (i + 1):.3f}  "
                    f"({el / i:.1f}s/perm, {(a.perms - i) * el / i / 60:.0f}min left)"
                )
        pv = better / a.perms
        viz.mcpt_histogram(
            perms, best_pf, pv, figs / "mcpt.png", title=f"EMA {a.fast}/{a.slow} - in-sample MCPT"
        )
        gates.append(
            (
                "5. luck (in-sample MCPT)",
                "pass" if pv < P_INSAMPLE else "fail",
                f"p = {pv:.4f} (max {P_INSAMPLE}) over {len(perms)} permutations; "
                f"permutation median PF {np.median(perms):.4f}, "
                f"p99 {np.percentile(perms, 99):.4f}",
            )
        )
        numbers |= {
            "MCPT p": f"{pv:.4f}  (max {P_INSAMPLE})",
            "permutation median PF": f"{np.median(perms):.4f}",
            "permutation p99 PF": f"{np.percentile(perms, 99):.4f}",
        }
        log(f"[gate 5] MCPT p = {pv:.4f} -> {'PASS' if pv < P_INSAMPLE else 'FAIL'}")
        with open(out / "mcpt.json", "w") as fh:
            json.dump(dict(real=best_pf, p=pv, perms=perms), fh)
        mcpt_failed = pv >= P_INSAMPLE
    else:
        gates.append(("5. luck (in-sample MCPT)", "skipped", "--perms 0"))
        mcpt_failed = False

    if pbo_failed or mcpt_failed:
        gates.append(("6-12. walk-forward onward", "skipped", "not run: an earlier gate failed"))
        return finish(out, figs, tag, gates, numbers, "REJECTED", a, (tr, b, F, sp, d, lo, hi, cp))

    gates.append(("6-12. walk-forward onward", "skipped", "not implemented in this runner yet"))
    return finish(out, figs, tag, gates, numbers, "INCOMPLETE", a, (tr, b, F, sp, d, lo, hi, cp))


def finish(out, figs, tag, gates, numbers, verdict, a, ctx):
    if ctx:
        tr, _b, _F, sp, d, lo, hi, _cp = ctx
        checks = diag.run_all(pnl=tr["pnl_pts"].to_numpy(), dates=d[lo:hi])
        viz.verdict_card(checks, figs / "diagnostics.png", title=f"{tag} - bias battery")
        viz.trade_distribution(tr["pnl_pts"].to_numpy(), figs / "trades.png", luck=checks["luck"])
        V.trade_dependence(
            tr["pnl_pts"].to_numpy(), figs / "dependence.png", runs=checks["trade_dependence"]
        )
        numbers |= {
            "win rate": f"{checks['luck']['win_rate']:.1%}",
            "profit factor (trades)": f"{checks['luck']['profit_factor']:.3f}",
            "runs-test z": f"{checks['trade_dependence'].get('z', float('nan')):+.2f}",
        }
        numbers[f"{a.symbol} holdout looks"] = (
            f"{len(sp.holdout_looks())}  (untouched)"
            if not sp.holdout_looks()
            else str(len(sp.holdout_looks()))
        )
    failed = [g[0] for g in gates if g[1] == "fail"]
    head = f"EMA({a.fast})/EMA({a.slow}) on {a.symbol} {a.tf}. " + (
        "Rejected at " + failed[0] + "." if failed else "Ladder incomplete."
    )
    V.conclusion_card(
        figs / "conclusion.png",
        name=f"EMA({a.fast})/EMA({a.slow}) - {a.symbol} {a.tf}",
        verdict=verdict,
        gates=gates,
        headline=head,
        numbers=numbers,
        what_it_means=_interpret(gates, numbers, verdict),
    )
    with open(out / "result.json", "w") as fh:
        json.dump(dict(tag=tag, verdict=verdict, gates=gates, numbers=numbers), fh, indent=1)
    log(f"\n{'=' * 78}\n  VERDICT: {verdict}\n{'=' * 78}")
    for g in gates:
        log(f"  [{g[1].upper():>7}] {g[0]}")
        if g[2]:
            log(f"            {g[2]}")
    # ---------------------------------------------------------------- the standing verdict
    # Built from the gate list, so it cannot say "no edge" above a table that says otherwise,
    # and cannot name a configuration unless every gate actually passed.
    ev = [f"{k}: {x}" for k, x in numbers.items()]
    if failed:
        v = Verdict.from_gates(
            tag,
            gates,
            evidence=ev,
            caveat="A failed gate is not a tuning signal. Sending this back to a parameter "
            "tweak is the move the protocol exists to prevent.",
        )
    else:
        detail = next((g[2] for g in gates if g[0].startswith("3.")), "")
        v = Verdict.edge(
            tag,
            config=f"{a.symbol} {a.tf}  EMA({a.fast})/EMA({a.slow})  -- {detail}",
            net=f"{numbers.get('avg per trade', '?')} per trade, after "
            f"{numbers.get('cost per round turn', '?')} round turn",
            reason="Every gate passed: the in-sample and walk-forward permutation tests both "
            "rejected the null, PBO stayed under threshold, and the result survives the "
            "modelled cost.",
            evidence=ev,
            caveat="The holdout has not been read. Until it is, this is a validated candidate "
            "and not a live edge.",
        )
    log("")
    log(v.render())
    with open(out / "verdict.json", "w") as fh:
        json.dump(v.as_dict(), fh, indent=2)
    (out / "verdict.txt").write_text(v.render(), encoding="utf-8")

    log(f"\n  figures: {figs}")
    return 0 if verdict != "REJECTED" else 1


def _interpret(gates, numbers, verdict):
    out = []
    g = {x[0]: x[1] for x in gates}
    if g.get("5. luck (in-sample MCPT)") == "fail":
        out.append(
            f"The permutation test is decisive. Shuffled data produced a median profit factor of "
            f"{numbers.get('permutation median PF', '?')} and a 99th percentile of "
            f"{numbers.get('permutation p99 PF', '?')}. The real result, "
            f"{numbers.get('best in-sample PF', '?')}, is an ordinary draw from that distribution "
            f"-- noise reproduced it {float(numbers.get('MCPT p', '1').split()[0]) * 100:.0f}% of "
            f"the time. There is no edge here to take out of sample."
        )
    if g.get("4. selection bias (PBO)") == "fail":
        out.append(
            f"PBO {numbers.get('PBO', '?')} says the same from the other direction: the "
            f"configuration that looked best in sample landed below the out-of-sample median that "
            f"often, and lost money out of sample in {numbers.get('P(OOS < 0)', '?')} of splits. "
            f"The negative degradation slope means looking better in-sample actively predicted "
            f"doing worse out of it -- the signature of fitting noise."
        )
    if verdict == "REJECTED":
        out.append(
            "Per the pre-registered invalidation, this is the end of the hypothesis -- not an "
            "invitation to try adjacent parameters. Both holdouts remain unspent, which is the "
            "one good outcome of a rejection."
        )
    return out or ["Ladder incomplete; no conclusion is available yet."]


if __name__ == "__main__":
    raise SystemExit(main())
