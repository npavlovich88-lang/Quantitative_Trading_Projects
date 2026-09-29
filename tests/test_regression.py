"""Regression lock against the real MNQ data.

`baseline.json` holds a SHA-256 of every preset's per-trade P&L vector, captured before the
2026-09-22 lint/format cleanup. Any future change that alters a single trade by a single tick
breaks these tests loudly.

That is the point: refactors, dependency bumps and "harmless" tidy-ups are exactly how a
verified backtest quietly stops being the thing that was verified. When one of these fails,
either the change was wrong, or it was right and the baseline needs regenerating deliberately
with a note in the decision log -- never silently.

Marked `slow` because it needs the CSVs on F:.  Run the fast suite with:  pytest -m "not slow"
"""

import hashlib
import json
import pathlib

from quant.data_splits import DATA_ROOT

import numpy as np
import pytest

from quant import strategy as S

CSV = pathlib.Path(DATA_ROOT) / "MNQ_OHLCV" / "MNQ_10m_full_session_ohlcv.csv"
BASELINE = pathlib.Path(__file__).parent / "baseline.json"
PRESETS = {
    "A": ("fixed", 6.0, 2.0),
    "B": ("fixed", 3.0, None),
    "C": ("st_flip", None, None),
    "D": ("fixed", 6.0, 6.0),
}

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def feats():
    if not CSV.exists():
        pytest.skip(f"market data not available at {CSV}")
    return S.build_features(S.load_data(str(CSV), "America/Chicago"), "08:30-15:00")


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE.exists():
        pytest.skip("no baseline captured")
    return json.loads(BASELINE.read_text())


@pytest.mark.parametrize("name", list(PRESETS))
def test_preset_trades_are_unchanged(feats, baseline, name):
    exit_mode, stop, tp = PRESETS[name]
    tr = S.simulate(feats, exit_mode, stop, tp, 2.5)
    sig = hashlib.sha256(tr["pnl_pts"].round(8).to_numpy().tobytes()).hexdigest()[:16]
    want = baseline[name]
    assert len(tr) == want["n"], f"preset {name}: trade count {len(tr)} != baseline {want['n']}"
    assert sig == want["sig"], (
        f"preset {name}: per-trade P&L changed (sha {sig} != {want['sig']}). "
        f"Total P&L now {tr['pnl_pts'].sum():.4f} vs baseline {want['pnl']}."
    )


def test_features_are_unchanged(feats, baseline):
    got = {
        k: (
            round(float(np.nansum(feats[k])), 6)
            if np.asarray(feats[k]).dtype != bool
            else int(np.sum(feats[k]))
        )
        for k in ("c", "atr", "el", "es", "ok_long", "ok_short", "st_bull", "inwin")
    }
    assert got == baseline["_features"], (
        f"feature checksums drifted:\n  got  {got}\n  want {baseline['_features']}"
    )
