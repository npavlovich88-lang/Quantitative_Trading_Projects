"""The trials register: every configuration we ever evaluated, on disk, append-only.

WHY THIS EXISTS

Bailey, Borwein, Lopez de Prado & Zhu, section 5.2, on the inputs to PBO:

    "the researcher must provide full information regarding the actual trials conducted, to
    avoid the file drawer problem ... Hiding trials will lead to an underestimation of the
    overfit, because each logit will be evaluated under a biased relative rank."

That is the whole problem with how backtests are normally reported. You try 300 things across
six months -- timeframes, regime thresholds, stop multiples, session windows, overnight
variants -- keep the best, and then compute an overfitting statistic over the 45 configurations
that happened to be in the final grid. The answer comes back reassuring and it is worthless,
because the search that actually produced the winner was six times larger.

A register fixes this only if it is written at the moment of evaluation, automatically, by the
code that does the evaluating. A list reconstructed afterwards from memory is exactly the file
drawer the paper is warning about.

WHAT COUNTS AS A TRIAL

Every configuration whose performance you LOOKED AT. Not every configuration you shipped.
If you ran it and saw the number, it is a trial, including the ones you rejected in disgust.

The paper's two caveats, both of which cut the other way:
  * padding with configurations designed to fail also biases the result -- "If a model
    configuration is obviously flawed, it should have never been tried in the first place."
  * for a guided search (an optimiser using earlier results to pick later ones), record the
    CONVERGED outcome of each search, not each intermediate step.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


def config_id(config: dict[str, Any]) -> str:
    """Stable short hash, so re-running the same configuration does not double-count it."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class TrialRegister:
    """Append-only JSONL of every evaluated configuration.

    Store the per-period P&L vector, not just a summary: CSCV needs the full series to re-rank
    configurations on arbitrary sub-blocks. Summaries cannot be re-sliced.
    """

    def __init__(self, path: str | pathlib.Path, *, session: str | None = None):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session = session or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._pnl_dir = self.path.parent / (self.path.stem + "_pnl")
        self._pnl_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- writing
    def record(
        self,
        config: dict[str, Any],
        *,
        segment: str,
        metrics: dict[str, float],
        pnl: np.ndarray | None = None,
        note: str = "",
    ) -> str:
        """Record one evaluated configuration. Returns its id."""
        cid = config_id({**config, "segment": segment})
        if pnl is not None:
            np.save(self._pnl_dir / f"{cid}.npy", np.asarray(pnl, dtype=np.float32))
        entry = {
            "id": cid,
            "when": dt.datetime.now().isoformat(timespec="seconds"),
            "session": self.session,
            "segment": segment,
            "config": config,
            "metrics": {
                k: (None if v is None or not np.isfinite(v) else round(float(v), 8))
                for k, v in metrics.items()
            },
            "has_pnl": pnl is not None,
            "note": note,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return cid

    # ---------------------------------------------------------------- reading
    def entries(self, segment: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if segment is None or e["segment"] == segment:
                out.append(e)
        return out

    def unique_trials(self, segment: str | None = None) -> list[dict]:
        """One row per distinct configuration -- re-running the same cell is not a new trial."""
        seen: dict[str, dict] = {}
        for e in self.entries(segment):
            seen[e["id"]] = e
        return list(seen.values())

    def n_trials(self, segment: str | None = None) -> int:
        return len(self.unique_trials(segment))

    def matrix(self, segment: str) -> tuple[np.ndarray, list[dict]]:
        """(T, N) per-period P&L matrix for CSCV, plus the configs, column-aligned.

        Only configurations with a stored P&L vector of the modal length are included --
        columns must be synchronous for the ranks to mean anything (the paper's condition (i)).
        """
        rows = [e for e in self.unique_trials(segment) if e["has_pnl"]]
        if not rows:
            return np.empty((0, 0)), []
        arrays, keep = [], []
        for e in rows:
            f = self._pnl_dir / f"{e['id']}.npy"
            if f.exists():
                arrays.append(np.load(f))
                keep.append(e)
        if not arrays:
            return np.empty((0, 0)), []
        lengths = [len(a) for a in arrays]
        modal = max(set(lengths), key=lengths.count)
        cols = [(a, e) for a, e in zip(arrays, keep, strict=True) if len(a) == modal]
        dropped = len(arrays) - len(cols)
        if dropped:
            print(f"  register: dropped {dropped} trial(s) whose P&L length != {modal}")
        m = np.column_stack([a for a, _ in cols]).astype(float)
        return m, [e for _, e in cols]

    # ---------------------------------------------------------------- reporting
    def summary(self) -> str:
        segs: dict[str, int] = {}
        for e in self.unique_trials():
            segs[e["segment"]] = segs.get(e["segment"], 0) + 1
        total = sum(segs.values())
        lines = [f"Trial register: {total} distinct configurations evaluated, {self.path}"]
        for s, n in sorted(segs.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {s:<24} {n:>5}")
        if total:
            lines.append(
                "  -> PBO must be computed over ALL of these, not just the final grid; "
                "a subset underestimates overfitting (Bailey et al. 5.2)."
            )
        return "\n".join(lines)
