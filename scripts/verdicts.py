#!/usr/bin/env python3
"""Every completed run's verdict, in one place.

    python scripts/verdicts.py

Reads whatever is in out/ and prints the standard block for each: edge or no edge, the
configuration if there is one, and the single binding reason. Information studies are replayed
from their saved statistics.npz, so no permutations are recomputed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quant.verdict import Verdict, from_information


def information_verdict(name: str, d: pathlib.Path) -> Verdict | None:
    """Replay an information study's verdict from its saved statistics."""
    npz = d / "statistics.npz"
    if not npz.exists():
        return None
    z = np.load(npz, allow_pickle=False)
    return from_information(
        name,
        z["real"],
        z["null"],
        [str(x) for x in z["labels"]],
        [str(x) for x in z["symbols"]],
        [str(x) for x in z["timeframes"]],
        [int(x) for x in z["horizons"]],
        np.array([str(x) for x in z["families"]]) if "families" in z.files else None,
    )


def ladder_verdict(name: str, d: pathlib.Path) -> Verdict | None:
    """Replay a traded ladder's verdict from its result.json."""
    f = d / "result.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text(encoding="utf-8"))
    gates = [tuple(g) for g in r.get("gates", [])]
    nums = r.get("numbers", {})
    ev = [f"{k}: {v}" for k, v in nums.items()]
    if any(g[1] == "fail" for g in gates):
        return Verdict.from_gates(
            name,
            gates,
            evidence=ev,
            caveat=(
                "A failed gate is not a tuning signal. Sending this back to a parameter tweak is "
                "the move the protocol exists to prevent."
            ),
        )
    if not gates:
        # An older run that stored only its headline verdict. Report that, and say why the
        # block is thin -- silently upgrading it to INCONCLUSIVE would misreport a real result.
        stored = r.get("verdict", "")
        return Verdict.no_edge(
            name,
            reason=(
                f"Recorded verdict {stored!r} from an earlier run that did not store its gate "
                f"list, so the binding gate cannot be named from the file. The run itself was "
                f"conclusive; only this replay is thin."
            ),
            evidence=ev or ["no per-gate detail stored"],
        )
    return Verdict.inconclusive(
        name,
        reason=(
            "Every gate recorded a pass, but no configuration and expectancy after costs were "
            "stored with the run, so this cannot be reported as a tradeable edge from the file "
            "alone."
        ),
        evidence=ev,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    root = pathlib.Path(a.out)
    if not root.exists():
        print(f"no such directory: {root}")
        return 1

    found = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        v = information_verdict(d.name, d) or ladder_verdict(d.name, d)
        if v is None:
            continue
        found += 1
        print(v.render())
        print()
        (d / "verdict.txt").write_text(v.render(), encoding="utf-8")
        (d / "verdict.json").write_text(json.dumps(v.as_dict(), indent=2), encoding="utf-8")

    if not found:
        print(f"no completed runs under {root}")
        return 1
    print(f"{found} run(s). Each one's block is also written to out/<run>/verdict.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
