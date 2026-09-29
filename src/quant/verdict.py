"""The block that ends every run: edge or no edge, the config if there is one, and why.

WHY THIS IS CODE AND NOT A HABIT

A run produces gates, p-values, PBO, cost tables and twenty figures. None of that answers the
only question that matters at the end -- can this be traded, with what settings, and for what
reason -- and a summary written by hand each time drifts from the numbers it is summarising.
This renders the answer FROM the gate results, so it cannot say "no edge" above a table that
says otherwise, or quote a p-value the run did not produce.

THE FOUR STATES, AND WHY THERE ARE FOUR RATHER THAN TWO

    NO EDGE        A gate failed. There is nothing to trade and nothing to tune.
    EDGE           Survived a TRADED ladder: costs charged, exits simulated, permutation tests
                   passed, PBO under threshold. Only this state may name a config to trade.
    SIGNAL ONLY    An information test found something, but no costs were charged and no exit
                   rule was simulated. This is NOT an edge and must never be traded on. Four
                   strategies in this project died between "signal present" and "money", so the
                   distinction earns its own state.
    INCONCLUSIVE   The run stopped for a reason that is not a result -- too few events, a data
                   gap, a coverage failure. "We could not tell" is an honest answer and must not
                   be rounded to "no".

THE BINDING REASON

Exactly one. Not a list of everything that looked wrong: the single gate that decided it. When
several gates fail, the FIRST one is binding, because everything downstream of it was computed
on a signal already known to be dead.

THE TRAP THIS IS BUILT TO AVOID

Printing a "best configuration" next to a "no edge" verdict. The best cell of a failed search is
the winner of a search through noise, and reads like a recommendation to anyone skimming. So
`config` is refused unless the state is EDGE.
"""

from __future__ import annotations

import dataclasses

from .information import family_pvalues, winner_support

NO_EDGE = "NO EDGE"
EDGE = "EDGE"
SIGNAL_ONLY = "SIGNAL ONLY"
INCONCLUSIVE = "INCONCLUSIVE"
STATES = (NO_EDGE, EDGE, SIGNAL_ONLY, INCONCLUSIVE)

_WIDTH = 92


class VerdictError(Exception):
    """Raised when a verdict would be misleading, e.g. a config attached to a failure."""


@dataclasses.dataclass(frozen=True)
class Verdict:
    """The end-of-run answer. Construct through the classmethods, not directly."""

    state: str
    name: str
    reason: str  # the ONE binding reason
    evidence: tuple[str, ...] = ()
    config: str = ""  # only permitted when state is EDGE
    net: str = ""  # expectancy after costs, only with a config
    caveat: str = ""  # what would break it / what would change the answer
    next_step: str = ""

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise VerdictError(f"state must be one of {STATES}, got {self.state!r}")
        if self.config and self.state != EDGE:
            raise VerdictError(
                f"a config may only be named when the state is {EDGE!r}. The best cell of a "
                f"failed search is the winner of a search through noise, and printing it here "
                f"reads as a recommendation."
            )
        if self.state == EDGE and not self.config:
            raise VerdictError("an EDGE verdict must name the configuration to trade")
        if self.state == EDGE and not self.net:
            raise VerdictError("an EDGE verdict must state the expectancy AFTER costs")
        if len(self.reason.strip()) < 20:
            raise VerdictError("the binding reason has to be a sentence, not a label")

    # ------------------------------------------------------------------ constructors
    @classmethod
    def no_edge(cls, name, reason, evidence=(), caveat="", next_step="") -> Verdict:
        return cls(NO_EDGE, name, reason, tuple(evidence), "", "", caveat, next_step)

    @classmethod
    def edge(cls, name, config, net, reason, evidence=(), caveat="", next_step="") -> Verdict:
        return cls(EDGE, name, reason, tuple(evidence), config, net, caveat, next_step)

    @classmethod
    def signal_only(cls, name, reason, evidence=(), caveat="", next_step="") -> Verdict:
        return cls(SIGNAL_ONLY, name, reason, tuple(evidence), "", "", caveat, next_step)

    @classmethod
    def inconclusive(cls, name, reason, evidence=(), caveat="", next_step="") -> Verdict:
        return cls(INCONCLUSIVE, name, reason, tuple(evidence), "", "", caveat, next_step)

    @classmethod
    def from_gates(cls, name, gates, evidence=(), caveat="", next_step="") -> Verdict:
        """Derive the verdict from a traded ladder's gate list of (label, 'pass'|'fail', detail).

        The first failure is binding. If nothing failed, the ladder completed and this is an EDGE
        -- but the caller still has to supply the config, so that path goes through `edge`.
        """
        failed = [g for g in gates if g[1] == "fail"]
        if not failed:
            raise VerdictError(
                "every gate passed, so the verdict is an EDGE and must name a configuration "
                "and its expectancy after costs -- use Verdict.edge() instead"
            )
        label, _, detail = failed[0]
        reason = f"Halted at gate '{label}'. {detail}".strip()
        rest = [f"also failed: {g[0]} -- {g[2]}" for g in failed[1:]]
        return cls.no_edge(name, reason, tuple(evidence) + tuple(rest), caveat, next_step)

    # ------------------------------------------------------------------ rendering
    def render(self) -> str:
        bar = "=" * _WIDTH
        out = [bar, f"  VERDICT - {self.name}", bar, "", f"  EDGE:  {self.state}", ""]

        if self.state == EDGE:
            out += ["  CONFIG TO TRADE", f"    {self.config}", f"    {self.net}", ""]
            out += ["  WHY IT WORKS", *_wrap(self.reason), ""]
        else:
            out += ["  BINDING REASON", *_wrap(self.reason), ""]
            out += ["  CONFIG TO TRADE", *_wrap(_no_config_text(self.state)), ""]

        if self.evidence:
            out += ["  EVIDENCE"]
            out += [f"    - {e}" for e in self.evidence]
            out += [""]
        if self.caveat:
            head = "  WHAT WOULD BREAK IT" if self.state == EDGE else "  WHAT WOULD CHANGE THIS"
            out += [head, *_wrap(self.caveat), ""]
        if self.next_step:
            out += ["  NEXT", *_wrap(self.next_step), ""]
        out += [bar]
        return "\n".join(out)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self) | {"edge": self.state == EDGE}


def _no_config_text(state: str) -> str:
    if state == SIGNAL_ONLY:
        return (
            "None yet. A signal measured without costs and without an exit rule is not a "
            "tradeable configuration, and naming one here would invite trading it."
        )
    if state == INCONCLUSIVE:
        return "None. The run did not reach a conclusion, so there is nothing to recommend."
    return (
        "None. The best cell of a failed search is the winner of a search through noise, "
        "not a setting."
    )


def _wrap(text: str, indent: str = "    ", width: int = _WIDTH - 6) -> list[str]:
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(indent + line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(indent + line)
    return out


def from_information(
    name: str,
    real,
    null,
    labels,
    syms,
    tfs,
    hors,
    fams=None,
    p_threshold: float = 0.05,
    spike_ratio: float = 2.5,
) -> Verdict:
    """The verdict for an information study, derived from its own statistics.

    Three outcomes, and the middle one is the point. A family p-value below the threshold is
    not automatically a finding: if it rests on one cell that towers over its own sub-family
    and has no support on the second market, the honest answer is INCONCLUSIVE with a named
    next step, not a pass. That is exactly the 2026-09-24 HMA/TEMA/VWAP case, where a
    nominal p = 0.042 reversed sign on one pre-committed out-of-sample look.
    """

    fp = family_pvalues(real, null)
    sup = winner_support(real, fams)
    si, ti, ci, hi = sup["index"]
    where = f"{syms[si]} {tfs[ti]} {labels[ci]} h={hors[hi]}"

    ev = [
        f"family of {real.size:,} cells, {fp['live_cells']:,} live, {fp['permutations']} "
        f"permutations, train only",
        f"max |t| {fp['observed_max_t']:.3f} at {where}; null 95th pct {fp['null_max_p95']:.3f}",
        f"max-statistic p = {fp['p_max']:.4f}",
        f"count statistic {fp['cells_over_own_null']} cells vs null median "
        f"{fp['null_count_median']:.0f}, p = {fp['p_count']:.4f}",
        f"winner is {sup['neighbour_ratio']:.1f}x its sub-family's median |t| "
        f"({sup['peer_median_abs_t']:.2f})",
        "same config on the other symbol: "
        + (", ".join(f"{x:+.2f}" for x in sup["cross_symbol_t"]) or "not live"),
    ]

    if fp["p_max"] >= p_threshold:
        return Verdict.no_edge(
            name,
            reason=(
                f"The family-corrected permutation test returned p = {fp['p_max']:.4f}. The best "
                f"of {fp['live_cells']:,} cells reached |t| = {fp['observed_max_t']:.3f}, and the "
                f"best of the same {fp['live_cells']:,} cells on shuffled markets reached "
                f"{fp['null_max_p95']:.3f} one time in twenty. Nothing here beats the noise "
                f"ceiling."
            ),
            evidence=ev,
            caveat=(
                "Only a mechanism that produces a larger effect, or a far larger sample, would "
                "change this. A different exit rule would not -- no exit can create direction "
                "that is not in the signal."
            ),
        )

    supported = sup["neighbour_ratio"] < spike_ratio and sup["cross_symbol_agrees"]
    if not supported:
        return Verdict.inconclusive(
            name,
            reason=(
                f"The family gate passed at p = {fp['p_max']:.4f}, but it is carried by a single "
                f"cell ({where}) that stands {sup['neighbour_ratio']:.1f}x above its own "
                f"sub-family's median and "
                + (
                    "has no support on the second market"
                    if not sup["cross_symbol_agrees"]
                    else "sits in an otherwise flat parameter field"
                )
                + ". A lone spike is presumed to be luck until an out-of-sample look says "
                "otherwise."
            ),
            evidence=ev,
            caveat=(
                "One pre-committed look at the validate segment on exactly this configuration "
                "settles it. That is cheaper and more decisive than more permutations, which "
                "would only refine a number that was never the question."
            ),
            next_step=f"Run the single-cell validate check on {where}.",
        )

    return Verdict.signal_only(
        name,
        reason=(
            f"The family gate passed at p = {fp['p_max']:.4f} and the winner is supported on both "
            f"the parameter axis and the second market. This is directional information, not an "
            f"edge: no costs were charged and no exit rule was simulated."
        ),
        evidence=ev,
        caveat=(
            "Four strategies in this project have died between 'signal present' and 'money'. "
            "The traded ladder with costs, exits, PBO and the walk-forward permutation test is "
            "what decides it."
        ),
        next_step=f"Run the traded ladder on {where}.",
    )
