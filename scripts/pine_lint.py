#!/usr/bin/env python3
"""Catch the Pine errors this project has actually shipped, before sending a file.

    python scripts/pine_lint.py pine/measured_context.pine

Two broken indicators reached the user before this existed:

  1. `ta.vwap()` called inside an `if` block. Pine functions that carry series state must be
     evaluated on EVERY bar; calling one conditionally corrupts it silently -- it compiles, it
     plots, and the values are wrong. This one was only caught from a screenshot.
  2. A string literal split across two lines, because a `\\n` written through a Python heredoc
     became a real newline instead of the escape. That one does not compile:
     "Missing enclosing character in the literal string."

Neither is exotic and both are mechanically detectable. This is not a Pine parser -- it checks
the specific mistakes that have been made here, which is the useful subset.
"""

from __future__ import annotations

import pathlib
import re
import sys

# Functions carrying series state. Conditional evaluation corrupts them without erroring.
STATEFUL = (
    "ta.vwap",
    "ta.ema",
    "ta.sma",
    "ta.rma",
    "ta.wma",
    "ta.atr",
    "ta.rsi",
    "ta.change",
    "ta.highest",
    "ta.lowest",
    "ta.stdev",
    "ta.variance",
    "ta.cum",
    "ta.barssince",
    "ta.crossover",
    "ta.crossunder",
    "ta.valuewhen",
    "ta.pivothigh",
    "ta.pivotlow",
    "request.security",
    "request.security_lower_tf",
)


def strip_comment(line: str) -> str:
    """Remove a trailing // comment without cutting inside a string literal."""
    out, in_str, quote = [], False, ""
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            if ch == "\\":
                out.append(line[i : i + 2])
                i += 2
                continue
            if ch == quote:
                in_str = False
            out.append(ch)
        elif ch in "\"'":
            in_str, quote = True, ch
            out.append(ch)
        elif ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def lint(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    errs: list[str] = []

    if not re.search(r"^//@version=\d+", text, re.M):
        errs.append("missing //@version= directive")
    if not re.search(r"^\s*(indicator|strategy|library)\s*\(", text, re.M):
        errs.append("no indicator(), strategy() or library() declaration")

    depth = 0
    for i, raw in enumerate(lines, 1):
        code = strip_comment(raw)

        # 1. unbalanced quotes -- a literal split across lines
        if code.count('"') % 2:
            errs.append(
                f"line {i}: unbalanced double quote (string split across lines?)\n    {raw}"
            )
        if code.count("'") % 2:
            errs.append(f"line {i}: unbalanced single quote\n    {raw}")

        # 2. comma-separated declarations, which Pine does not accept
        if re.search(r",\s*var\s+\w+", code):
            errs.append(f"line {i}: comma-separated `var` declarations are invalid Pine\n    {raw}")
        if re.search(r":=[^,]*,\s*\w+\s*:=", code):
            errs.append(f"line {i}: comma-separated `:=` assignments are invalid Pine\n    {raw}")

        # 3. stateful function called at an indent, i.e. inside if/for/while
        indent = len(raw) - len(raw.lstrip())
        if indent > 0 and raw.strip() and not raw.strip().startswith("//"):
            for fn in STATEFUL:
                if re.search(rf"(?<![\w.]){re.escape(fn)}\s*\(", code):
                    errs.append(
                        f"line {i}: `{fn}()` called inside a block (indent {indent}). "
                        f"Stateful functions must run on EVERY bar.\n    {raw.strip()}"
                    )
                    break

        depth += code.count("(") - code.count(")")

    if depth:
        errs.append(f"unbalanced parentheses across the file: {depth:+d}")
    return errs


def main() -> int:
    if len(sys.argv) < 2:
        targets = sorted(pathlib.Path("pine").glob("*.pine"))
    else:
        targets = [pathlib.Path(a) for a in sys.argv[1:]]
    if not targets:
        print("no .pine files found")
        return 1

    bad = 0
    for t in targets:
        errs = lint(t)
        status = "OK" if not errs else f"{len(errs)} problem(s)"
        print(f"{t}  --  {status}")
        for e in errs:
            print(f"  {e}")
        bad += len(errs)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
