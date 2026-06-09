"""Pure token helpers over a libclang cursor's spelling stream.

Every function here is pure (no libclang calls beyond `c.get_tokens()` in
`toks`) and framework-agnostic — they operate on lists of token spellings. This
is where the trailing-dot float fix lives, in ONE place: both importers consume
`numeric_seq` from here, so the regression that previously had to be fixed in
two copies can now only live in one.
"""
from __future__ import annotations

import re


def toks(c) -> list[str]:
    return [t.spelling for t in c.get_tokens()]


def first_string(tokens: list[str]) -> str | None:
    for t in tokens:
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
            return t[1:-1]
    return None


def all_strings(tokens: list[str]) -> list[str]:
    return [t[1:-1] for t in tokens if len(t) >= 2 and t[0] == '"' and t[-1] == '"']


def first_int(tokens: list[str]) -> int | None:
    for t in tokens:
        if re.fullmatch(r"[+-]?\d+", t):
            return int(t)
    return None


# C++ float/int literal spellings, INCLUDING the trailing-dot form (`0.`, `100.`)
# that real framework / DSP source uses heavily — e.g. a default+range written as
# `(..., 0., 0., 100.0, 0.01, "%")`, or `-60.f` / `1000.`. A naive `\d*\.?\d+`
# pattern silently drops the trailing-dot forms (no digit after the dot), losing
# the default and range of any param spelled that way. The hermetic stub fixtures
# only used `0.0`-style literals, so the goldens missed it; the bug had to be
# fixed twice (once per importer) before this helper was shared. The three
# explicit alternatives below are `<digits>.<digits?>`, `.<digits>`, and
# `<digits>` — proven equivalent to the legacy `\d+\.?\d*|\.\d+` core over every
# token shape (locked by a test in this package).
_NUM_RE = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFlL]?")


def numeric_seq(tokens: list[str]) -> list[float]:
    """Numeric literals in order, honoring a leading unary minus."""
    vals: list[float] = []
    neg = False
    for t in tokens:
        if t == "-":
            neg = True
            continue
        if t == "+":
            continue
        m = _NUM_RE.fullmatch(t)
        if m:
            v = float(t.rstrip("fFlL"))
            vals.append(-v if neg else v)
            neg = False
        else:
            neg = False  # a non-numeric token breaks a pending sign
    return vals


def first_bool(tokens: list[str]) -> bool | None:
    for t in tokens:
        if t == "true":
            return True
        if t == "false":
            return False
    return None
