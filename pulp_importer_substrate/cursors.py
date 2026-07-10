"""libclang cursor traversal + generic AST predicates.

Framework-agnostic helpers for walking a translation unit, deciding whether a
cursor belongs to the user's main file, detecting computed (non-literal)
arguments, and locating loops / methods. None of these name a framework.

`walk` is the hardened variant: real framework headers can surface a
`CursorKind` id the installed `clang.cindex` enum doesn't know (e.g. a newer
template-argument kind), which raises `ValueError`. `walk` skips yielding such
a cursor but still descends into its children, so a real `--framework-path`
parse degrades to a partial extraction rather than aborting. On a clean parse
(every cursor kind resolves) this is identical to a plain pre-order walk.
"""
from __future__ import annotations

import re
from pathlib import Path

import clang.cindex as ci

_RUNTIME_REF_KINDS = {
    ci.CursorKind.ARRAY_SUBSCRIPT_EXPR,
}

# A call expression as an argument is a runtime-COMPUTED value (a function/method
# result), never a literal — flag it so its value is never guessed. This is the
# blindness the token-based reader had: `InitDouble("Gain", DefaultGain(), ...)`
# looked literal because `arg_is_computed` only inspected variable/field refs.
_COMPUTED_EXPR_KINDS = {
    ci.CursorKind.CALL_EXPR,
}

# libclang literal cursor kinds. Reading numeric values from THESE (not from the
# call's token stream) is what sees through a `#define`: `MACRO_MIN` expands to a
# FLOATING_LITERAL in the AST, whereas the token stream of the enclosing call
# still shows the macro NAME `MACRO_MIN`. The token reader therefore silently
# recovers zero numbers for a macro-table range; the AST reader recovers them.
_NUMERIC_LITERAL_KINDS = {
    ci.CursorKind.INTEGER_LITERAL,
    ci.CursorKind.FLOATING_LITERAL,
}

_LITERAL_NUM_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")

_LOOP_KINDS = {
    ci.CursorKind.FOR_STMT, ci.CursorKind.WHILE_STMT,
    ci.CursorKind.CXX_FOR_RANGE_STMT, ci.CursorKind.DO_STMT,
}


def _cursor_kind(cursor):
    """Safe cursor.kind: real framework headers can surface a CursorKind id the
    installed clang.cindex enum doesn't know (e.g. a newer template-argument
    kind), which raises ValueError. Return None instead of crashing so a real
    --framework-path parse degrades to a partial extraction rather than
    aborting."""
    try:
        return cursor.kind
    except ValueError:
        return None


def walk(cursor):
    # Only yield cursors whose kind resolves in the installed clang.cindex enum;
    # still descend into the children of an unknown-kind cursor. This keeps every
    # downstream `n.kind` access safe even when a real --framework-path parse
    # surfaces a CursorKind id the binding doesn't know (degrade, don't crash).
    if _cursor_kind(cursor) is not None:
        yield cursor
    try:
        children = cursor.get_children()
    except ValueError:
        return
    for ch in children:
        yield from walk(ch)


def in_main_file(cursor, main: Path) -> bool:
    try:
        return cursor.location.file is not None and Path(cursor.location.file.name) == main
    except Exception:
        return False


def arg_is_computed(cursor) -> bool:
    """True if the argument depends on a runtime value (variable, subscript,
    function-call result) rather than being literal-only. Drives confidence: a
    literal-only arg is extractable; a computed one is flagged, never guessed.

    Flags three shapes of non-literal argument:
      * an array subscript (``_RUNTIME_REF_KINDS``);
      * a reference to a variable / parameter / field / **function** — a
        ``DECL_REF_EXPR`` whose referent is data or a callee;
      * a **call expression** (``_COMPUTED_EXPR_KINDS``) — the previously-missed
        ``InitDouble("Gain", DefaultGain(), ...)`` case, where the result is a
        runtime value even though the token stream looks call-free.
    """
    for n in walk(cursor):
        k = n.kind
        if k in _RUNTIME_REF_KINDS or k in _COMPUTED_EXPR_KINDS:
            return True
        if k == ci.CursorKind.DECL_REF_EXPR:
            ref = n.referenced
            try:
                ref_kind = ref.kind if ref is not None else None
            except ValueError:
                ref_kind = None
            if ref_kind in (
                ci.CursorKind.VAR_DECL,
                ci.CursorKind.PARM_DECL,
                ci.CursorKind.FIELD_DECL,
                ci.CursorKind.FUNCTION_DECL,
                ci.CursorKind.CXX_METHOD,
            ):
                return True
    return False


def numeric_literals(cursor) -> list[float]:
    """Numeric literal VALUES under ``cursor``, read from the AST in source
    order — the macro-transparent counterpart to ``tokens.numeric_seq``.

    A ``#define``d bound (``InitDouble("M", 50.0, MACRO_MIN, MACRO_MAX, 1.0)``)
    expands to ``FLOATING_LITERAL`` / ``INTEGER_LITERAL`` cursors in the AST, so
    reading the literals here recovers ``[50.0, 0.0, 100.0, 1.0]`` where the
    call's token stream only shows ``[50.0, 1.0]`` plus the macro *names*. That
    silent gap is what made a macro-table range extract as an EMPTY range at
    high confidence, with no diagnostic (the S4 bug).

    A literal that sits inside a macro expansion has an EMPTY token stream (a
    known libclang quirk); such a literal is counted but its value cannot be
    recovered here, so a caller comparing ``len(numeric_literals(call))`` against
    an overload's expected arity still detects the arity mismatch even when the
    value itself is unreadable. Values that can be read are returned; unreadable
    ones are skipped, so the returned list is "every macro-transparent literal
    whose spelling was recoverable."

    Caveat — SIGN: a negated literal (``-60.0``) is a ``UNARY_OPERATOR`` wrapping
    a positive ``FLOATING_LITERAL``, so the recovered value is the MAGNITUDE
    (``60.0``); the sign lives on the operator, not the literal. This is fine for
    the arity/never-silent-empty use and for the positive ranges macro tables
    usually hold (Hz, %, ms), but a caller that needs signed macro bounds should
    pair this with token-level sign handling (see ``tokens.numeric_seq``)."""
    vals: list[float] = []
    for n in walk(cursor):
        if n.kind not in _NUMERIC_LITERAL_KINDS:
            continue
        spelling = None
        for t in n.get_tokens():
            spelling = t.spelling
            break
        if spelling is None:
            continue  # macro-expansion literal: countable but value unreadable
        m = _LITERAL_NUM_RE.match(spelling)
        if not m:
            continue
        try:
            vals.append(float(m.group(0)))
        except ValueError:
            continue
    return vals


def literal_count(cursor) -> int:
    """Count of numeric literal cursors under ``cursor`` (INTEGER/FLOATING),
    INCLUDING macro-expansion literals whose value is unreadable. This is the
    arity a caller checks against an overload's expected numeric-arg count to
    emit an arity-mismatch diagnostic — it never under-counts a macro range the
    way the token reader does."""
    n_lits = 0
    for n in walk(cursor):
        if n.kind in _NUMERIC_LITERAL_KINDS:
            n_lits += 1
    return n_lits


def find_method(tu, main: Path, name: str):
    for n in walk(tu.cursor):
        if n.kind == ci.CursorKind.CXX_METHOD and n.spelling == name and n.is_definition() \
                and in_main_file(n, main):
            return n
    return None


def find_loops(tu, main: Path) -> list[tuple[int, int]]:
    spans = []
    for n in walk(tu.cursor):
        if n.kind in _LOOP_KINDS and in_main_file(n, main):
            e = n.extent
            spans.append((e.start.offset, e.end.offset))
    return spans


def in_loop(call, loops: list[tuple[int, int]]) -> bool:
    o = call.location.offset
    return any(s <= o <= e for s, e in loops)
