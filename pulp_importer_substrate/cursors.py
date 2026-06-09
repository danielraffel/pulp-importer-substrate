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

from pathlib import Path

import clang.cindex as ci

_RUNTIME_REF_KINDS = {
    ci.CursorKind.ARRAY_SUBSCRIPT_EXPR,
}

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
    function result) rather than being literal-only. Drives confidence: a
    literal-only arg is extractable; a computed one is flagged, never guessed.
    """
    for n in walk(cursor):
        if n.kind in _RUNTIME_REF_KINDS:
            return True
        if n.kind == ci.CursorKind.DECL_REF_EXPR:
            ref = n.referenced
            if ref is not None and ref.kind in (
                ci.CursorKind.VAR_DECL,
                ci.CursorKind.PARM_DECL,
                ci.CursorKind.FIELD_DECL,
            ):
                return True
    return False


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
