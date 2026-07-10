#!/usr/bin/env python3
"""Guarded libclang tests for the AST-literal / computed-arg helpers.

These exercise the fixes for the `arg_is_computed` blindness (S4) that affects
BOTH importers:

  - a function-call argument (`InitDouble("Gain", DefaultGain(), ...)`) is now
    flagged computed — the previously-missed case that let a runtime value
    extract as a high-confidence literal;
  - a `#define`-table range (`InitDouble("M", 50.0, MACRO_MIN, MACRO_MAX, 1.0)`)
    is RECOVERED from the AST by `numeric_literals` (macro-transparent), where
    the token reader `numeric_seq` only sees the macro NAMES and silently
    recovers a SHORT/empty range — the root of the empty-range-at-0.92 bug.

Guarded with the same `skipUnless(libclang)` pattern as `tests/test_substrate.py`
so the suite still runs where the binding is absent.

Run: python3 tests/test_cursors_literals.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def _libclang_available() -> bool:
    try:
        import clang.cindex  # noqa: F401
        from pulp_importer_substrate import _configure_libclang
        _configure_libclang()
        clang.cindex.Index.create()
        return True
    except Exception:
        return False


@unittest.skipUnless(_libclang_available(), "libclang not installed/usable")
class CursorLiteralTest(unittest.TestCase):
    def _parse(self, src: str):
        import clang.cindex as ci
        from pulp_importer_substrate import _configure_libclang
        _configure_libclang()
        with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
            f.write(src)
            path = pathlib.Path(f.name)
        tu = ci.Index.create().parse(str(path), args=["-std=c++20"])
        return tu, path

    def _init_calls(self, tu):
        import clang.cindex as ci
        from pulp_importer_substrate import walk
        return [n for n in walk(tu.cursor)
                if n.kind == ci.CursorKind.CALL_EXPR and n.spelling == "Init"]

    _MACRO_SRC = (
        "#define MACRO_MIN 0.0\n"
        "#define MACRO_MAX 100.0\n"
        "double DefaultGain();\n"
        "struct P { void Init(const char*, double, double, double, double); };\n"
        "void f(P& p){\n"
        '  p.Init("Macro", 50.0, MACRO_MIN, MACRO_MAX, 1.0);\n'
        '  p.Init("Gain", DefaultGain(), 0.0, 12.0, 0.1);\n'
        "}\n"
    )

    def test_function_call_arg_flagged_computed(self):
        from pulp_importer_substrate import arg_is_computed
        tu, _ = self._parse(self._MACRO_SRC)
        gain_call = self._init_calls(tu)[1]  # the DefaultGain() one
        args = [a for a in gain_call.get_children()][1:]
        # arg[1] is DefaultGain() — a function-call result, must be flagged.
        self.assertTrue(arg_is_computed(args[1]))
        # The plain literal args are NOT flagged.
        self.assertFalse(arg_is_computed(args[2]))
        self.assertFalse(arg_is_computed(args[3]))

    def test_variable_ref_arg_still_flagged(self):
        from pulp_importer_substrate import arg_is_computed
        src = (
            "struct P { void Init(const char*, double, double, double, double); };\n"
            "void f(P& p){ double lo = 3.0;\n"
            '  p.Init("V", 0.0, lo, 10.0, 1.0);\n'
            "}\n"
        )
        tu, _ = self._parse(src)
        call = self._init_calls(tu)[0]
        args = [a for a in call.get_children()][1:]
        self.assertTrue(arg_is_computed(args[2]))  # `lo` variable reference

    def test_macro_range_recovered_from_ast_never_silent_empty(self):
        from pulp_importer_substrate import (
            numeric_literals, literal_count, numeric_seq, toks)
        tu, _ = self._parse(self._MACRO_SRC)
        macro_call = self._init_calls(tu)[0]
        # AST reader sees THROUGH the macros: recovers all four numeric args.
        self.assertEqual(numeric_literals(macro_call), [50.0, 0.0, 100.0, 1.0])
        self.assertEqual(literal_count(macro_call), 4)
        # The token reader only sees the macro NAMES -> it recovers FEWER numbers
        # (this is the silent gap the AST reader closes; never silent-empty now).
        tok_nums = numeric_seq(toks(macro_call))
        self.assertLess(len(tok_nums), 4)

    def test_computed_call_has_arity_mismatch_signal(self):
        from pulp_importer_substrate import numeric_literals
        tu, _ = self._parse(self._MACRO_SRC)
        gain_call = self._init_calls(tu)[1]  # Init("Gain", DefaultGain(), 0,12,0.1)
        # Only three numeric literals are present (default came from a call), so a
        # caller comparing against the 4-numeric InitDouble arity detects the
        # mismatch instead of emitting a high-confidence short range.
        self.assertEqual(numeric_literals(gain_call), [0.0, 12.0, 0.1])


if __name__ == "__main__":
    unittest.main()
