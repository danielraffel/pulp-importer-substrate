#!/usr/bin/env python3
"""Unit tests for the vendor-agnostic extraction substrate.

The token/id helpers are pure (operate on lists of token spellings / strings),
so they're tested WITHOUT libclang. The trailing-dot-float regression for
`numeric_seq` lives here so it can only ever exist in one place again. A guarded
libclang-dependent smoke runs only when the binding is importable.

Run: python3 tests/test_substrate.py
"""
from __future__ import annotations

import io
import json
import pathlib
import re
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pulp_importer_substrate import (  # noqa: E402
    CATEGORY_TO_PULP,
    all_strings,
    first_bool,
    first_int,
    first_string,
    fnv1a_u32,
    numeric_seq,
)
from pulp_importer_substrate import spi  # noqa: E402


class NumericSeqTest(unittest.TestCase):
    """The trailing-dot float (`0.`, `100.`) regression — the bug that used to
    be fixed twice (once per importer). It can now only live here."""

    def test_trailing_dot_floats_are_parsed(self):
        # Tokens as libclang emits them for InitDouble("Gain", 0., 0., 100.0, 0.01, "%").
        toks = ['"Gain"', ",", "0.", ",", "0.", ",", "100.0", ",", "0.01", ",", '"%"']
        self.assertEqual(numeric_seq(toks), [0.0, 0.0, 100.0, 0.01])

    def test_leading_dot_floats_are_parsed(self):
        self.assertEqual(numeric_seq([".5", ",", "1."]), [0.5, 1.0])

    def test_unary_minus_and_suffixes(self):
        # Real-source spellings: `-60.f`, `1000.`
        self.assertEqual(numeric_seq(["-", "60.f", ",", "1000."]), [-60.0, 1000.0])

    def test_plain_integers_and_exponents(self):
        self.assertEqual(numeric_seq(["1", "+", "2", "1e3", "1.5e-2"]),
                         [1.0, 2.0, 1000.0, 0.015])

    def test_non_numeric_token_breaks_pending_sign(self):
        # A `-` followed by a non-numeric token must not negate a later number.
        self.assertEqual(numeric_seq(["-", "foo", "5"]), [5.0])

    def test_empty(self):
        self.assertEqual(numeric_seq([]), [])

    def test_regex_core_equivalent_to_legacy_pattern(self):
        """Lock the proven equivalence of the unified regex to the legacy
        `\\d+\\.?\\d*|\\.\\d+` core, so a future edit that diverges them is
        caught here rather than via golden drift."""
        legacy = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFlL]?")
        import itertools
        alphabet = "01.eE+-fFlL"
        for length in range(1, 5):
            for combo in itertools.product(alphabet, repeat=length):
                t = "".join(combo)
                got = numeric_seq([t])
                legacy_match = bool(legacy.fullmatch(t))
                self.assertEqual(bool(got), legacy_match, f"token {t!r} disagrees")


class TokenHelperTest(unittest.TestCase):
    def test_first_string(self):
        self.assertEqual(first_string(["foo", '"Gain"', '"unit"']), "Gain")
        self.assertIsNone(first_string(["foo", "bar"]))

    def test_all_strings(self):
        self.assertEqual(all_strings(['"a"', ",", '"b"', "x"]), ["a", "b"])

    def test_first_int(self):
        self.assertEqual(first_int(["x", "-3", "4"]), -3)
        self.assertIsNone(first_int(["x", "1.5"]))

    def test_first_bool(self):
        self.assertTrue(first_bool(["x", "true"]))
        self.assertFalse(first_bool(["false", "true"]))
        self.assertIsNone(first_bool(["x", "y"]))


class IdsTest(unittest.TestCase):
    def test_fnv1a_known_vectors(self):
        # FNV-1a 32-bit reference: empty string is the offset basis.
        self.assertEqual(fnv1a_u32(""), 0x811C9DC5)
        # Stability: deterministic and within uint32.
        h = fnv1a_u32("Gain")
        self.assertEqual(h, fnv1a_u32("Gain"))
        self.assertTrue(0 <= h <= 0xFFFFFFFF)
        self.assertNotEqual(fnv1a_u32("Gain"), fnv1a_u32("gain"))


class MappingTest(unittest.TestCase):
    def test_category_map(self):
        self.assertEqual(CATEGORY_TO_PULP["effect"], "Effect")
        self.assertEqual(CATEGORY_TO_PULP["instrument"], "Instrument")
        self.assertEqual(CATEGORY_TO_PULP["midi_effect"], "MidiEffect")


class SpiShellTest(unittest.TestCase):
    """The vendor-agnostic SPI verb-envelope + JSON-stdio shell. Pure (no
    libclang): the shell only frames envelopes and dispatches to injected verb
    handlers, so it is tested with trivial in-memory handlers. This is the
    behaviour both importers' spi.py used to duplicate byte-for-byte; it can now
    only live here."""

    def _run(self, requests: list[dict], handlers=None) -> list[dict]:
        """Drive main_cli over in-memory stdio; return parsed responses."""
        if handlers is None:
            handlers = {
                "detect": lambda p: {"echo": p.get("project_dir")},
                "boom": lambda p: (_ for _ in ()).throw(RuntimeError("kaboom")),
            }
        stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
        stdout = io.StringIO()
        rc = spi.main_cli(handlers, framework_id="acme", importer_id="acme-spike",
                          stdin=stdin, stdout=stdout)
        self.assertEqual(rc, 0)
        return [json.loads(ln) for ln in stdout.getvalue().splitlines() if ln.strip()]

    def test_dispatch_calls_injected_handler(self):
        [resp] = self._run([{
            "spi_version": 0, "verb": "detect", "id": "d1",
            "payload": {"project_dir": "/proj"},
        }])
        self.assertEqual(resp["spi_version"], 0)
        self.assertEqual(resp["id"], "d1")
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["result"], {"echo": "/proj"})

    def test_unimplemented_verb_is_honest(self):
        [resp] = self._run([{"spi_version": 0, "verb": "plan", "id": "p1",
                             "payload": {}}])
        self.assertFalse(resp["ok"])
        self.assertEqual(resp["error"]["code"], "unimplemented_verb")

    def test_spi_version_mismatch_is_loud(self):
        [resp] = self._run([{"spi_version": 99, "verb": "detect", "id": "v1",
                             "payload": {}}])
        self.assertFalse(resp["ok"])
        self.assertEqual(resp["error"]["code"], "spi_version_mismatch")

    def test_handler_exception_becomes_error_envelope(self):
        # A raising handler must not crash the wire — it becomes an honest error.
        [resp] = self._run([{"spi_version": 0, "verb": "boom", "id": "b1",
                             "payload": {}}])
        self.assertFalse(resp["ok"])
        self.assertEqual(resp["error"]["code"], "analyze_error")
        self.assertIn("kaboom", resp["error"]["message"])

    def test_bad_json_line_is_reported(self):
        stdin = io.StringIO("not json\n")
        stdout = io.StringIO()
        rc = spi.main_cli({}, framework_id="acme", importer_id="acme-spike",
                          stdin=stdin, stdout=stdout)
        self.assertEqual(rc, 0)
        [resp] = [json.loads(ln) for ln in stdout.getvalue().splitlines() if ln.strip()]
        self.assertFalse(resp["ok"])
        self.assertEqual(resp["error"]["code"], "bad_json")

    def test_batch_and_blank_lines(self):
        # Blank lines are skipped; every real request gets exactly one response,
        # ids preserved in order.
        reqs = [{"spi_version": 0, "verb": "detect", "id": str(i),
                 "payload": {"project_dir": f"/p{i}"}} for i in range(3)]
        stdin = io.StringIO(
            "\n".join(json.dumps(r) for r in reqs).replace("\n", "\n\n") + "\n")
        stdout = io.StringIO()
        spi.main_cli({"detect": lambda p: {"echo": p.get("project_dir")}},
                     framework_id="acme", importer_id="acme-spike",
                     stdin=stdin, stdout=stdout)
        resps = [json.loads(ln) for ln in stdout.getvalue().splitlines() if ln.strip()]
        self.assertEqual([r["id"] for r in resps], ["0", "1", "2"])

    def test_handle_is_directly_usable(self):
        # handle() is the single-envelope primitive main_cli loops over.
        resp = spi.handle({"spi_version": 0, "verb": "detect", "id": "x",
                           "payload": {"project_dir": "/z"}},
                          {"detect": lambda p: {"echo": p["project_dir"]}})
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["result"], {"echo": "/z"})


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
class LibclangSmokeTest(unittest.TestCase):
    """Guarded smoke: parse a tiny TU and exercise walk/toks/find_loops end to
    end against a real libclang cursor tree."""

    def test_walk_toks_loops_on_real_tu(self):
        import tempfile
        import clang.cindex as ci
        from pulp_importer_substrate import (
            _configure_libclang, walk, toks, find_loops, find_method, in_main_file,
        )
        _configure_libclang()
        src = (
            "struct S {\n"
            "  void go() {\n"
            "    int total = 0;\n"
            "    for (int i = 0; i < 4; ++i) { total += i; }\n"
            "  }\n"
            "};\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
            f.write(src)
            path = pathlib.Path(f.name)
        tu = ci.Index.create().parse(str(path), args=["-std=c++20"])
        # walk yields the whole tree; find a known token from a cursor.
        all_cursors = list(walk(tu.cursor))
        self.assertTrue(len(all_cursors) > 5)
        # find the method definition and confirm it's in the main file.
        m = find_method(tu, path, "go")
        self.assertIsNotNone(m)
        self.assertTrue(in_main_file(m, path))
        self.assertIn("go", toks(m))
        # the for-loop is detected.
        self.assertEqual(len(find_loops(tu, path)), 1)


if __name__ == "__main__":
    unittest.main()
