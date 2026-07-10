#!/usr/bin/env python3
"""Unit tests for the shared extraction-health contract (the false-clean guard).

`extraction_health` centralises "is this extraction trustworthy?" so both
importers derive exit status the same way. These tests are dependency-free (no
libclang) and unittest style to match `tests/test_substrate.py`. They lock the
framework-aware discriminator that is the whole point:

  - a `file not found` on a FRAMEWORK header => not ok;
  - an unrelated `severity: error` (JUCE's rvalue-reference-binding one, which a
    CORRECT run also emits) => STILL ok;
  - an `error`-key IR => not ok;
  - an explicit framework path with zero params and no construct => not ok;
  - the `--allow-unresolved-includes` opt-out suppresses the header reason.

Run: python3 tests/test_extraction_health.py
"""
from __future__ import annotations

import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pulp_importer_substrate import (  # noqa: E402
    IPLUG2_SPEC,
    JUCE_SPEC,
    extraction_health,
    stamp_extraction_health,
)
from pulp_importer_substrate.emit import produce  # noqa: E402


def _ir(*, parse_errors=None, params=None, constructs=None, error=None) -> dict:
    ir: dict = {
        "schema": "pulp.import.project_ir.draft1",
        "source": {"framework": "iplug2"},
        "metadata": {"name": "Demo", "pulp_category": "Effect"},
        "parameters": params if params is not None else [],
        "constructs": constructs if constructs is not None else [],
        "extraction_context": {"parse_errors": parse_errors or []},
        "diagnostics": [],
    }
    if error is not None:
        ir["error"] = error
    return ir


def _one_param():
    return [{"id": "gain", "source_id_string": "gain", "proposed_pulp_id": 1,
             "name": "Gain", "pulp_range": {"min": 0.0, "max": 1.0, "step": 0.0},
             "default": 0.0, "confidence": 0.9,
             "source_curve": {"shape": "linear", "skew": 1.0}}]


class ExtractionHealthTest(unittest.TestCase):
    def test_framework_header_not_found_is_not_ok(self):
        # iPlug2 header unresolved -> the --framework-path is wrong/incomplete.
        ir = _ir(parse_errors=["'IPlugAPIBase.h' file not found"],
                 params=_one_param())
        ok, reasons = extraction_health(ir, framework_path_given=True,
                                        spec=IPLUG2_SPEC)
        self.assertFalse(ok)
        self.assertTrue(any("IPlugAPIBase.h" in r for r in reasons))

    def test_juce_framework_header_not_found_is_not_ok(self):
        ir = _ir(parse_errors=[
            "'juce_audio_processors/juce_audio_processors.h' file not found"],
            params=_one_param())
        ok, reasons = extraction_health(ir, framework_path_given=True,
                                        spec=JUCE_SPEC)
        self.assertFalse(ok)

    def test_unrelated_rvalue_parse_error_is_ok(self):
        # A CORRECT JUCE parse also emits an unrelated severity=error diagnostic;
        # it must NOT be read as a wrong include path.
        ir = _ir(parse_errors=[
            "non-const lvalue reference to type cannot bind to a temporary"],
            params=_one_param())
        ok, reasons = extraction_health(ir, framework_path_given=True,
                                        spec=JUCE_SPEC)
        self.assertTrue(ok, reasons)
        self.assertEqual(reasons, [])

    def test_error_key_ir_is_not_ok(self):
        ir = _ir(error="no source file found")
        ok, reasons = extraction_health(ir, framework_path_given=False,
                                        spec=IPLUG2_SPEC)
        self.assertFalse(ok)
        self.assertTrue(any("no source file found" in r for r in reasons))

    def test_explicit_path_zero_params_no_construct_is_not_ok(self):
        ir = _ir(params=[], constructs=[])
        ok, reasons = extraction_health(ir, framework_path_given=True,
                                        spec=IPLUG2_SPEC)
        self.assertFalse(ok)
        self.assertTrue(any("no parameters" in r for r in reasons))

    def test_explicit_path_zero_params_with_construct_is_ok(self):
        # A construct explains WHY params is empty (loop/factory) -> trustworthy.
        ir = _ir(params=[], constructs=[{"construct_type": "loop_or_factory"}])
        ok, reasons = extraction_health(ir, framework_path_given=True,
                                        spec=IPLUG2_SPEC)
        self.assertTrue(ok, reasons)

    def test_no_explicit_path_zero_params_is_ok(self):
        # Without an explicit framework path, zero params alone is not a failure
        # (e.g. a hermetic stub run) — only (a)/(b) can fail it.
        ir = _ir(params=[], constructs=[])
        ok, _ = extraction_health(ir, framework_path_given=False,
                                  spec=IPLUG2_SPEC)
        self.assertTrue(ok)

    def test_allow_unresolved_includes_suppresses_header_reason(self):
        ir = _ir(parse_errors=["'IPlugAPIBase.h' file not found"],
                 params=_one_param())
        ok, reasons = extraction_health(ir, framework_path_given=True,
                                        spec=IPLUG2_SPEC,
                                        allow_unresolved_includes=True)
        self.assertTrue(ok, reasons)

    def test_diagnostics_dict_message_is_also_scanned(self):
        # The header-not-found may arrive as a rolled-up diagnostics dict, not
        # only in extraction_context.parse_errors.
        ir = _ir(params=_one_param())
        ir["diagnostics"] = [{"severity": "error", "code": "parse",
                              "message": "'IGraphics.h' file not found"}]
        ok, _ = extraction_health(ir, framework_path_given=True, spec=IPLUG2_SPEC)
        self.assertFalse(ok)


class StampAndReportTest(unittest.TestCase):
    def test_stamp_writes_verdict_into_ir(self):
        ir = _ir(parse_errors=["'IPlugAPIBase.h' file not found"],
                 params=_one_param())
        ok, reasons = stamp_extraction_health(ir, framework_path_given=True,
                                              spec=IPLUG2_SPEC)
        self.assertFalse(ok)
        self.assertTrue(ir["extraction_failed"])
        self.assertEqual(ir["extraction_health"]["ok"], False)
        self.assertEqual(ir["extraction_health"]["reasons"], reasons)

    def test_stamp_clean_ir_has_no_failed_flag(self):
        ir = _ir(params=_one_param())
        ok, _ = stamp_extraction_health(ir, framework_path_given=True,
                                        spec=IPLUG2_SPEC)
        self.assertTrue(ok)
        self.assertNotIn("extraction_failed", ir)
        self.assertTrue(ir["extraction_health"]["ok"])

    def test_report_renders_extraction_failed_banner(self):
        # A stamped-failed IR renders a loud banner at the top of the report so
        # the scaffold does not read as a clean import.
        ir = _ir(parse_errors=["'IPlugAPIBase.h' file not found"],
                 params=_one_param())
        ir["metadata"] = {"name": "Demo", "pulp_category": "Effect",
                          "formats": ["CLAP"]}
        ir["state_model"] = {"classification": "default-params"}
        ir["dsp"] = {"classification": "stub", "reachability_scope": "n/a",
                     "diagnostics": []}
        stamp_extraction_health(ir, framework_path_given=True, spec=IPLUG2_SPEC)
        report = next(f.content for f in produce(ir)["files"]
                      if f.path == "IMPORT_REPORT.md")
        self.assertIn("EXTRACTION FAILED", report)
        self.assertIn("IPlugAPIBase.h", report)


if __name__ == "__main__":
    unittest.main()
