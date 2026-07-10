#!/usr/bin/env python3
"""Unit tests for the Pulp-convention `source_curve` contract in the EMIT core.

The emitter is DUMB: `source_curve` already carries Pulp-convention values (the
vendor extractor converted vendor->Pulp), so these tests drive the emitter with
synthetic Pulp-convention IRs — no libclang, no framework — and lock:

  - a `pow` curve with a Pulp-convention skew emits a 6-field shaped ParamRange
    carrying that exact skew, and (fidelity="exact") claims the float-tolerance
    round-trip;
  - an `exp` curve emits `ParamRange::with_centre(...)` + a CLOSE/PARTIAL
    diagnostic, and NEVER a linear 4-field range (the S2 silent-linear bug);
  - fidelity=="close" suppresses the "round-trips within float tolerance" claim;
  - a call already claimed by a `loop_or_factory` construct does NOT also emit a
    high-confidence `parameters[]` entry (the S3 double-report).

Run: python3 tests/test_curve_contract.py
"""
from __future__ import annotations

import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pulp_importer_substrate.emit import produce  # noqa: E402


def _ir(param, **over) -> dict:
    """A minimal vendor-neutral ProjectIR carrying one parameter."""
    base = {
        "schema": "pulp.import.project_ir.draft1",
        "source": {"framework": "demo", "project_dir": "Demo"},
        "metadata": {
            "name": "Demo Gain", "manufacturer": "Acme", "version": "1.0.0",
            "bundle_id": "com.acme.demo", "pulp_category": "Effect",
            "formats": ["CLAP"],
        },
        "midi": {"accepts_midi": False, "produces_midi": False},
        "buses": {"inputs": [{"name": "Input", "channels": 2}],
                  "outputs": [{"name": "Output", "channels": 2}]},
        "parameters": [param],
        "constructs": [],
        "state_model": {"classification": "default-params"},
        "dsp": {"classification": "stub", "reachability_scope": "n/a",
                "diagnostics": []},
        "confidence_overall": 0.7,
    }
    base.update(over)
    return base


def _cpp(ir) -> str:
    return next(f.content for f in produce(ir)["files"]
                if f.path == "src/PluginProcessor.cpp")


class PowCurveTest(unittest.TestCase):
    def test_pow_curve_emits_six_field_shaped_range_with_exact_skew(self):
        # Pulp-convention skew (already reciprocated from any vendor exponent).
        p = {
            "id": "gain", "source_id_string": "gain", "proposed_pulp_id": 12345,
            "name": "Gain", "pulp_range": {"min": -60.0, "max": 12.0, "step": 0.1},
            "default": 0.0, "confidence": 0.9,
            "source_curve": {"shape": "pow", "skew": 0.3333333,
                             "symmetric": False, "fidelity": "exact",
                             "centre": None},
        }
        cpp = _cpp(_ir(p))
        # 6-field {min,max,default,step,skew,symmetric} carrying the EXACT skew.
        self.assertIn(".range = {-60.0f, 12.0f, 0.0f, 0.1f, 0.3333333f, false}", cpp)
        # exact fidelity => it may claim the round-trip.
        self.assertIn("round-trips within float tolerance", cpp)
        self.assertIn("source curve skew=0.3333333", cpp)
        # Never falls back to a 4-field linear range for a shaped curve.
        self.assertNotIn(".range = {-60.0f, 12.0f, 0.0f, 0.1f},", cpp)


class ExpCurveTest(unittest.TestCase):
    def _exp_param(self, **over):
        p = {
            "id": "cutoff", "source_id_string": "cutoff",
            "proposed_pulp_id": 54321, "name": "Cutoff",
            "pulp_range": {"min": 20.0, "max": 20000.0, "step": 0.0},
            "default": 1000.0, "confidence": 0.9,
            # exp has NO pow-skew value (skew=None); the extractor supplies a
            # geometric-mean centre for with_centre() and marks it CLOSE.
            "source_curve": {"shape": "exp", "skew": None, "symmetric": False,
                             "fidelity": "close", "centre": 632.4555},
        }
        p.update(over)
        return p

    def test_exp_emits_with_centre_partial_and_never_linear(self):
        cpp = _cpp(_ir(self._exp_param()))
        # with_centre factory, NOT an aggregate brace-init range.
        self.assertIn(
            "pulp::state::ParamRange::with_centre(20.0f, 20000.0f, 632.4555f, "
            "1000.0f, 0.0f)", cpp)
        # NEVER a 4-field (or 6-field) linear/aggregate range for this param.
        self.assertNotIn(".range = {20.0f, 20000.0f", cpp)
        # A CLOSE/PARTIAL diagnostic naming the exp provenance is present.
        self.assertIn("shape=exp", cpp)
        self.assertIn("CLOSE/PARTIAL", cpp)
        # And it does NOT claim a faithful round-trip.
        self.assertNotIn("round-trips within float tolerance", cpp)

    def test_exp_records_partial_migration_todo(self):
        status = produce(_ir(self._exp_param()))["migration_status"]
        self.assertTrue(any("CLOSE/PARTIAL" in t and "Cutoff" in t
                            for t in status["todos"]),
                        f"expected a CLOSE/PARTIAL curve TODO, got {status['todos']}")

    def test_exp_without_centre_still_never_emits_linear(self):
        # Defensive: even if the extractor omitted centre, a positive range gets
        # a geometric-mean fallback and is STILL emitted via with_centre.
        p = self._exp_param()
        p["source_curve"]["centre"] = None
        cpp = _cpp(_ir(p))
        self.assertIn("pulp::state::ParamRange::with_centre(20.0f, 20000.0f", cpp)
        self.assertNotIn(".range = {20.0f, 20000.0f", cpp)


class FidelityCommentTest(unittest.TestCase):
    def test_close_fidelity_suppresses_roundtrip_comment(self):
        # A pow curve the extractor could only match CLOSELY (fidelity="close")
        # must NOT claim the float-tolerance round-trip, but is STILL shaped.
        p = {
            "id": "res", "source_id_string": "res", "proposed_pulp_id": 999,
            "name": "Res", "pulp_range": {"min": 0.0, "max": 1.0, "step": 0.0},
            "default": 0.5, "confidence": 0.9,
            "source_curve": {"shape": "pow", "skew": 0.5, "symmetric": False,
                             "fidelity": "close", "centre": None},
        }
        cpp = _cpp(_ir(p))
        self.assertNotIn("round-trips within float tolerance", cpp)
        self.assertIn("CLOSE/PARTIAL: approximate", cpp)
        # Still a shaped 6-field range (never downgraded to linear).
        self.assertIn(".range = {0.0f, 1.0f, 0.5f, 0.0f, 0.5f, false}", cpp)


class ConstructClaimTest(unittest.TestCase):
    def test_construct_claimed_call_yields_no_phantom_param(self):
        # A loop-registered call emits BOTH a phantom parameter and a construct
        # sharing the same source_ref. The phantom must be suppressed so it is
        # not double-reported as a high-confidence parameter (the S3 bug).
        phantom = {
            "id": "kBand0", "source_id_string": "kBand0",
            "proposed_pulp_id": 111, "name": "Band 0",
            "pulp_range": {"min": 0.0, "max": 1.0, "step": 0.0},
            "default": 0.0, "confidence": 0.92,
            "source_curve": {"shape": "linear", "skew": 1.0},
            "source_ref": {"file": "Plugin.cpp", "line": 42},
        }
        real = {
            "id": "gain", "source_id_string": "gain", "proposed_pulp_id": 222,
            "name": "Gain", "pulp_range": {"min": -60.0, "max": 12.0, "step": 0.1},
            "default": 0.0, "confidence": 0.9,
            "source_curve": {"shape": "linear", "skew": 1.0},
            "source_ref": {"file": "Plugin.cpp", "line": 50},
        }
        ir = _ir(real)
        ir["parameters"] = [phantom, real]
        ir["constructs"] = [{
            "kind": "parameter_construct", "construct_type": "loop_or_factory",
            "source_ref": {"file": "Plugin.cpp", "line": 42},
            "enumeration_status": "not_enumerated",
        }]
        prod = produce(ir)
        cpp = next(f.content for f in prod["files"]
                   if f.path == "src/PluginProcessor.cpp")
        hpp = next(f.content for f in prod["files"]
                   if f.path == "src/PluginProcessor.hpp")
        # The phantom must not appear as a live parameter anywhere.
        self.assertNotIn("kBand0", hpp)
        self.assertNotIn("kBand0", cpp)
        self.assertNotIn('"Band 0"', cpp)
        # The real param survives.
        self.assertIn("kParam_gain", hpp)
        # And the construct is still reported (as an unresolved TODO).
        self.assertTrue(any("loop_or_factory" in t
                            for t in prod["migration_status"]["todos"]))
        # The phantom is gone from the bind-grid too (single source of truth).
        keys = {e["key"] for e in prod["migration_status"]["bind_grid"]}
        self.assertNotIn("kBand0", keys)
        self.assertIn("gain", keys)

    def test_computed_args_construct_does_not_suppress_its_param(self):
        # A single computed-args call IS a real (low-confidence) parameter, not a
        # phantom — only loop_or_factory claims/suppresses. Same source_ref, but
        # construct_type "computed_args" must NOT drop the param.
        p = {
            "id": "gain", "source_id_string": "gain", "proposed_pulp_id": 222,
            "name": "Gain", "pulp_range": {"min": 0.0, "max": 1.0, "step": 0.0},
            "default": 0.0, "confidence": 0.2,
            "source_curve": {"shape": "linear", "skew": 1.0},
            "source_ref": {"file": "Plugin.cpp", "line": 50},
        }
        ir = _ir(p)
        ir["constructs"] = [{
            "construct_type": "computed_args",
            "source_ref": {"file": "Plugin.cpp", "line": 50},
        }]
        hpp = next(f.content for f in produce(ir)["files"]
                   if f.path == "src/PluginProcessor.hpp")
        self.assertIn("kParam_gain", hpp)  # not suppressed


if __name__ == "__main__":
    unittest.main()
