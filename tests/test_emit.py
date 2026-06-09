#!/usr/bin/env python3
"""Unit tests for the vendor-agnostic EMIT core (`pulp_importer_substrate.emit`).

The emit core consumes a vendor-neutral ProjectIR and proposes a Pulp migration
scaffold (FileSpec list + migration_status + report). It names no vendor; the
framework-specific touch-points (report renderer, framework-free predicate,
boundary-file markers, cosmetic labels, emit-tool id) are injected as DATA. These
tests drive the core with a synthetic IR — no libclang, no framework — and lock:

  - the scaffold file set + per-file provenance,
  - shaped-vs-linear ParamRange emission,
  - the injected report renderer + labels appear in the output,
  - the framework-free predicate / boundary markers gate portable-core copies,
  - the default (no-injection) path stays vendor-neutral.

Run: python3 tests/test_emit.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pulp_importer_substrate.emit import (  # noqa: E402
    FileSpec,
    emit,
    make_framework_free_predicate,
    produce,
)


def _ir(**over) -> dict:
    """A minimal but realistic vendor-neutral ProjectIR for an effect."""
    base = {
        "schema": "pulp.import.project_ir.draft1",
        "source": {"framework": "demo", "project_dir": "Demo"},
        "metadata": {
            "name": "Demo Gain", "manufacturer": "Acme", "version": "1.0.0",
            "bundle_id": "com.acme.demo", "pulp_category": "Effect",
            "formats": ["VST3", "CLAP"],
        },
        "midi": {"accepts_midi": False, "produces_midi": False},
        "buses": {"inputs": [{"name": "Input", "channels": 2}],
                  "outputs": [{"name": "Output", "channels": 2}]},
        "parameters": [{
            "id": "gain", "source_id_string": "gain", "proposed_pulp_id": 12345,
            "name": "Gain", "pulp_range": {"min": -60.0, "max": 12.0, "step": 0.1},
            "default": 0.0, "source_curve": {"skew": 0.25, "symmetric": False},
            "confidence": 0.9,
        }],
        "constructs": [],
        "state_model": {"classification": "default-params"},
        "dsp": {"classification": "stub", "reachability_scope": "n/a",
                "diagnostics": []},
        "confidence_overall": 0.7,
    }
    base.update(over)
    return base


class ProduceShapeTest(unittest.TestCase):
    def test_file_set_and_provenance(self):
        prod = produce(_ir(), sdk_version="0.1.0")
        paths = {f.path: f for f in prod["files"]}
        for p in ("src/PluginProcessor.hpp", "src/PluginProcessor.cpp",
                  "src/clap_entry.cpp", "CMakeLists.txt",
                  "migration_status.json", "IMPORT_REPORT.md"):
            self.assertIn(p, paths)
            self.assertEqual(paths[p].provenance, "generated")
        # CLAP-only formats; the source VST3 is deferred, not emitted.
        self.assertEqual(prod["formats"], ["CLAP"])
        self.assertIn("VST3", prod["deferred_formats"])
        self.assertEqual(prod["migration_status"]["status"], "unresolved")

    def test_shaped_range_emitted_as_six_field_aggregate(self):
        cpp = next(f.content for f in produce(_ir())["files"]
                   if f.path == "src/PluginProcessor.cpp")
        # skew=0.25 -> 6-field {min,max,default,step,skew,symmetric} aggregate.
        self.assertIn(".range = {-60.0f, 12.0f, 0.0f, 0.1f, 0.25f, false}", cpp)
        self.assertIn("source curve skew=0.25 emitted as a shaped ParamRange", cpp)

    def test_linear_param_emitted_as_four_field_aggregate(self):
        ir = _ir()
        ir["parameters"][0]["source_curve"] = {"skew": 1.0, "symmetric": False}
        cpp = next(f.content for f in produce(ir)["files"]
                   if f.path == "src/PluginProcessor.cpp")
        self.assertIn(".range = {-60.0f, 12.0f, 0.0f, 0.1f}", cpp)
        self.assertNotIn("0.1f, 1.0f, false}", cpp)

    def test_stable_param_id_in_header(self):
        hpp = next(f.content for f in produce(_ir())["files"]
                   if f.path == "src/PluginProcessor.hpp")
        self.assertIn("kParam_gain = 12345u", hpp)

    def test_instrument_emits_labelled_silence(self):
        ir = _ir()
        ir["metadata"]["pulp_category"] = "Instrument"
        cpp = next(f.content for f in produce(ir)["files"]
                   if f.path == "src/PluginProcessor.cpp")
        self.assertIn("Labelled silence", cpp)
        self.assertIn("out[i] = 0.0f;", cpp)


class InjectionTest(unittest.TestCase):
    def test_injected_labels_and_emit_tool(self):
        prod = produce(_ir(), id_label="Acme", tool_label="Acme ",
                       emit_tool="acme-emit/9")
        cpp = next(f.content for f in prod["files"]
                   if f.path == "src/PluginProcessor.cpp")
        hpp = next(f.content for f in prod["files"]
                   if f.path == "src/PluginProcessor.hpp")
        cmake = next(f.content for f in prod["files"]
                     if f.path == "CMakeLists.txt")
        self.assertIn('// source Acme id "gain"', cpp)
        self.assertIn("Generated by the Acme importer EMIT step", hpp)
        self.assertIn("generated by the Acme importer EMIT step", cmake)
        self.assertEqual(prod["migration_status"]["emit_tool"], "acme-emit/9")

    def test_default_path_names_no_vendor_label(self):
        # The no-injection default must stay vendor-neutral.
        prod = produce(_ir())
        cpp = next(f.content for f in prod["files"]
                   if f.path == "src/PluginProcessor.cpp")
        self.assertIn('// source param id "gain"', cpp)

    def test_injected_report_renderer_is_used(self):
        def render(ir):
            return "RENDERED-BY-IMPORTER " + ir["metadata"]["name"] + "\n"
        report = next(f.content for f in produce(_ir(), render_report=render)["files"]
                      if f.path == "IMPORT_REPORT.md")
        # The core appends an EMIT verdict block on top of the importer's prose.
        self.assertIn("**EMIT verdict:**", report)
        self.assertIn("RENDERED-BY-IMPORTER Demo Gain", report)


class PortableCoreTest(unittest.TestCase):
    def _portable_ir(self):
        ir = _ir()
        ir["dsp"]["classification"] = "portable-core"
        return ir

    def test_framework_free_header_copied_metadata_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            src = pathlib.Path(td)
            # Framework-free DSP core (only mentions the framework in a comment).
            (src / "Core.h").write_text(
                "// no demoframework:: anywhere\n#pragma once\n"
                "namespace c { struct S { void go(){} }; }\n")
            # A framework-bound header (real usage) — must be rejected.
            (src / "Bound.h").write_text(
                '#include "DemoFramework/api.h"\nusing namespace df;\n')
            # A metadata/boundary header — rejected by name marker.
            (src / "config.h").write_text("#define PLUG_NAME \"X\"\n")

            pred = make_framework_free_predicate(["demoframework", "DemoFramework", "df"])
            prod = produce(self._portable_ir(), source_dir=src,
                           framework_free_predicate=pred,
                           boundary_name_markers=["config.h"])
            copied = {f.path: f for f in prod["files"]
                      if f.provenance == "copied-user-file"}
            self.assertIn("src/Core.h", copied)
            self.assertNotIn("src/Bound.h", copied)
            self.assertNotIn("src/config.h", copied)
            # A copied file ships copy_from, never inline content.
            self.assertIsNotNone(copied["src/Core.h"].copy_from)
            self.assertIsNone(copied["src/Core.h"].content)

    def test_emit_writes_scaffold_to_disk(self):
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "scaffold"
            info = emit(_ir(), out, sdk_version="0.1.0")
            self.assertTrue((out / "src" / "PluginProcessor.cpp").exists())
            status = json.loads((out / "migration_status.json").read_text())
            self.assertEqual(status["status"], "unresolved")
            self.assertEqual(info["formats"], ["CLAP"])


class FileSpecTest(unittest.TestCase):
    def test_generated_entry_inlines_content(self):
        e = FileSpec("a.cpp", content="x", provenance="generated",
                     classification="source").as_manifest_entry()
        self.assertEqual(e, {"path": "a.cpp", "provenance": "generated",
                             "classification": "source", "content": "x"})

    def test_copied_entry_ships_copy_from(self):
        e = FileSpec("a.h", provenance="copied-user-file",
                     copy_from="/abs/a.h").as_manifest_entry()
        self.assertEqual(e["copy_from"], "/abs/a.h")
        self.assertNotIn("content", e)


if __name__ == "__main__":
    unittest.main()
