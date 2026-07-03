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
    DEFAULT_CONFIDENCE_FLOOR,
    FileSpec,
    _gen_param_bind_grid,
    _param_discrete_divisor,
    emit,
    make_framework_free_predicate,
    produce,
)


def _choice_param(choices, **over):
    """A resolvable choice/enum parameter (JUCE AudioParameterChoice shape)."""
    p = {
        "id": "osc_wave", "source_id_string": "osc_wave",
        "proposed_pulp_id": 777, "name": "Osc Wave",
        "pulp_range": {"min": 0, "max": max(len(choices) - 1, 0), "step": 1},
        "default": 0, "choices": list(choices), "confidence": 0.9,
    }
    p.update(over)
    return p


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


class ConfidenceGatingTest(unittest.TestCase):
    """Honesty gate (plan §§7.4/16.3): a low-confidence param must downgrade to a
    labelled TODO stub, never emit its guessed value as if certain. A
    high-confidence param emits the concrete add_parameter block."""

    def _cpp(self, ir, **kw):
        return next(f.content for f in produce(ir, **kw)["files"]
                    if f.path == "src/PluginProcessor.cpp")

    def test_high_confidence_emits_concrete_value(self):
        ir = _ir()  # gain confidence 0.9
        cpp = self._cpp(ir)
        self.assertIn("store.add_parameter({", cpp)
        self.assertIn(".range = {-60.0f, 12.0f, 0.0f, 0.1f, 0.25f, false}", cpp)
        self.assertNotIn("low-confidence", cpp)

    def test_low_confidence_downgrades_to_todo_stub(self):
        ir = _ir()
        ir["parameters"][0]["confidence"] = 0.2
        cpp = self._cpp(ir)
        # Labelled TODO naming the low confidence; no LIVE add_parameter call.
        self.assertIn("TODO(import): low-confidence (Gain, confidence 0.2) — "
                      "verify before trusting.", cpp)
        self.assertIn("GUESSED — verify", cpp)
        # The concrete registration is commented out, never emitted live.
        self.assertNotIn("\n        store.add_parameter({", cpp)
        self.assertIn("// store.add_parameter({", cpp)

    def test_floor_is_configurable_and_boundary_is_strict_less_than(self):
        ir = _ir()
        ir["parameters"][0]["confidence"] = 0.5  # exactly at the default floor
        # At the floor: NOT below it -> concrete value (strict <).
        self.assertIn("store.add_parameter({", self._cpp(ir))
        # Raise the floor above 0.5 -> now gated.
        cpp = self._cpp(ir, confidence_floor=0.6)
        self.assertIn("TODO(import): low-confidence", cpp)

    def test_default_floor_constant_value(self):
        self.assertEqual(DEFAULT_CONFIDENCE_FLOOR, 0.5)


class StateSkeletonTest(unittest.TestCase):
    """The emitted serialize/deserialize_plugin_state hooks are a working
    param-state save/restore skeleton (APVTS / IParam -> Pulp state bridge)."""

    def _cpp(self, ir):
        return next(f.content for f in produce(ir)["files"]
                    if f.path == "src/PluginProcessor.cpp")

    def test_default_state_round_trips_params(self):
        cpp = self._cpp(_ir())
        # serialize snapshots the StateStore param payload...
        self.assertIn("std::vector<uint8_t> blob = state().serialize();", cpp)
        self.assertIn("return blob;", cpp)
        # ...and deserialize restores it (with the empty-payload legacy path).
        self.assertIn("if (data.empty())", cpp)
        self.assertIn("if (!state().deserialize(data))", cpp)
        # no stale "return {};" stub.
        self.assertNotIn("    return {};", cpp)

    def test_opaque_state_keeps_session_compat_todo(self):
        ir = _ir()
        ir["state_model"]["classification"] = "opaque-custom"
        cpp = self._cpp(ir)
        # Param state still round-trips...
        self.assertIn("state().serialize();", cpp)
        # ...but binary session compat is an explicit TODO, not a silent claim.
        self.assertIn("binary", cpp.lower())
        self.assertIn("DAW-session compatibility with the original plugin is NOT "
                      "supported", cpp)


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


class WebViewUITest(unittest.TestCase):
    """When the IR's vendor-neutral `ui.kind` is `"webview"`, EMIT produces a
    Pulp webview-ui scaffold (WebViewPanel-hosting create_view + native
    param-by-string-key bridge shim + placeholder ui/ assets), NOT the native
    Canvas2D/custom-paint path. Non-webview UIs keep the existing path. The core
    branches on the `ui.kind` DATA alone and names no framework."""

    def _webview_ir(self, *, html_entry=None, relays=None):
        ir = _ir()
        ir["ui"] = {
            "kind": "webview", "model": "webview", "confidence": 0.8,
            "asset_hints": {
                "html_entry": html_entry,
                "param_relays": relays or [],
                "bridge_calls": ["withNativeIntegrationEnabled"],
            },
        }
        return ir

    def _files(self, ir):
        return {f.path: f for f in produce(ir)["files"]}

    def test_native_path_unchanged_without_webview_kind(self):
        # The default IR has no ui block -> native path: no create_view, no ui/.
        files = self._files(_ir())
        self.assertNotIn("ui/index.html", files)
        hpp = files["src/PluginProcessor.hpp"].content
        cpp = files["src/PluginProcessor.cpp"].content
        self.assertNotIn("create_view", hpp)
        self.assertNotIn("WebViewPanel", cpp)

    def test_explicit_native_kind_stays_native(self):
        ir = _ir()
        ir["ui"] = {"kind": "native", "model": "custom-paint"}
        files = self._files(ir)
        self.assertNotIn("ui/index.html", files)
        self.assertNotIn("create_view", files["src/PluginProcessor.hpp"].content)

    def test_webview_emits_scaffold_not_canvas2d(self):
        files = self._files(self._webview_ir())
        # Placeholder web asset shipped (generated, not a copied user file).
        self.assertIn("ui/index.html", files)
        self.assertEqual(files["ui/index.html"].provenance, "generated")
        self.assertEqual(files["ui/index.html"].classification, "asset")
        hpp = files["src/PluginProcessor.hpp"].content
        cpp = files["src/PluginProcessor.cpp"].content
        # Header declares the view lifecycle.
        self.assertIn("std::unique_ptr<pulp::view::View> create_view() override;", hpp)
        self.assertIn("pulp::format::ViewSize view_size() const override;", hpp)
        self.assertIn("#include <pulp/view/web_view.hpp>", hpp)
        # Source hosts a WebViewPanel via the directory resource fetcher.
        self.assertIn("pulp::view::WebViewPanel::create(options)", cpp)
        self.assertIn("make_webview_directory_resource_fetcher", cpp)
        self.assertIn("class WebViewEditorRoot", cpp)
        self.assertIn("attach_native_child_view", cpp)
        # The asset-copy TODO is present and honest.
        self.assertIn("TODO(import): copy your WebView assets (HTML/JS/CSS) into ui/",
                      cpp)
        self.assertIn("the importer cannot extract bundled binary resources", cpp)

    def test_webview_native_param_bridge_shim_maps_by_key(self):
        cpp = self._files(self._webview_ir())["src/PluginProcessor.cpp"].content
        # The bridge maps the source param id string ("gain") to the stable enum.
        self.assertIn("set_message_handler", cpp)
        self.assertIn('if (key == "gain") { out = kParam_gain; return true; }', cpp)
        self.assertIn('if (msg.type == "param")', cpp)
        # A TODO for bespoke (non-param) message handlers.
        self.assertIn("bespoke message", cpp)

    def test_webview_uses_literal_html_entry_when_present(self):
        cpp = self._files(self._webview_ir(html_entry="editor.html"))[
            "src/PluginProcessor.cpp"].content
        self.assertIn('navigate("pulp://app/editor.html")', cpp)
        self.assertIn("INSPECT found a literal HTML entry reference", cpp)

    def test_webview_defaults_entry_with_todo_when_unresolved(self):
        cpp = self._files(self._webview_ir(html_entry=None))[
            "src/PluginProcessor.cpp"].content
        self.assertIn('navigate("pulp://app/index.html")', cpp)
        self.assertIn("could not statically resolve the HTML entry", cpp)

    def test_webview_strips_directory_from_entry_hint(self):
        # An entry hint with a path -> only the filename is used (ui/ is root).
        cpp = self._files(self._webview_ir(html_entry="web/public/app.html"))[
            "src/PluginProcessor.cpp"].content
        self.assertIn('navigate("pulp://app/app.html")', cpp)

    def test_webview_migration_status_records_ui(self):
        status = produce(self._webview_ir())["migration_status"]
        self.assertEqual(status["ui_kind"], "webview")
        self.assertEqual(status["verdict"]["ui_parity"], "partial")
        self.assertTrue(any("WebView UI" in t for t in status["todos"]))

    def test_webview_index_html_exercises_bridge(self):
        html = self._files(self._webview_ir())["ui/index.html"].content
        self.assertIn("window.pulp", html)
        self.assertIn("placeholder", html.lower())
        self.assertIn("<!doctype html>", html)


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


class BindGridTest(unittest.TestCase):
    """The bind-grid codegen: the discrete normalization divisor (normDen) is
    derived once from the parameter's own definition so a UI binding cannot
    disagree with it — the 3-option-control-wired-to-a-6-step-divisor class of
    hand-transcription bug is impossible when the divisor is generated."""

    def test_divisor_is_option_count_minus_one(self):
        # A 3-option control has divisor 2 (positions 0,1,2 -> 0, .5, 1), NOT 6.
        self.assertEqual(_param_discrete_divisor(_choice_param(["A", "B", "C"])), 2)
        self.assertEqual(
            _param_discrete_divisor(_choice_param(["Saw", "Sq", "Tri", "Sin"])), 3)

    def test_single_or_empty_choice_has_no_divisor(self):
        self.assertIsNone(_param_discrete_divisor(_choice_param(["Only"])))
        self.assertIsNone(_param_discrete_divisor(_choice_param([])))

    def test_integer_control_divisor_is_the_span(self):
        intp = {"source_id_string": "voices", "proposed_pulp_id": 9,
                "pulp_range": {"min": 1, "max": 8, "step": 1}}
        self.assertEqual(_param_discrete_divisor(intp), 7)  # 1..8 -> 7 intervals

    def test_continuous_param_has_no_divisor(self):
        # The _ir() gain is {-60, 12, step 0.1} — a fine cosmetic step must NOT be
        # mistaken for a 720-position discrete control.
        self.assertIsNone(_param_discrete_divisor(_ir()["parameters"][0]))

    def test_bind_grid_entry_carries_divisor_and_key(self):
        ir = _ir()
        ir["parameters"] = [_choice_param(["A", "B", "C"]), ir["parameters"][0]]
        grid = _gen_param_bind_grid(ir)
        self.assertEqual(len(grid), 2)
        choice, gain = grid[0], grid[1]
        self.assertEqual(choice["key"], "osc_wave")
        self.assertEqual(choice["discrete_divisor"], 2)
        self.assertEqual(choice["choices"], ["A", "B", "C"])
        self.assertIsNone(gain["discrete_divisor"])  # continuous
        self.assertNotIn("choices", gain)

    def test_unresolved_param_is_skipped(self):
        ir = _ir()
        ir["parameters"] = [{"source_id_string": None, "proposed_pulp_id": None}]
        self.assertEqual(_gen_param_bind_grid(ir), [])

    def test_manifest_exposes_bind_grid(self):
        ir = _ir()
        ir["parameters"] = [_choice_param(["A", "B", "C"])]
        status = produce(ir)["migration_status"]
        self.assertIn("bind_grid", status)
        self.assertEqual(status["bind_grid"][0]["discrete_divisor"], 2)

    def test_emitted_registration_documents_the_divisor(self):
        ir = _ir()
        ir["parameters"] = [_choice_param(["A", "B", "C"])]
        cpp = next(f.content for f in produce(ir)["files"]
                   if f.path == "src/PluginProcessor.cpp")
        self.assertIn("normalized position = index / 2", cpp)
        self.assertNotIn("index / 6", cpp)  # the transcription bug never appears


if __name__ == "__main__":
    unittest.main()
