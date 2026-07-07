#!/usr/bin/env python3
"""Tests for shared optional-integration detectors."""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pulp_importer_substrate import detect_tuning_integration_requirements  # noqa: E402


class TuningIntegrationDetectionTest(unittest.TestCase):
    def test_detects_mts_and_local_tuning_assets(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "Source").mkdir()
            (root / "Source" / "Plugin.cpp").write_text(
                '#include "libMTSClient.h"\n'
                "void f() { MTS_NoteToFrequency(nullptr, 60, 0); }\n",
                encoding="utf-8",
            )
            (root / "Scales").mkdir()
            (root / "Scales" / "factory.scl").write_text("! demo\n12\n", encoding="utf-8")
            (root / "Scales" / "keyboard.kbm").write_text("! demo\n12\n", encoding="utf-8")

            reqs, diagnostics, tasks = detect_tuning_integration_requirements(root)

            ids = {p["id"] for p in reqs["packages"]}
            self.assertEqual(ids, {"mts-esp", "sst-tuning-library"})
            opts = {o["name"] for o in reqs["cmake_options"]}
            self.assertEqual(opts, {"PULP_ENABLE_MTS_ESP", "PULP_ENABLE_SCALA_TUNING"})
            assets = {a["path"]: a for a in reqs["asset_inputs"]}
            self.assertEqual(assets["Scales/factory.scl"]["copy_policy"], "copy_to_scaffold")
            self.assertEqual(assets["Scales/keyboard.kbm"]["kind"], "keyboard_mapping")
            self.assertTrue(any(d["code"] == "tuning.mts_esp" for d in diagnostics))
            self.assertTrue(any("MtsEspFallbackTuningProvider" in t for t in tasks))

    def test_tun_files_are_preserved_for_manual_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "tunings").mkdir()
            (root / "tunings" / "legacy.tun").write_text("[Tuning]\n", encoding="utf-8")

            reqs, diagnostics, tasks = detect_tuning_integration_requirements(root)

            self.assertNotIn("packages", reqs)
            self.assertEqual(reqs["asset_inputs"][0]["copy_policy"], "copy_to_scaffold")
            self.assertTrue(reqs["asset_inputs"][0]["requires_manual_review"])
            self.assertEqual(reqs["asset_inputs"][0]["path"], "tunings/legacy.tun")
            self.assertTrue(any(d["code"] == "tuning.tun_manual_review" for d in diagnostics))
            self.assertTrue(any("`.tun`" in t for t in tasks))

    def test_ignores_generated_and_dependency_directories(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            for rel, text in {
                "build/Generated.cpp": "void f() { MTS_NoteToFrequency(nullptr, 60, 0); }\n",
                "cmake-build-debug/CMakeFiles/Probe.cpp": "void f() { Tunings::Tuning t; }\n",
                "third_party/mts/Client.cpp": "void f() { MTS_RegisterClient(); }\n",
                "vendor/tuning-library/Demo.cpp": "void f() { Tunings::readSCLFile(\"demo.scl\"); }\n",
                "Builds/MacOSX/Old.cpp": "#include \"libMTSClient.h\"\n",
            }.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            for rel in ("build/cache.scl", "third_party/unused.kbm"):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("unused\n", encoding="utf-8")

            reqs, diagnostics, tasks = detect_tuning_integration_requirements(root)

            self.assertEqual(reqs, {})
            self.assertEqual(diagnostics, [])
            self.assertEqual(tasks, [])

    def test_project_sources_win_over_ignored_noise(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "Source").mkdir()
            (root / "Source" / "Synth.cpp").write_text(
                "void f() { Tunings::parseSCLData(\"! demo\\n12\\n\"); }\n",
                encoding="utf-8",
            )
            (root / "build").mkdir()
            (root / "build" / "Generated.cpp").write_text(
                "void f() { MTS_NoteToFrequency(nullptr, 60, 0); }\n",
                encoding="utf-8",
            )

            reqs, diagnostics, tasks = detect_tuning_integration_requirements(root)

            ids = {p["id"] for p in reqs["packages"]}
            self.assertEqual(ids, {"sst-tuning-library"})
            refs = reqs["packages"][0]["source_refs"]
            self.assertEqual(refs[0]["file"], "Source/Synth.cpp")
            self.assertTrue(any(d["code"] == "tuning.local_files" for d in diagnostics))
            self.assertTrue(any("ScalaTuningProvider" in t for t in tasks))


if __name__ == "__main__":
    unittest.main()
