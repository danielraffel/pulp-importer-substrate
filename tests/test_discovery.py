#!/usr/bin/env python3
"""Unit tests for the shared framework checkout discovery module.

Every filesystem case builds its own fake checkout under a ``tempfile``
directory and, where auto-discovery is exercised, injects the well-known roots
via ``dataclasses.replace`` — so NO test depends on this machine's real
``~/Code`` layout. The tests are dependency-free (no libclang) and unittest
style to match ``tests/test_substrate.py``.

Run: python3 tests/test_discovery.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import unittest
from dataclasses import replace

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pulp_importer_substrate import (  # noqa: E402
    Candidate,
    FrameworkAmbiguous,
    FrameworkNotFound,
    IPLUG2_SPEC,
    JUCE_SPEC,
    include_resolution_failed,
    resolve_framework_root,
)


# --------------------------------------------------------------------------- #
# Fixtures — build fake checkouts on disk.
# --------------------------------------------------------------------------- #
_JUCE_CORE_H = """
/*
BEGIN_JUCE_MODULE_DECLARATION
  ID:                 juce_core
  vendor:             juce
  version:            8.0.13
  name:               JUCE core classes
END_JUCE_MODULE_DECLARATION
*/
#pragma once
"""


def _make_juce_modules(modules_dir: pathlib.Path, version: str = "8.0.13") -> pathlib.Path:
    """Create a valid JUCE *modules* root at ``modules_dir``."""
    core = modules_dir / "juce_core"
    core.mkdir(parents=True, exist_ok=True)
    (core / "juce_core.h").write_text(_JUCE_CORE_H.replace("8.0.13", version), encoding="utf-8")
    return modules_dir


def _make_juce_repo(repo_dir: pathlib.Path, version: str = "8.0.13") -> pathlib.Path:
    """Create a JUCE *repository root* (valid tree lives under ``modules/``)."""
    _make_juce_modules(repo_dir / "modules", version=version)
    return repo_dir


def _make_iplug(root_dir: pathlib.Path, version: str = "2.0.0") -> pathlib.Path:
    """Create a valid iPlug2 checkout root at ``root_dir``."""
    (root_dir / "IPlug").mkdir(parents=True, exist_ok=True)
    (root_dir / "IPlug" / "IPlugAPIBase.h").write_text("#pragma once\n", encoding="utf-8")
    (root_dir / "CMakeLists.txt").write_text(
        f"cmake_minimum_required(VERSION 3.14)\nproject(iPlug2 VERSION {version})\n",
        encoding="utf-8")
    return root_dir


class _TmpBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


# --------------------------------------------------------------------------- #
# Explicit CLI value.
# --------------------------------------------------------------------------- #
class ExplicitTest(_TmpBase):
    def test_explicit_valid_modules_path(self):
        mods = _make_juce_modules(self.tmp / "JUCE" / "modules")
        cand = resolve_framework_root(JUCE_SPEC, explicit=str(mods))
        self.assertIsInstance(cand, Candidate)
        self.assertEqual(cand.source, "--juce-path")
        self.assertEqual(cand.path, mods)
        self.assertIsNone(cand.note)
        self.assertEqual(cand.version, "8.0.13")

    def test_explicit_repo_root_is_normalized_to_modules_with_note(self):
        repo = _make_juce_repo(self.tmp / "JUCE")
        cand = resolve_framework_root(JUCE_SPEC, explicit=str(repo))
        self.assertEqual(cand.path, repo / "modules")
        self.assertIsNotNone(cand.note)
        self.assertIn("modules", cand.note)
        self.assertEqual(cand.source, "--juce-path")

    def test_explicit_bogus_path_is_fatal_names_flag_and_missing_file(self):
        bogus = self.tmp / "nope"
        bogus.mkdir()
        with self.assertRaises(FrameworkNotFound) as ctx:
            resolve_framework_root(JUCE_SPEC, explicit=str(bogus))
        msg = str(ctx.exception)
        self.assertIn("--juce-path", msg)
        self.assertIn("juce_core.h", msg)  # names what file was missing
        self.assertEqual(ctx.exception.candidates, [])

    def test_explicit_iplug_root(self):
        root = _make_iplug(self.tmp / "iPlug2")
        cand = resolve_framework_root(IPLUG2_SPEC, explicit=str(root))
        self.assertEqual(cand.source, "--iplug-path")
        self.assertEqual(cand.path, root)
        self.assertEqual(cand.version, "2.0.0")


# --------------------------------------------------------------------------- #
# Env vars.
# --------------------------------------------------------------------------- #
class EnvTest(_TmpBase):
    def test_env_var_is_honored(self):
        mods = _make_juce_modules(self.tmp / "env-juce" / "modules")
        cand = resolve_framework_root(JUCE_SPEC, env={"JUCE_PATH": str(mods)})
        self.assertEqual(cand.source, "env:JUCE_PATH")
        self.assertEqual(cand.path, mods)

    def test_explicit_beats_env(self):
        explicit_mods = _make_juce_modules(self.tmp / "explicit" / "modules")
        env_mods = _make_juce_modules(self.tmp / "fromenv" / "modules")
        cand = resolve_framework_root(
            JUCE_SPEC, explicit=str(explicit_mods),
            env={"JUCE_PATH": str(env_mods)})
        self.assertEqual(cand.path, explicit_mods)
        self.assertEqual(cand.source, "--juce-path")


# --------------------------------------------------------------------------- #
# Auto-discovery: tiers + ambiguity + dedupe.
# --------------------------------------------------------------------------- #
class AutoDiscoveryTest(_TmpBase):
    def test_vendored_outranks_well_known(self):
        # A copy vendored in the project must win over a well-known dir.
        project = self.tmp / "myplugin"
        project.mkdir()
        _make_juce_repo(project / "JUCE")           # vendored ./JUCE
        well_known = _make_juce_modules(self.tmp / "elsewhere" / "modules")
        spec = replace(JUCE_SPEC, well_known_roots=(well_known,))
        cand = resolve_framework_root(spec, project_dir=str(project))
        self.assertTrue(cand.source.startswith("vendored:"))
        self.assertEqual(cand.path, project / "JUCE" / "modules")
        # It is NOT the well-known dir.
        self.assertNotEqual(cand.path, well_known)

    def test_two_discovered_roots_raise_ambiguous(self):
        a = _make_juce_modules(self.tmp / "a" / "modules")
        b = _make_juce_modules(self.tmp / "b" / "modules")
        spec = replace(JUCE_SPEC, well_known_roots=(a, b))
        with self.assertRaises(FrameworkAmbiguous) as ctx:
            resolve_framework_root(spec)
        exc = ctx.exception
        paths = {c.path for c in exc.candidates}
        self.assertEqual(paths, {a, b})
        msg = str(exc)
        self.assertIn(str(a), msg)
        self.assertIn(str(b), msg)
        self.assertIn("--juce-path", msg)   # override command shown
        self.assertIn("JUCE_PATH", msg)

    def test_symlinked_duplicate_is_deduped(self):
        real = _make_juce_modules(self.tmp / "real" / "modules")
        link = self.tmp / "link"
        try:
            os.symlink(real.parent, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not supported on this platform")
        link_modules = link / "modules"
        spec = replace(JUCE_SPEC, well_known_roots=(real, link_modules))
        cand = resolve_framework_root(spec)   # must NOT be ambiguous
        self.assertEqual(_resolve(cand.path), _resolve(real))

    def test_zero_candidates_raise_not_found(self):
        empty = self.tmp / "no-checkouts-here"
        spec = replace(JUCE_SPEC, well_known_roots=(empty,))
        with self.assertRaises(FrameworkNotFound) as ctx:
            resolve_framework_root(spec, project_dir=str(self.tmp))
        msg = str(ctx.exception)
        self.assertIn("--juce-path", msg)
        self.assertIn("JUCE_PATH", msg)
        self.assertIn("git clone", msg)

    def test_project_cmake_add_subdirectory_reference(self):
        project = self.tmp / "proj"
        project.mkdir()
        sdk = _make_juce_repo(self.tmp / "sdk" / "JUCE")
        (project / "CMakeLists.txt").write_text(
            f"cmake_minimum_required(VERSION 3.15)\n"
            f"add_subdirectory({sdk} JUCE)\n", encoding="utf-8")
        # No vendored copy and no well-known dir: the project reference wins.
        spec = replace(JUCE_SPEC, well_known_roots=())
        cand = resolve_framework_root(spec, project_dir=str(project))
        self.assertTrue(cand.source.startswith("project:CMakeLists.txt:"))
        self.assertEqual(cand.path, sdk / "modules")


def _resolve(p: pathlib.Path) -> pathlib.Path:
    return p.resolve()


# --------------------------------------------------------------------------- #
# include_resolution_failed — distinguish a wrong include path from the
# unrelated parse errors a CORRECT run also emits.
# --------------------------------------------------------------------------- #
class IncludeResolutionTest(unittest.TestCase):
    def test_unrelated_rvalue_binding_error_is_ignored(self):
        diags = [
            "rvalue reference to type 'juce::String' cannot bind to lvalue of "
            "type 'juce::String'",
            {"severity": "error", "spelling": "expected ';' after top level "
                                              "declarator"},
        ]
        self.assertEqual(include_resolution_failed(diags, JUCE_SPEC), [])

    def test_file_not_found_on_framework_header_is_reported(self):
        diags = ["'juce_core/juce_core.h' file not found"]
        self.assertEqual(include_resolution_failed(diags, JUCE_SPEC),
                         ["juce_core/juce_core.h"])

    def test_file_not_found_on_juce_audio_processors(self):
        diags = ["'juce_audio_processors/juce_audio_processors.h' file not found"]
        self.assertEqual(include_resolution_failed(diags, JUCE_SPEC),
                         ["juce_audio_processors/juce_audio_processors.h"])

    def test_non_framework_missing_header_is_ignored(self):
        diags = ["'some/unrelated/header.h' file not found"]
        self.assertEqual(include_resolution_failed(diags, JUCE_SPEC), [])

    def test_iplug_header_prefix(self):
        diags = ["'IPlugAPIBase.h' file not found",
                 "'IGraphics.h' file not found"]
        self.assertEqual(include_resolution_failed(diags, IPLUG2_SPEC),
                         ["IPlugAPIBase.h", "IGraphics.h"])

    def test_accepts_diagnostic_objects_with_spelling_attribute(self):
        class _Diag:
            def __init__(self, spelling):
                self.spelling = spelling
        diags = [_Diag("'juce_dsp/juce_dsp.h' file not found")]
        self.assertEqual(include_resolution_failed(diags, JUCE_SPEC),
                         ["juce_dsp/juce_dsp.h"])


if __name__ == "__main__":
    unittest.main()
