"""Optional-integration detectors shared by framework importers.

The functions here stay source-framework neutral: they scan a user's project
for integration-specific evidence (headers, calls, and data files) and emit
ProjectIR `integration_requirements` data. A framework importer decides when to
call them and how to phrase its source-framework report.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ipp",
    ".m", ".mm",
}
TUNING_ASSET_SUFFIXES = {".scl", ".kbm", ".tun"}

SKIP_DIR_NAMES = {
    ".cache",
    ".git",
    ".hg",
    ".idea",
    ".svn",
    ".vs",
    ".vscode",
    "__pycache__",
    "_deps",
    "build",
    "builds",
    "deriveddata",
    "deps",
    "dependencies",
    "external",
    "extern",
    "juce",
    "jucelibrarycode",
    "node_modules",
    "third-party",
    "third_party",
    "vendor",
}

MTS_MARKERS = (
    "libMTSClient.h",
    "Client/libMTSClient.cpp",
    "MTS_RegisterClient",
    "MTS_DeregisterClient",
    "MTS_NoteToFrequency",
    "MTS_RetuningInSemitones",
    "MTS_RetuningAsRatio",
    "MTS_ShouldFilterNote",
    "MTS_FrequencyToNote",
    "MTS_ParseMIDIData",
    "MTS_ParseMIDIDataU",
    "MTS_HasMaster",
    "MTS_GetScaleName",
)

SCALA_MARKERS = (
    "Tunings::",
    "Tunings::Tuning",
    "Tunings::Scale",
    "Tunings::KeyboardMapping",
    "readSCLFile",
    "readKBMFile",
    "parseSCLData",
    "parseKBMData",
    "frequencyForMidiNote",
)


def _rel(project_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(project_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _line_for(text: str, needle: str) -> int:
    idx = text.find(needle)
    if idx < 0:
        return 1
    return text.count("\n", 0, idx) + 1


def _strip_comments_preserve_newlines(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)

    def repl(match: re.Match[str]) -> str:
        body = match.group(0)
        return "\n" * body.count("\n")

    return re.sub(r"/\*.*?\*/", repl, text, flags=re.DOTALL)


def _should_skip_dir(name: str) -> bool:
    lower = name.lower()
    return (
        lower in SKIP_DIR_NAMES
        or lower.startswith("build-")
        or lower.startswith("cmake-build")
    )


def _project_files(project_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root, dirs, names in os.walk(project_dir):
        dirs[:] = [d for d in dirs if not _should_skip_dir(d)]
        root_path = Path(root)
        files.extend(root_path / name for name in names)
    return sorted(files)


def _source_files(project_dir: Path) -> list[Path]:
    out: list[Path] = []
    for path in _project_files(project_dir):
        if not path.is_file():
            continue
        if path.suffix.lower() in SOURCE_SUFFIXES:
            out.append(path)
    return sorted(out)


def _asset_files(project_dir: Path) -> list[Path]:
    out: list[Path] = []
    for path in _project_files(project_dir):
        if not path.is_file():
            continue
        if path.suffix.lower() in TUNING_ASSET_SUFFIXES:
            out.append(path)
    return sorted(out)


def _scan_source_markers(project_dir: Path, markers: tuple[str, ...]) -> list[dict]:
    refs: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for path in _source_files(project_dir):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        code = _strip_comments_preserve_newlines(text)
        for marker in markers:
            if marker not in code:
                continue
            line = _line_for(code, marker)
            key = (_rel(project_dir, path), line)
            if key in seen:
                continue
            refs.append({"file": key[0], "line": line})
            seen.add(key)
            break
        if len(refs) >= 12:
            break
    return refs


def detect_tuning_integration_requirements(project_dir: Path) -> tuple[dict, list[dict], list[str]]:
    """Detect MTS-ESP and local tuning-file usage.

    Returns `(integration_requirements, diagnostics, migration_tasks)`.
    The first object is ProjectIR-shaped. Diagnostics and tasks are optional
    human surfaces the caller can append to its existing IR.
    """
    project_dir = project_dir.resolve()
    mts_refs = _scan_source_markers(project_dir, MTS_MARKERS)
    scala_refs = _scan_source_markers(project_dir, SCALA_MARKERS)
    assets = _asset_files(project_dir)

    scl_kbm_assets = [p for p in assets if p.suffix.lower() in (".scl", ".kbm")]
    tun_assets = [p for p in assets if p.suffix.lower() == ".tun"]

    packages: list[dict] = []
    options: list[dict] = []
    asset_inputs: list[dict] = []
    notes: list[str] = []
    diagnostics: list[dict] = []
    tasks: list[str] = []

    if mts_refs:
        packages.append({
            "id": "mts-esp",
            "feature_key": "session_tuning",
            "required": True,
            "reason": "Source project uses the ODDSound MTS-ESP client API.",
            "cmake_targets": ["mts_esp_client"],
            "source_refs": mts_refs,
        })
        options.append({
            "name": "PULP_ENABLE_MTS_ESP",
            "value": True,
            "reason": "Enable Pulp's provider-neutral MTS-ESP tuning provider.",
            "source_refs": mts_refs,
        })
        diagnostics.append({
            "severity": "info",
            "code": "tuning.mts_esp",
            "message": "MTS-ESP client usage detected; scaffold should map note-frequency queries through Pulp's provider-neutral tuning API.",
            "source_ref": mts_refs[0],
        })
        tasks.append("Map MTS-ESP note/frequency/status calls to `pulp::midi::TuningProvider` and keep client setup/status UI off the audio callback.")

    if scala_refs or scl_kbm_assets:
        refs = scala_refs or [{"file": _rel(project_dir, scl_kbm_assets[0])}]
        packages.append({
            "id": "sst-tuning-library",
            "feature_key": "local_tuning_files",
            "required": True,
            "reason": "Source project uses local Scala SCL/KBM tuning files or tuning-library symbols.",
            "cmake_targets": ["sst::tuning-library"],
            "source_refs": refs,
        })
        options.append({
            "name": "PULP_ENABLE_SCALA_TUNING",
            "value": True,
            "reason": "Enable Pulp's provider-neutral Scala SCL/KBM tuning provider.",
            "source_refs": refs,
        })
        diagnostics.append({
            "severity": "info",
            "code": "tuning.local_files",
            "message": "Local SCL/KBM tuning usage detected; scaffold should load these assets with Pulp's Scala tuning provider off the audio callback.",
            "source_ref": refs[0],
        })
        tasks.append("Load copied `.scl` / `.kbm` assets with `pulp::midi::ScalaTuningProvider` on the UI/main side, then query the provider from note-on or the original retuning cadence.")

    for asset in scl_kbm_assets:
        rel = _rel(project_dir, asset)
        asset_inputs.append({
            "path": rel,
            "kind": "keyboard_mapping" if asset.suffix.lower() == ".kbm" else "tuning_scale",
            "copy_policy": "copy_to_scaffold",
            "reason": "Preserve source-project local tuning asset.",
            "source_ref": {"file": rel},
        })

    for asset in tun_assets:
        rel = _rel(project_dir, asset)
        asset_inputs.append({
            "path": rel,
            "kind": "tuning_file",
            "copy_policy": "copy_to_scaffold",
            "requires_manual_review": True,
            "reason": "Preserve source-project .tun asset; direct .tun parsing is not represented by the Scala SCL/KBM provider.",
            "source_ref": {"file": rel},
        })
        diagnostics.append({
            "severity": "warning",
            "code": "tuning.tun_manual_review",
            "message": ".tun tuning asset detected; preserve it for review and use MTS-ESP Mini/session tuning or convert it to SCL/KBM.",
            "source_ref": {"file": rel},
        })
        tasks.append("Review copied `.tun` assets: use MTS-ESP Mini/session tuning or convert them to `.scl` / `.kbm` before loading through `ScalaTuningProvider`.")

    if mts_refs and (scala_refs or scl_kbm_assets):
        notes.append("Both session-wide MTS-ESP tuning and local SCL/KBM tuning were detected; generated Pulp code should prefer `MtsEspFallbackTuningProvider` so an active session master wins and local files remain the fallback.")
        tasks.append("Use `pulp::midi::MtsEspFallbackTuningProvider` when both session tuning and local tuning files are present.")

    reqs: dict = {}
    if packages:
        deduped: dict[str, dict] = {}
        for pkg in packages:
            deduped.setdefault(pkg["id"], pkg)
        reqs["packages"] = list(deduped.values())
    if options:
        deduped_opts: dict[str, dict] = {}
        for opt in options:
            deduped_opts.setdefault(opt["name"], opt)
        reqs["cmake_options"] = list(deduped_opts.values())
    if asset_inputs:
        reqs["asset_inputs"] = asset_inputs
    if notes:
        reqs["notes"] = notes

    return reqs, diagnostics, tasks
