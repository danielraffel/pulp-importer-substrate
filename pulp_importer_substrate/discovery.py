"""Framework checkout discovery — find (and validate) a JUCE / iPlug2 tree.

Why this module exists (verified bugs, reproduced by running the importers):

* `pulp-import-juce` and `pulp-import-iplug` NEVER validated their
  ``--juce-path`` / ``--iplug-path``. They just appended ``-I{path}``. A bogus
  path — or JUCE's *repository root* instead of ``JUCE/modules`` — makes
  libclang emit ``'juce_audio_processors/juce_audio_processors.h' file not
  found`` and the tool STILL exits 0 with ``parameters: []``. A silent, false
  clean import.
* Users should not have to pass the path at all when it is discoverable, but a
  machine can hold several checkouts. Multiple candidates must NOT be silently
  guessed.

This module is the shared, dependency-free (stdlib only) answer:

* ``resolve_framework_root(spec, ...)`` resolves an *authoritative* source
  (explicit CLI value, then env var) with strict validation, or falls back to
  auto-discovery (project-local, then well-known). It NEVER guesses between
  multiple discovered roots — it raises :class:`FrameworkAmbiguous` carrying
  machine-readable ``.candidates`` so a caller can emit ProjectIR diagnostics.
* :data:`JUCE_SPEC` / :data:`IPLUG2_SPEC` describe each framework: how to
  validate a root, how to normalize a repo root to its canonical include root,
  and how to probe a version.
* :func:`include_resolution_failed` turns libclang parse diagnostics into the
  list of *framework* headers that were ``file not found`` — while ignoring the
  unrelated ``severity: error`` diagnostics a CORRECT run also emits.

The module is importable without libclang and pulls in nothing outside the
standard library.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

# A normalizer returns the canonical include root plus an optional human note
# describing any non-trivial mapping (e.g. "repo root -> modules/"), or None if
# the supplied path cannot be made into a valid root.
NormalizeResult = Optional["tuple[Path, Optional[str]]"]


@dataclass(frozen=True)
class Candidate:
    """A discovered, validated framework include root.

    ``path`` is the canonical include root (JUCE -> ``.../modules`` ; iPlug2 ->
    the checkout root). ``source`` is provenance — e.g. ``"--juce-path"``,
    ``"env:JUCE_PATH"``, ``"project:CMakeLists.txt:12"``, ``"vendored:./JUCE"``,
    ``"well-known:~/JUCE"``. ``note`` records a normalization that happened on
    the way here (e.g. a repo root remapped to its modules dir).
    """

    path: Path
    source: str
    version: Optional[str] = None
    note: Optional[str] = None


@dataclass(frozen=True)
class FrameworkSpec:
    """Everything discovery needs to know about one framework.

    All behaviour is injected as callables/data so this module names no vendor
    logic itself — the same resolution engine drives JUCE and iPlug2.
    """

    name: str                          # "JUCE" / "iPlug2"
    cli_flag: str                      # "--juce-path"
    env_vars: tuple[str, ...]          # ("JUCE_PATH", "JUCE_DIR")
    clone_command: str                 # shown in FrameworkNotFound

    # Is ``path`` already a canonical include root? (JUCE: <p>/juce_core/... ;
    # iPlug2: <p>/IPlug/IPlugAPIBase.h)
    validator: Callable[[Path], bool] = field(repr=False)
    # Map an arbitrary user-supplied path to its canonical include root.
    # Returns (canonical_root, note) or None when it cannot be resolved.
    normalizer: Callable[[Path], NormalizeResult] = field(repr=False)
    # Human string naming the file(s) that were missing, for fatal messages.
    missing_hint: Callable[[Path], str] = field(repr=False)
    # Detected version of a validated root, or None.
    version_probe: Callable[[Path], Optional[str]] = field(repr=False)

    # Header path prefixes that identify a *framework* include (for
    # include_resolution_failed). JUCE: ("juce_",) ; iPlug2: ("IPlug","IGraphics")
    header_prefixes: tuple[str, ...] = ()

    # Auto-discovery inputs.
    vendored_dir_names: tuple[str, ...] = ()      # ("JUCE",)
    vendored_parent_subdirs: tuple[str, ...] = ()  # ("", "libs", "external", ...)
    # (glob, regex-with-group-1-path) pairs applied under project_dir. A match's
    # captured path (resolved against the file's dir) is normalized + validated.
    project_hint_patterns: tuple[tuple[str, str], ...] = ()
    well_known_roots: tuple[Path, ...] = ()


# --------------------------------------------------------------------------- #
# Exceptions — both carry machine-readable ``.candidates``.
# --------------------------------------------------------------------------- #
class FrameworkResolutionError(Exception):
    """Base for all discovery failures. ``.candidates`` is always present (a
    possibly-empty list) so a caller can emit ProjectIR diagnostics."""

    def __init__(self, message: str, spec: "FrameworkSpec",
                 candidates: Sequence[Candidate]):
        super().__init__(message)
        self.spec = spec
        self.candidates: list[Candidate] = list(candidates)


class FrameworkNotFound(FrameworkResolutionError):
    """No usable checkout — either an invalid explicit/env path, or zero
    auto-discovered candidates."""


class FrameworkAmbiguous(FrameworkResolutionError):
    """More than one distinct checkout was auto-discovered. Discovery refuses to
    guess; the caller must disambiguate."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _display_path(p: Path) -> str:
    """Render a path with ``~`` for the home dir, for readable provenance."""
    try:
        home = Path.home()
        rp = p if p.is_absolute() else p
        return "~/" + str(rp.relative_to(home)) if _is_relative_to(rp, home) else str(rp)
    except (ValueError, RuntimeError):
        return str(p)


def _is_relative_to(p: Path, other: Path) -> bool:
    try:
        p.relative_to(other)
        return True
    except ValueError:
        return False


def _resolved_key(p: Path) -> Path:
    """Canonical identity for dedupe — follows symlinks; falls back to
    absolute-normalized if the path does not exist on disk."""
    try:
        return p.resolve()
    except (OSError, RuntimeError):
        return Path(os.path.abspath(str(p)))


def _make_candidate(spec: FrameworkSpec, raw: Path, source: str) -> Optional[Candidate]:
    """Normalize + validate ``raw``; build a Candidate or return None."""
    norm = spec.normalizer(raw)
    if norm is None:
        return None
    path, note = norm
    version = spec.version_probe(path)
    return Candidate(path=path, source=source, version=version, note=note)


def _dedupe(cands: Iterable[Candidate]) -> list[Candidate]:
    """Collapse candidates that resolve to the same on-disk root, keeping the
    first (highest priority) occurrence."""
    seen: dict[Path, Candidate] = {}
    for c in cands:
        key = _resolved_key(c.path)
        if key not in seen:
            seen[key] = c
    return list(seen.values())


# --------------------------------------------------------------------------- #
# Authoritative sources: explicit CLI value, then env var.
# --------------------------------------------------------------------------- #
def _resolve_authoritative(spec: FrameworkSpec, raw: str, source: str) -> Candidate:
    p = Path(raw).expanduser()
    cand = _make_candidate(spec, p, source)
    if cand is not None:
        return cand
    missing = spec.missing_hint(p)
    raise FrameworkNotFound(
        f"{source} '{raw}' is not a valid {spec.name} checkout: {missing}. "
        f"Pass a valid {spec.cli_flag} (or set {spec.env_vars[0]}), "
        f"or clone {spec.name}: {spec.clone_command}",
        spec, [])


# --------------------------------------------------------------------------- #
# Auto-discovery — two tiers. Tier A (project-local) outranks Tier B
# (well-known): if the project vendors or references a checkout, that is what it
# actually builds against, so well-known dirs are never consulted as rivals.
# --------------------------------------------------------------------------- #
def _discover_project_local(spec: FrameworkSpec, project_dir: Path) -> list[Candidate]:
    out: list[Candidate] = []

    # (a) vendored copies: <project>/JUCE, <project>/libs/JUCE, etc.
    for sub in spec.vendored_parent_subdirs:
        for name in spec.vendored_dir_names:
            raw = (project_dir / sub / name) if sub else (project_dir / name)
            if not raw.exists():
                continue
            rel = "./" + str(raw.relative_to(project_dir)) if _is_relative_to(raw, project_dir) else str(raw)
            c = _make_candidate(spec, raw, f"vendored:{rel}")
            if c:
                out.append(c)

    # (b) git submodules referenced in .gitmodules.
    gm = project_dir / ".gitmodules"
    if gm.is_file():
        try:
            text = gm.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for m in re.finditer(r"^\s*path\s*=\s*(.+?)\s*$", text, re.MULTILINE):
            raw = (project_dir / m.group(1)).resolve()
            c = _make_candidate(spec, raw, f"submodule:{m.group(1)}")
            if c:
                out.append(c)

    # (c) paths referenced by the project's own build files.
    for glob_pat, regex in spec.project_hint_patterns:
        rx = re.compile(regex)
        for build_file in sorted(project_dir.glob(glob_pat)):
            if not build_file.is_file():
                continue
            try:
                lines = build_file.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            rel_name = build_file.relative_to(project_dir) if _is_relative_to(build_file, project_dir) else build_file.name
            for lineno, line in enumerate(lines, start=1):
                mm = rx.search(line)
                if not mm:
                    continue
                ref = mm.group(1).strip().strip('"').strip("'")
                if not ref:
                    continue
                cand_path = Path(ref).expanduser()
                if not cand_path.is_absolute():
                    cand_path = (build_file.parent / cand_path)
                c = _make_candidate(spec, cand_path, f"project:{rel_name}:{lineno}")
                if c:
                    out.append(c)

    return _dedupe(out)


def _discover_well_known(spec: FrameworkSpec, project_dir: Optional[Path]) -> list[Candidate]:
    out: list[Candidate] = []
    roots: list[tuple[Path, str]] = []

    for r in spec.well_known_roots:
        roots.append((r.expanduser(), f"well-known:{_display_path(r)}"))

    # Siblings of the project dir (a checkout next to the project).
    if project_dir is not None:
        for name in spec.vendored_dir_names:
            sib = project_dir.parent / name
            roots.append((sib, f"sibling:{sib}"))

    for raw, source in roots:
        if not raw.exists():
            continue
        c = _make_candidate(spec, raw, source)
        if c:
            out.append(c)
    return _dedupe(out)


def _prompt_choice(spec: FrameworkSpec, candidates: list[Candidate]) -> Candidate:
    sys.stderr.write(_ambiguous_message(spec, candidates) + "\n")
    while True:
        sys.stderr.write(f"Select a {spec.name} checkout [1-{len(candidates)}]: ")
        sys.stderr.flush()
        try:
            choice = sys.stdin.readline().strip()
        except (EOFError, KeyboardInterrupt):
            raise FrameworkAmbiguous(_ambiguous_message(spec, candidates), spec, candidates)
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]


def _ambiguous_message(spec: FrameworkSpec, candidates: list[Candidate]) -> str:
    lines = [f"Multiple {spec.name} checkouts found; refusing to guess. Candidates:"]
    for i, c in enumerate(candidates, start=1):
        ver = c.version or "unknown version"
        lines.append(f"  {i}. {c.path}  (source: {c.source}, version: {ver})")
    lines.append("Choose one explicitly:")
    env = spec.env_vars[0]
    for c in candidates:
        lines.append(f"  {spec.cli_flag} {c.path}   (or {env}={c.path})")
    return "\n".join(lines)


def _not_found_message(spec: FrameworkSpec) -> str:
    envs = ", ".join(spec.env_vars)
    return (f"No {spec.name} checkout found. Pass {spec.cli_flag} <path>, "
            f"set {envs}, or clone {spec.name}: {spec.clone_command}")


def resolve_framework_root(
    spec: FrameworkSpec,
    explicit: Optional[str] = None,
    project_dir: Optional[os.PathLike | str] = None,
    env: "os._Environ | dict[str, str]" = os.environ,
    interactive: bool = False,
) -> Candidate:
    """Resolve a framework include root.

    Order — the first *authoritative* source wins; auto-discovery then collects
    ALL candidates and never guesses:

    1. ``explicit`` CLI value — validated. Invalid raises :class:`FrameworkNotFound`
       naming the flag and the missing file. A JUCE *repo root* whose
       ``modules/`` is valid is accepted with a normalization note.
    2. ``env`` var(s) — same validation.
    3. project-local auto-detect (vendored copy / submodule / build-file
       reference) — outranks well-known dirs.
    4. well-known locations + siblings of ``project_dir``.

    Auto-discovery returning exactly one distinct root returns it; more than one
    raises :class:`FrameworkAmbiguous` (unless ``interactive`` and stdin is a
    TTY, in which case the user is prompted); zero raises
    :class:`FrameworkNotFound`.
    """
    pdir = Path(project_dir).expanduser() if project_dir is not None else None

    # 1. explicit — authoritative, exactly one answer or fatal.
    if explicit is not None and str(explicit) != "":
        return _resolve_authoritative(spec, str(explicit), spec.cli_flag)

    # 2. env var(s) — authoritative.
    for var in spec.env_vars:
        val = env.get(var)
        if val:
            return _resolve_authoritative(spec, val, f"env:{var}")

    # 3. project-local (Tier A) — short-circuits well-known.
    if pdir is not None:
        tier_a = _discover_project_local(spec, pdir)
        if len(tier_a) == 1:
            return tier_a[0]
        if len(tier_a) > 1:
            if interactive and sys.stdin.isatty():
                return _prompt_choice(spec, tier_a)
            raise FrameworkAmbiguous(_ambiguous_message(spec, tier_a), spec, tier_a)

    # 4. well-known + siblings (Tier B).
    tier_b = _discover_well_known(spec, pdir)
    if len(tier_b) == 0:
        raise FrameworkNotFound(_not_found_message(spec), spec, [])
    if len(tier_b) == 1:
        return tier_b[0]
    if interactive and sys.stdin.isatty():
        return _prompt_choice(spec, tier_b)
    raise FrameworkAmbiguous(_ambiguous_message(spec, tier_b), spec, tier_b)


# --------------------------------------------------------------------------- #
# libclang include-resolution diagnostics.
# --------------------------------------------------------------------------- #
_FILE_NOT_FOUND_RX = re.compile(r"'([^']+)'\s+file not found")


def _diag_message(d: object) -> str:
    """Coerce a libclang Diagnostic, a dict, or a str into its message text."""
    for attr in ("spelling", "message"):
        val = getattr(d, attr, None)
        if isinstance(val, str) and val:
            return val
    if isinstance(d, dict):
        for k in ("spelling", "message", "text"):
            v = d.get(k)
            if isinstance(v, str) and v:
                return v
    return str(d)


def _header_matches_spec(header: str, spec: FrameworkSpec) -> bool:
    # Match against the leading path component and the basename, so both
    # 'juce_audio_processors/juce_audio_processors.h' and 'IPlugAPIBase.h'
    # resolve against the spec's prefixes.
    lead = header.replace("\\", "/").split("/", 1)[0]
    base = header.replace("\\", "/").rsplit("/", 1)[-1]
    return any(lead.startswith(pfx) or base.startswith(pfx) for pfx in spec.header_prefixes)


def include_resolution_failed(diagnostics: Iterable[object], spec: FrameworkSpec) -> list[str]:
    """Return the *framework* headers that libclang reported as ``file not
    found``.

    A CORRECT parse also emits unrelated ``severity: error`` diagnostics (e.g.
    JUCE's rvalue-reference-binding one), so callers must NOT treat "any parse
    error" as a wrong include path. Only a ``file not found`` on a header whose
    path matches this spec's :attr:`~FrameworkSpec.header_prefixes` (``juce_*`` /
    ``IPlug* / IGraphics*``) proves the include path is wrong. Returns ``[]``
    when the include path is fine, preserving order and de-duplicating.
    """
    out: list[str] = []
    for d in diagnostics:
        msg = _diag_message(d)
        m = _FILE_NOT_FOUND_RX.search(msg)
        if not m:
            continue
        header = m.group(1)
        if _header_matches_spec(header, spec) and header not in out:
            out.append(header)
    return out


# --------------------------------------------------------------------------- #
# JUCE spec.
# --------------------------------------------------------------------------- #
def _juce_validate(p: Path) -> bool:
    return (p / "juce_core" / "juce_core.h").is_file()


def _juce_normalize(p: Path) -> NormalizeResult:
    if _juce_validate(p):
        return (p, None)
    # The #1 mistake: user passed the repository root instead of modules/.
    mods = p / "modules"
    if _juce_validate(mods):
        return (mods, f"normalized repository root '{p}' to its JUCE modules "
                      f"directory '{mods}'")
    return None


def _juce_missing_hint(p: Path) -> str:
    return (f"neither '{p / 'juce_core' / 'juce_core.h'}' nor "
            f"'{p / 'modules' / 'juce_core' / 'juce_core.h'}' exists "
            f"(expected the JUCE modules/ directory or the repo root above it)")


_JUCE_DECL_VERSION_RX = re.compile(
    r"BEGIN_JUCE_MODULE_DECLARATION(.*?)END_JUCE_MODULE_DECLARATION", re.DOTALL)
_JUCE_VERSION_FIELD_RX = re.compile(r"^\s*version:\s*(\S+)", re.MULTILINE)
_JUCE_MAJOR_RX = re.compile(r"#define\s+JUCE_MAJOR_VERSION\s+(\d+)")
_JUCE_MINOR_RX = re.compile(r"#define\s+JUCE_MINOR_VERSION\s+(\d+)")
_JUCE_BUILD_RX = re.compile(r"#define\s+JUCE_BUILDNUMBER\s+(\d+)")


def _juce_version(root: Path) -> Optional[str]:
    # Primary: the version: field of juce_core's module-declaration block.
    header = root / "juce_core" / "juce_core.h"
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    block = _JUCE_DECL_VERSION_RX.search(text)
    if block:
        vf = _JUCE_VERSION_FIELD_RX.search(block.group(1))
        if vf:
            return vf.group(1)
    # Fallback: the JUCE_*_VERSION macros in juce_StandardHeader.h.
    std = root / "juce_core" / "system" / "juce_StandardHeader.h"
    try:
        stext = std.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    maj = _JUCE_MAJOR_RX.search(stext)
    minr = _JUCE_MINOR_RX.search(stext)
    bld = _JUCE_BUILD_RX.search(stext)
    if maj and minr and bld:
        return f"{maj.group(1)}.{minr.group(1)}.{bld.group(1)}"
    return None


JUCE_SPEC = FrameworkSpec(
    name="JUCE",
    cli_flag="--juce-path",
    env_vars=("JUCE_PATH", "JUCE_DIR"),
    clone_command="git clone https://github.com/juce-framework/JUCE",
    validator=_juce_validate,
    normalizer=_juce_normalize,
    missing_hint=_juce_missing_hint,
    version_probe=_juce_version,
    header_prefixes=("juce_",),
    vendored_dir_names=("JUCE",),
    vendored_parent_subdirs=("", "libs", "external", "third_party", "third-party", "modules", "deps"),
    project_hint_patterns=(
        # add_subdirectory(<path> ...) — captures the first argument.
        ("CMakeLists.txt", r"add_subdirectory\s*\(\s*([^\s\)]+)"),
        ("**/CMakeLists.txt", r"add_subdirectory\s*\(\s*([^\s\)]+)"),
        # set(JUCE_PATH <path>) / set(JUCE_DIR <path>) and cache forms.
        ("CMakeLists.txt", r"set\s*\(\s*JUCE_(?:PATH|DIR)\s+([^\s\)]+)"),
        ("CMakeCache.txt", r"JUCE_(?:PATH|DIR)(?::[A-Z]+)?\s*=\s*(.+)"),
        # .jucer projects reference their JUCE modules folder.
        ("*.jucer", r'juceFolder\s*=\s*"([^"]+)"'),
    ),
    well_known_roots=(
        Path("~/JUCE"), Path("~/SDKs/JUCE"), Path("~/Code/JUCE"),
        Path("~/dev/JUCE"), Path("~/Developer/JUCE"),
        Path("/usr/local/JUCE"), Path("/opt/JUCE"), Path("/Applications/JUCE"),
    ),
)


# --------------------------------------------------------------------------- #
# iPlug2 spec.
#
# Asymmetry vs JUCE, on purpose: JUCE's canonical include root is a SUBDIR
# (modules/) of the repository, so a supplied repo root is remapped DOWN into
# modules/. iPlug2's canonical include root IS the repository root itself
# (headers live in <root>/IPlug and <root>/IGraphics, both added by the iPlug2
# CMake), so there is nothing to remap — the normalizer only accepts a path that
# is already the checkout root and never rewrites it into a subdir.
# --------------------------------------------------------------------------- #
def _iplug_validate(p: Path) -> bool:
    # IPlug/IPlugAPIBase.h genuinely proves an iPlug2 checkout root (verified
    # against /Users/danielraffel/Code/iPlug2).
    return (p / "IPlug" / "IPlugAPIBase.h").is_file()


def _iplug_normalize(p: Path) -> NormalizeResult:
    if _iplug_validate(p):
        return (p, None)
    return None


def _iplug_missing_hint(p: Path) -> str:
    return (f"'{p / 'IPlug' / 'IPlugAPIBase.h'}' does not exist "
            f"(expected the iPlug2 checkout root, not a subdirectory)")


_IPLUG_CMAKE_VERSION_RX = re.compile(r"project\s*\(\s*iPlug2\b[^)]*?VERSION\s+([0-9][0-9.]*)",
                                     re.IGNORECASE | re.DOTALL)
_IPLUG_MACRO_VERSION_RX = re.compile(r"#define\s+IPLUG_VERSION\s+0x([0-9A-Fa-f]+)")


def _iplug_version(root: Path) -> Optional[str]:
    # Primary: iPlug2 CMake project() version. (No IPlugVersion.h ships in the
    # tree; the CMake project() call is the authoritative source.)
    cml = root / "CMakeLists.txt"
    try:
        ctext = cml.read_text(encoding="utf-8", errors="replace")
    except OSError:
        ctext = ""
    m = _IPLUG_CMAKE_VERSION_RX.search(ctext)
    if m:
        return m.group(1)
    # Fallback: IPLUG_VERSION macro (0xVVVVRRMM) in IPlug/IPlugConstants.h.
    consts = root / "IPlug" / "IPlugConstants.h"
    try:
        htext = consts.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    mm = _IPLUG_MACRO_VERSION_RX.search(htext)
    if mm:
        raw = int(mm.group(1), 16)
        return f"{(raw >> 16) & 0xFFFF}.{(raw >> 8) & 0xFF}.{raw & 0xFF}"
    return None


IPLUG2_SPEC = FrameworkSpec(
    name="iPlug2",
    cli_flag="--iplug-path",
    env_vars=("IPLUG2_PATH", "IPLUG2_DIR"),
    clone_command="git clone https://github.com/iPlug2/iPlug2",
    validator=_iplug_validate,
    normalizer=_iplug_normalize,
    missing_hint=_iplug_missing_hint,
    version_probe=_iplug_version,
    header_prefixes=("IPlug", "IGraphics"),
    vendored_dir_names=("iPlug2",),
    vendored_parent_subdirs=("", "libs", "external", "third_party", "third-party", "deps", "Dependencies"),
    project_hint_patterns=(
        ("CMakeLists.txt", r"add_subdirectory\s*\(\s*([^\s\)]+)"),
        ("**/CMakeLists.txt", r"add_subdirectory\s*\(\s*([^\s\)]+)"),
        ("CMakeLists.txt", r"set\s*\(\s*IPLUG2_(?:PATH|DIR)\s+([^\s\)]+)"),
        ("CMakeCache.txt", r"IPLUG2_(?:PATH|DIR)(?::[A-Z]+)?\s*=\s*(.+)"),
    ),
    well_known_roots=(
        Path("~/iPlug2"), Path("~/SDKs/iPlug2"), Path("~/Code/iPlug2"),
        Path("~/dev/iPlug2"), Path("~/Developer/iPlug2"),
        Path("/usr/local/iPlug2"), Path("/opt/iPlug2"),
    ),
)


__all__ = [
    "Candidate",
    "FrameworkSpec",
    "FrameworkResolutionError",
    "FrameworkNotFound",
    "FrameworkAmbiguous",
    "resolve_framework_root",
    "include_resolution_failed",
    "JUCE_SPEC",
    "IPLUG2_SPEC",
]
