"""Extraction-health contract — the shared false-clean guard.

Both importers historically returned exit 0 even when the extraction was
worthless: a bogus ``--framework-path`` made libclang emit
``'juce_audio_processors/...' file not found`` yet the tool still printed
``parameters: []`` and exited clean; an empty project produced an IR with an
``error`` key that ``--report`` rendered anyway. This module centralises the
decision "is this extraction trustworthy?" so every importer CLI derives its
exit status the same way, and stamps a machine-readable verdict into the IR that
the EMIT report renders as a loud banner.

The failure discriminator is framework-aware on purpose (see the JUCE vs iPlug2
asymmetry in the plan): a CORRECT JUCE parse also emits an unrelated
``severity: error`` rvalue-reference-binding diagnostic, so "any parse error =>
fail" would false-fail every JUCE import. We therefore only treat a
``file not found`` on a header matching the framework's own prefixes
(``juce_*`` / ``IPlug* / IGraphics*``) as a resolution failure — exactly what
:func:`discovery.include_resolution_failed` already computes.

Stdlib + :mod:`discovery` only; importable without libclang.
"""
from __future__ import annotations

from typing import Optional

from .discovery import FrameworkSpec, include_resolution_failed

__all__ = ["extraction_health", "stamp_extraction_health"]


def _diag_strings(ir: dict) -> list[object]:
    """Every diagnostic string the IR carries that could name an unresolved
    header: the authoritative ``extraction_context.parse_errors`` list plus the
    rolled-up ``diagnostics`` messages. Passed as-is to
    :func:`include_resolution_failed`, which coerces libclang Diagnostics, dicts,
    and plain strings to message text."""
    out: list[object] = []
    ctx = ir.get("extraction_context")
    if isinstance(ctx, dict):
        pe = ctx.get("parse_errors")
        if isinstance(pe, (list, tuple)):
            out.extend(pe)
    diags = ir.get("diagnostics")
    if isinstance(diags, (list, tuple)):
        out.extend(diags)
    return out


def extraction_health(
    ir: dict,
    *,
    framework_path_given: bool,
    spec: FrameworkSpec,
    allow_unresolved_includes: bool = False,
) -> tuple[bool, list[str]]:
    """Decide whether ``ir`` is a trustworthy extraction.

    Returns ``(ok, reasons)``. ``ok`` is False (and ``reasons`` non-empty) when
    ANY of the false-clean conditions hold:

      (a) a *framework* header failed to resolve — a ``file not found`` whose
          header matches ``spec``'s prefixes (an unrelated ``severity: error``
          from a CORRECT run does NOT count). Suppressed when the caller passes
          ``allow_unresolved_includes`` (the ``--allow-unresolved-includes``
          opt-out each extractor wires into its own argparse).
      (b) ``build_ir`` returned an ``error`` key (e.g. "no source file found").
      (c) a framework path was explicitly supplied but ``parameters`` is empty
          AND no ``constructs`` entry explains the absence.

    Callers use ``ok`` to choose a non-zero exit; :func:`stamp_extraction_health`
    records the verdict into the IR for the report banner.
    """
    reasons: list[str] = []

    # (b) explicit extractor error key — nothing else is meaningful.
    err = ir.get("error") if isinstance(ir, dict) else None
    if err:
        reasons.append(f"build_ir returned an error: {err}")

    # (a) framework include resolution failure.
    if not allow_unresolved_includes:
        missing = include_resolution_failed(_diag_strings(ir), spec)
        if missing:
            reasons.append(
                f"{spec.name} framework header(s) failed to resolve (file not "
                f"found): {', '.join(missing)} — the --framework-path is wrong "
                f"or incomplete, so extraction is unreliable")

    # (c) explicit path but nothing extracted and nothing explains why.
    if framework_path_given and not err:
        params = ir.get("parameters") or []
        constructs = ir.get("constructs") or []
        if not params and not constructs:
            reasons.append(
                "a framework path was supplied but no parameters were extracted "
                "and no construct explains the absence — likely the wrong source "
                "file / translation unit (params may live in another .cpp or a "
                "header)")

    return (not reasons), reasons


def stamp_extraction_health(
    ir: dict,
    *,
    framework_path_given: bool,
    spec: FrameworkSpec,
    allow_unresolved_includes: bool = False,
) -> tuple[bool, list[str]]:
    """Compute :func:`extraction_health` and record the verdict into ``ir`` in
    place: sets ``ir['extraction_health'] = {'ok', 'reasons'}`` and, when not
    ok, ``ir['extraction_failed'] = True`` (the flag the EMIT report renders as
    a banner). Returns the same ``(ok, reasons)`` so the caller can also pick an
    exit code without re-reading the IR.
    """
    ok, reasons = extraction_health(
        ir, framework_path_given=framework_path_given, spec=spec,
        allow_unresolved_includes=allow_unresolved_includes)
    ir["extraction_health"] = {"ok": ok, "reasons": reasons}
    if not ok:
        ir["extraction_failed"] = True
    return ok, reasons
