"""Vendor-agnostic EMIT core shared by Pulp framework importers.

INSPECT (each importer's `extract.py`) reads the user's own plugin source
read-only and produces a vendor-neutral ProjectIR. EMIT consumes that IR and
proposes a Pulp *migration scaffold* — a project that builds against the Pulp
SDK and is an honest starting point for the port, NOT a finished plugin.

The ProjectIR is framework-neutral by design (it carries already-resolved
fields like `dsp.mappings`, `parameters[].pulp_range`, `source_curve`,
classifications), so every generator here consumes IR fields and names no
vendor. The two genuinely framework-specific touch-points are injected by the
caller:

  * `render_report` — each importer renders its own IMPORT_REPORT.md prose
    (column sets and provenance wording differ per framework); the shared core
    appends an EMIT verdict block on top of whatever the importer renders.
  * `framework_free_predicate` — deciding whether a user header is
    framework-free enough to copy verbatim into the scaffold requires knowing
    the framework's source tells (its namespace qualifier / include prefix).
    The substrate provides `make_framework_free_predicate(tokens)` so an
    importer passes DATA (its token list), never code.

Honesty by construction (plan §§7.4/7.5/16.3):
  - Never guess. Every value that could not be statically resolved is emitted
    as a labelled stub with a `// TODO(import): …` comment, not invented.
  - Skewed / symmetric source parameter curves are emitted as a shaped Pulp
    ParamRange (the {min,max,default,step,skew,symmetric_skew} aggregate), so
    the non-linear curve round-trips through Pulp's normalize/denormalize
    (CLOSE-with-tolerance) instead of being downgraded to LINEAR+PARTIAL.

The `source_curve` IR field (the curve contract) — READ THIS BEFORE TOUCHING
the emitter. It carries values that are ALREADY IN PULP CONVENTION; each vendor
extractor converts vendor->Pulp *before* populating the IR, so the emitter here
stays dumb and never special-cases a vendor:

    source_curve = {
      "shape":     "linear" | "pow" | "exp",  # provenance: what the SOURCE used
      "skew":      float | None,   # PULP convention already (norm^(1/skew));
                                   #   None for shapes with no pow-skew value.
      "symmetric": bool,
      "fidelity":  "exact" | "close",  # "close" => keep a PARTIAL diagnostic
      "centre":    float | None,   # for ParamRange::with_centre() emission
    }

Why Pulp-convention and not raw vendor values: Pulp's skew explicitly matches
`juce::NormalisableRange` (denormalize = min + pow(norm, 1/skew)*Δ), so a JUCE
skew copies through unchanged, but an iPlug2 `ShapePowCurve(n)` exponent is the
RECIPROCAL — the iPlug extractor stores `skew = 1/n`. That conversion is the
extractor's job. The emitter only reads Pulp-convention fields:
  - "shaped" is driven off the shape enum + centre + symmetric, NEVER off a
    sentinel `skew == None` (an exp curve legitimately has skew=None and must
    NOT read as unshaped/linear).
  - shape=="exp" (logarithmic) is not representable in the pow-skew family, so
    it emits `ParamRange::with_centre(min, max, centre)` + a PARTIAL/CLOSE
    diagnostic — NEVER a plain linear range.
  - the "round-trips within float tolerance" comment is emitted only when
    fidelity=="exact".
  - DSP is scaffolded per the IR classification: pass-through for effects,
    labelled silence for instruments. The real DSP migration is a TODO.
  - Opaque-custom state => an explicit "binary session compatibility is NOT
    supported" TODO. Parameter state still round-trips.
  - Framework-free `portable-core` DSP files are COPIED verbatim
    (provenance: copied-user-file) — never rewritten.

This is spike code, not the shipped importer. It emits a CLAP-first scaffold
(self-contained, no external SDK, dlopen-testable) and mirrors the metadata
into whatever other FORMATS the IR reported so `pulp_add_plugin` lists them.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Callable

from .integration_requirements import (
    cmake_value as _cmake_value,
    gen_cmake_prelude as _gen_integration_cmake_prelude,
    gen_target_links as _gen_integration_target_links,
    integration_requirements as _integration_requirements,
)

__all__ = [
    "FileSpec",
    "DEFAULT_CONFIDENCE_FLOOR",
    "produce",
    "emit",
    "make_framework_free_predicate",
]


# Confidence floor for emitting a concrete parameter value. The INSPECT step
# scores every parameter with a `confidence` in [0, 1]; a value below this floor
# was inferred from non-literal / data-driven / runtime-computed source (e.g. an
# iPlug2 InitDouble with computed arguments, or a JUCE range built from a runtime
# expression) and is therefore a GUESS, not a fact. Honesty rule (plan §§7.4/16.3):
# never emit a guessed value as if it were certain. Below the floor we emit a
# clearly-labelled `// TODO(import): low-confidence … — verify before trusting`
# stub instead of a concrete `store.add_parameter({...})` block.
#
# 0.5 splits the importers' two confidence bands cleanly: statically-resolved
# params score high (~0.9+), computed/data-driven params score low (~0.2), so the
# floor downgrades exactly the guessed ones and leaves the resolved ones concrete.
DEFAULT_CONFIDENCE_FLOOR = 0.5


# --- naming helpers ---------------------------------------------------------

def _pascal(name: str) -> str:
    """A C++-identifier-safe PascalCase token from an arbitrary product name."""
    parts = re.split(r"[^0-9A-Za-z]+", name or "")
    tok = "".join(p[:1].upper() + p[1:] for p in parts if p)
    if not tok:
        tok = "ImportedPlugin"
    if tok[0].isdigit():
        tok = "P" + tok
    return tok


def _snake(name: str) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", name or "")
    tok = "_".join(p.lower() for p in parts if p)
    return tok or "imported_plugin"


def _member_token(symbol: str) -> str:
    """`pulp::signal::Compressor` -> `compressor`.

    A snake_case member-name hint for the suggested-declaration comment in the
    DSP scaffold. Best-effort cosmetic only — never load-bearing.
    """
    leaf = symbol.split("::")[-1] or "dsp"
    out = []
    for i, c in enumerate(leaf):
        if c.isupper() and i > 0 and not leaf[i - 1].isupper():
            out.append("_")
        out.append(c.lower())
    return "".join(out) or "dsp"


def _cpp_str(s) -> str:
    """Escape a Python string for a C++ string literal."""
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _cpp_float(v, fallback: float = 0.0) -> str:
    """Render a numeric IR value as a C++ float literal, never crashing on None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = fallback
    # keep it a float literal so {min,max,default,step} aggregate-inits cleanly.
    # Use repr() and guarantee a decimal point so integral values like -60.0
    # render as "-60.0f" (a valid float literal), never "-60f".
    s = repr(f)
    if "." not in s and "e" not in s and "E" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s + "f"


CATEGORY_MAP = {
    "Effect": "Effect",
    "Instrument": "Instrument",
    "MidiEffect": "MidiEffect",
}

# Format tokens the IR may carry -> pulp_add_plugin FORMATS tokens. The spike
# scaffold always builds CLAP (self-contained); other formats are listed when
# the source declared them so the migration target matches the original.
FORMAT_MAP = {
    "VST3": "VST3",
    "AU": "AU",
    "AUv3": "AUv3",
    "AAX": "AAX",
    "CLAP": "CLAP",
    "Standalone": "Standalone",
    "LV2": "LV2",
}


# --- audio precision --------------------------------------------------------

def _audio_precision(ir: dict) -> dict:
    value = ir.get("audio_precision")
    return value if isinstance(value, dict) else {}


def _source_native_f64(ir: dict) -> bool:
    return bool(_audio_precision(ir).get("source_native_f64"))


def _zero_literal(sample_type: str) -> str:
    return "0.0" if sample_type == "double" else "0.0f"


# --- parameter emission -----------------------------------------------------

# Construct types whose call site cannot be reduced to a single static
# parameter — a loop / factory registers an unknown COUNT of params at one call
# site, so any single `parameters[]` entry extracted at that exact site is a
# PHANTOM (the S3 double-report bug). `computed_args` is a single-call whose
# arguments are runtime-computed: that IS a real (if low-confidence) parameter,
# so it is deliberately NOT treated as claiming/suppressing its own entry.
_CLAIMING_CONSTRUCT_TYPES = frozenset({"loop_or_factory"})


def _construct_claimed_refs(ir: dict) -> set[tuple]:
    """(file, line) source-refs claimed by a loop/factory construct."""
    claimed: set[tuple] = set()
    for c in ir.get("constructs", []) or []:
        if c.get("construct_type") not in _CLAIMING_CONSTRUCT_TYPES:
            continue
        ref = c.get("source_ref") or {}
        f, ln = ref.get("file"), ref.get("line")
        if f is not None and ln is not None:
            claimed.add((f, ln))
    return claimed


def _suppress_construct_claimed_params(ir: dict) -> dict:
    """Drop any `parameters[]` entry whose source_ref collides with a
    loop_or_factory construct — such a call is already reported (with unknown
    cardinality) as a construct, so also emitting it as a high-confidence param
    is the S3 double-report. Returns a shallow copy with the phantom filtered
    out (originals are never mutated); a no-op copy when nothing collides.
    """
    claimed = _construct_claimed_refs(ir)
    if not claimed:
        return ir
    kept: list[dict] = []
    for p in ir.get("parameters", []) or []:
        ref = p.get("source_ref") or {}
        key = (ref.get("file"), ref.get("line"))
        if key in claimed:
            continue  # phantom: the construct already accounts for this call
        kept.append(p)
    new = dict(ir)
    new["parameters"] = kept
    return new


def _param_id_enum(params: list[dict]) -> list[tuple[str, int, str]]:
    """(enum_name, stable_uint32_id, source_id_string) for each resolvable param.

    The stable id is the IR's `proposed_pulp_id` (FNV-1a of the source id
    string) so preset/automation identity survives the import. Params the AST
    could not resolve (id is null) are skipped here and surfaced as
    constructs/TODOs.
    """
    out = []
    seen = set()
    for p in params:
        pid = p.get("proposed_pulp_id")
        sid = p.get("source_id_string") or p.get("id")
        if pid is None or sid is None:
            continue
        enum = "kParam_" + _snake(sid)
        if enum in seen:  # defensive: dedupe collisions
            enum = f"{enum}_{pid}"
        seen.add(enum)
        out.append((enum, int(pid), sid))
    return out


def _param_discrete_divisor(p: dict) -> int | None:
    """The normalization divisor (normDen) for a DISCRETE parameter: the number
    of intervals between its positions, so a UI maps position `i` to `i /
    divisor` in [0, 1] (an N-position control has divisor N-1). Derived once,
    here, from the parameter's own definition so a generated binding cannot
    disagree with it — the class of hand-transcription bug where a 3-option
    control is wired to a 6-step divisor is impossible when the divisor is
    codegen'd from the same `choices` the parameter carries.

    Only genuinely discrete parameters get a divisor:
      - a choice/enum parameter -> len(choices) - 1;
      - an integer control (an exact step of 1 over integer bounds, e.g. a JUCE
        AudioParameterInt / iPlug2 int param) -> max - min.
    A continuous parameter — including one with a fine cosmetic step like 0.1 —
    returns None; its knob is not quantized and has no divisor.
    """
    choices = p.get("choices")
    if choices:
        n = len(choices)
        return n - 1 if n > 1 else None
    rng = p.get("pulp_range") or {}
    step, mn, mx = rng.get("step"), rng.get("min"), rng.get("max")
    try:
        if step is not None and float(step) == 1.0 \
                and float(mn).is_integer() and float(mx).is_integer():
            span = int(round(float(mx) - float(mn)))
            return span if span >= 1 else None
    except (TypeError, ValueError):
        return None
    return None


def _gen_param_bind_grid(ir: dict) -> list[dict]:
    """The authoritative widget-binding table for the import: one entry per
    resolvable parameter carrying exactly what a generated UI binding needs — the
    string key it binds by, the stable id, the range, and (for a discrete
    control) the normalization divisor derived from the source's own option
    count. Emitted once from the shared IR so a JUCE *or* iPlug2 UI scaffold
    reads the divisor from ONE place that cannot disagree with the parameter it
    binds (porting-feedback ask: codegen the bind-grid from the definitions).
    """
    grid: list[dict] = []
    for p in ir.get("parameters", []):
        key = p.get("source_id_string") or p.get("id")
        if key is None or p.get("proposed_pulp_id") is None:
            continue  # unresolved params are surfaced as constructs/TODOs
        rng = p.get("pulp_range") or {}
        entry = {
            "key": key,
            "id": p.get("proposed_pulp_id"),
            "name": p.get("name") or key,
            "min": rng.get("min"),
            "max": rng.get("max"),
            "step": rng.get("step"),
            "discrete_divisor": _param_discrete_divisor(p),
        }
        if p.get("choices"):
            entry["choices"] = list(p.get("choices"))
        grid.append(entry)
    return grid


def _emit_param_registration(p: dict, enum_name: str, id_label: str,
                             confidence_floor: float) -> list[str]:
    """One store.add_parameter({...}) block for a resolvable IR parameter.

    Honesty gate (plan §§7.4/16.3): when the IR's `confidence` for this param is
    below `confidence_floor`, the resolved id/name/range/default came from a
    non-literal / runtime-computed source and is a GUESS. We must not emit it as
    if certain — instead emit a clearly-labelled `// TODO(import)` stub naming
    the low confidence so the porter verifies before trusting it.
    """
    rng = p.get("pulp_range") or {}
    lines: list[str] = []
    curve = p.get("source_curve") or {}
    # The curve contract (see module docstring): every field is ALREADY in Pulp
    # convention — the extractor converted vendor->Pulp. The emitter is dumb.
    shape = curve.get("shape")
    skew = curve.get("skew")
    symmetric = bool(curve.get("symmetric", False))
    centre = curve.get("centre")
    fidelity = curve.get("fidelity", "exact")
    if shape is None:
        # Back-compat for pre-contract IRs that only carried `skew` (Pulp conv.).
        shape = "pow" if (skew is not None and skew != 1.0) else "linear"
    # "shaped" is driven off the shape enum + centre + symmetric, NOT off a
    # sentinel skew of None. An exp curve legitimately has skew=None and MUST
    # still read as shaped (else it silently emits a LINEAR range — the S2 bug).
    shaped = shape != "linear" or symmetric or centre is not None
    # An exp/logarithmic curve has no exact pow-skew representation; emit it via
    # ParamRange::with_centre and keep a PARTIAL/CLOSE diagnostic. Never linear.
    use_centre = shaped and shape == "exp"
    if use_centre:
        fidelity = "close"
    src_id = p.get("source_id_string") or p.get("id")

    conf = p.get("confidence")
    try:
        low_confidence = conf is not None and float(conf) < confidence_floor
    except (TypeError, ValueError):
        low_confidence = False

    if low_confidence:
        # Downgrade to a labelled stub — do NOT emit the guessed concrete value.
        name_hint = _cpp_str(p.get("name") or src_id)
        df = _cpp_float(p.get("default"), 0.0)
        lines.append(f"        // TODO(import): low-confidence ({name_hint}, "
                     f"confidence {conf}) — verify before trusting.")
        lines.append("        //   The source values below were inferred from a "
                     "non-literal / runtime-computed")
        lines.append("        //   construct, so they are a GUESS, not a fact "
                     "(plan §§7.4/16.3). Confirm the")
        lines.append("        //   id/name/range/default against the original "
                     "plugin, then uncomment:")
        lines.append(f"        // store.add_parameter({{")
        lines.append(f"        //     .id = {enum_name},")
        lines.append(f'        //     .name = "{name_hint}",')
        mn_c = _cpp_float(rng.get("min"), 0.0)
        mx_c = _cpp_float(rng.get("max"), 1.0)
        st_c = _cpp_float(rng.get("step"), 0.0)
        lines.append(f"        //     .range = {{{mn_c}, {mx_c}, {df}, {st_c}}},  "
                     f"// GUESSED — verify")
        lines.append("        // });")
        return lines

    lines.append(f"        // source {id_label} id \"{_cpp_str(src_id)}\" "
                 f"(version hint {p.get('source_version_hint')}), "
                 f"confidence {p.get('confidence')}")
    if use_centre:
        # exp / logarithmic: not representable in Pulp's pow-skew family.
        lines.append("        // source curve shape=exp (logarithmic) is NOT exactly "
                     "representable in Pulp's")
        lines.append("        //   pow-skew ParamRange; emitted via "
                     "ParamRange::with_centre(min, max, centre) —")
        lines.append("        //   CLOSE/PARTIAL: endpoints and the normalized midpoint "
                     "match, the interior is")
        lines.append("        //   an approximation (NOT a faithful round-trip). Verify "
                     "against the source, or")
        lines.append("        //   add a true log shape to ParamRange for an EXACT import.")
    elif shaped:
        # Pulp's ParamRange carries the skew/symmetric shape directly. Whether it
        # round-trips depends on the SOURCE fidelity the extractor recorded: only
        # an EXACT curve may claim the float-tolerance round-trip.
        note = ("EXACT: round-trips within float tolerance" if fidelity == "exact"
                else "CLOSE/PARTIAL: approximate — does NOT round-trip exactly; "
                     "verify against the source")
        lines.append(f"        // source curve skew={skew}"
                     f"{' (symmetric)' if symmetric else ''} emitted as a shaped "
                     f"ParamRange ({note})")

    name = _cpp_str(p.get("name") or src_id)
    unit = ""
    # Choice params get a to_string table; numeric params just carry a range.
    choices = p.get("choices")
    mn = _cpp_float(rng.get("min"), 0.0)
    mx = _cpp_float(rng.get("max"), 1.0)
    df = _cpp_float(p.get("default"), 0.0)
    st = _cpp_float(rng.get("step"), 0.0)

    lines.append("        store.add_parameter({")
    lines.append(f"            .id = {enum_name},")
    lines.append(f'            .name = "{name}",')
    lines.append(f'            .unit = "{unit}",')
    if use_centre:
        # ParamRange::with_centre(min, max, centre, default, step) derives the
        # skew that maps the normalized midpoint to `centre`. NEVER a 4-field
        # linear range for an exp curve.
        c = centre
        if c is None:
            # Contract expects the extractor to supply centre. Derive a
            # geometric-mean fallback for a strictly-positive range; otherwise
            # flag loudly and approximate — but still never a silent linear.
            try:
                lo, hi = float(rng.get("min")), float(rng.get("max"))
            except (TypeError, ValueError):
                lo, hi = 0.0, 1.0
            if lo > 0.0 and hi > 0.0:
                c = math.sqrt(lo * hi)
            else:
                lines.append("            // TODO(import): exp/log curve centre was not "
                             "resolved and the range is not")
                lines.append("            //   strictly positive; with_centre() below "
                             "approximates it — VERIFY.")
                c = lo + (hi - lo) * 0.5
        cl = _cpp_float(c, 0.5)
        lines.append(f"            .range = pulp::state::ParamRange::with_centre("
                     f"{mn}, {mx}, {cl}, {df}, {st}),")
    elif shaped:
        # 6-field aggregate: {min, max, default, step, skew, symmetric_skew}.
        sk = _cpp_float(skew if skew is not None else 1.0, 1.0)
        sym = "true" if symmetric else "false"
        lines.append(f"            .range = {{{mn}, {mx}, {df}, {st}, {sk}, {sym}}},")
    else:
        lines.append(f"            .range = {{{mn}, {mx}, {df}, {st}}},")
    divisor = _param_discrete_divisor(p)
    if divisor is not None:
        # Authoritative, codegen'd from this parameter's own definition: a UI
        # binding maps discrete position i -> i/divisor in [0,1]. Generating it
        # here (not hand-writing it in the UI) is what stops a control's step
        # count from disagreeing with the parameter — see migration_status.json's
        # bind_grid for the machine-readable table a UI scaffold consumes.
        lines.append(f"            // bind-grid: discrete control, normalized "
                     f"position = index / {divisor} (divisor codegen'd from the "
                     f"source's option count)")
    if choices:
        # to_string table for a discrete choice param.
        labels = ", ".join(f'"{_cpp_str(c)}"' for c in choices)
        lines.append("            .to_string = [](float v) -> std::string {")
        lines.append(f"                static const char* kChoices[] = {{{labels}}};")
        lines.append("                int idx = static_cast<int>(v + 0.5f);")
        lines.append("                if (idx < 0) idx = 0;")
        lines.append(f"                if (idx > {len(choices) - 1}) idx = {len(choices) - 1};")
        lines.append("                return kChoices[idx];")
        lines.append("            },")
    lines.append("        });")
    return lines


# --- DSP body emission ------------------------------------------------------

def _emit_process_body(ir: dict, copied_cores: list[str],
                       sample_type: str = "float") -> list[str]:
    """The process() body, chosen by the IR's dsp.classification.

    portable-core: copy-in -> copy-out, with a note pointing at the verbatim
                   user core files we copied next door (the real wiring is a
                   TODO — we don't fabricate the buffer-accessor glue here).
    instrument / labelled-silence: clear the output (honest silence).
    everything else (effect): pass-through (copy in -> out).
    """
    dsp = ir.get("dsp", {})
    cls = dsp.get("classification", "stub")
    category = (ir.get("metadata", {}) or {}).get("pulp_category", "Effect")
    is_instrument = category == "Instrument"
    silence = is_instrument or cls == "labelled-silence"
    zero = _zero_literal(sample_type)

    L: list[str] = []
    L.append(f"        // TODO(import): migrate DSP — {cls}")
    for d in dsp.get("diagnostics", []):
        L.append(f"        // {d}")

    # When the IR carries a resolved DSP mapping table (framework-bound-mappable
    # bodies), emit a clearly-commented core/signal scaffold per mapped symbol
    # instead of pretending the pass-through stub is the migration. This is a
    # comment-guided scaffold, NOT a claim of correctness — the golden-match
    # caveat from the equivalence level is carried through verbatim.
    mappings = dsp.get("mappings", [])
    if mappings:
        L.append("        // ---- DSP mapping scaffold (plan §16.3) "
                 "----------------------------")
        L.append("        // Each source symbol below was resolved to a Pulp "
                 "core/signal target with")
        L.append("        // an equivalence level. These are NOT verified ports — "
                 "wire them into the")
        L.append("        // signal flow and golden-match before trusting.")
        for m in mappings:
            src = m.get("source_symbol", "?")
            pulp = m.get("pulp_primitive")
            eq = m.get("equivalence", "unsupported")
            note = (m.get("notes") or "").strip()
            if pulp is None or eq == "unsupported":
                L.append(f"        // TODO(import): UNSUPPORTED {src} — no "
                         f"core/signal equivalent; manual port / feature-gap.")
                if note:
                    L.append(f"        //   note: {note}")
            else:
                # Member-declaration hint + the load-bearing equivalence caveat.
                inst = _member_token(src)
                L.append(f"        // TODO(import): mapped {src} -> {pulp} "
                         f"({eq}); golden-match within tolerance before trusting.")
                if note:
                    L.append(f"        //   semantic caveat: {note}")
                L.append(f"        //   suggested member: {pulp} {inst}_;  "
                         f"// declare + prepare() in prepare(); call its process() here")
        L.append("        // ------------------------------------------------"
                 "----------------------------")

    if copied_cores:
        L.append("        // Framework-free core(s) copied verbatim into this "
                 "scaffold (provenance: copied-user-file):")
        for c in copied_cores:
            L.append(f"        //   - {c}")
        L.append("        // TODO(import): wire the copied core(s) into this "
                 "process() once buffer-accessor glue is migrated.")

    if silence:
        L.append("        // Labelled silence: instrument/voice render path not "
                 "yet migrated.")
        L.append("        (void)audio_input;")
        L.append("        for (std::size_t ch = 0; ch < audio_output.num_channels(); ++ch) {")
        L.append("            auto out = audio_output.channel(ch);")
        L.append("            for (std::size_t i = 0; i < audio_output.num_samples(); ++i)")
        L.append(f"                out[i] = {zero};")
        L.append("        }")
    else:
        L.append("        // Pass-through scaffold: copy input to output verbatim.")
        L.append("        const std::size_t chans = "
                 "std::min(audio_output.num_channels(), audio_input.num_channels());")
        L.append("        for (std::size_t ch = 0; ch < chans; ++ch) {")
        L.append("            auto out = audio_output.channel(ch);")
        L.append("            auto in = audio_input.channel(ch);")
        L.append("            for (std::size_t i = 0; i < audio_output.num_samples(); ++i)")
        L.append("                out[i] = in[i];")
        L.append("        }")
        L.append("        // Clear any output channels with no matching input.")
        L.append("        for (std::size_t ch = chans; ch < audio_output.num_channels(); ++ch) {")
        L.append("            auto out = audio_output.channel(ch);")
        L.append("            for (std::size_t i = 0; i < audio_output.num_samples(); ++i)")
        L.append(f"                out[i] = {zero};")
        L.append("        }")
    return L


# --- descriptor emission ----------------------------------------------------

def _emit_buses(buses: list[dict]) -> str:
    if not buses:
        return "{}"
    items = []
    for b in buses:
        name = _cpp_str(b.get("name") or "Bus")
        chans = b.get("channels")
        chans = 2 if chans is None else int(chans)
        opt = "true" if b.get("optional") else "false"
        items.append(f'{{"{name}", {chans}, {opt}}}')
    return "{" + ", ".join(items) + "}"


def _emit_descriptor(ir: dict, class_name: str) -> list[str]:
    md = ir.get("metadata", {})
    midi = ir.get("midi", {})
    buses = ir.get("buses", {})
    category = CATEGORY_MAP.get(md.get("pulp_category", "Effect"), "Effect")

    name = _cpp_str((md.get("name") or class_name) + " Migration Scaffold")
    L: list[str] = []
    L.append("    pulp::format::PluginDescriptor descriptor() const override {")
    L.append("        return {")
    L.append(f'            .name = "{name}",')
    L.append(f'            .manufacturer = "{_cpp_str(md.get("manufacturer"))}",')
    L.append(f'            .bundle_id = "{_cpp_str(md.get("bundle_id"))}",')
    ver = md.get("version") or "0.0.0"
    L.append(f'            .version = "{_cpp_str(ver)}",')
    L.append(f"            .category = pulp::format::PluginCategory::{category},")
    in_buses = _emit_buses(buses.get("inputs", []))
    out_buses = _emit_buses(buses.get("outputs", []))
    if in_buses != "{}":
        L.append(f"            .input_buses = {in_buses},")
    else:
        L.append("            .input_buses = {},  // no input buses (e.g. instrument)")
    if out_buses != "{}":
        L.append(f"            .output_buses = {out_buses},")
    L.append(f"            .accepts_midi = {'true' if midi.get('accepts_midi') else 'false'},")
    L.append(f"            .produces_midi = {'true' if midi.get('produces_midi') else 'false'},")
    if md.get("vendor_url"):
        L.append(f'            .vendor_url = "{_cpp_str(md.get("vendor_url"))}",')
    if md.get("vendor_email"):
        L.append(f'            .vendor_email = "{_cpp_str(md.get("vendor_email"))}",')
    if _source_native_f64(ir):
        L.append("            .supports_f64_audio = true,")
    L.append("        };")
    L.append("    }")
    return L


# --- UI classification ------------------------------------------------------

def _is_webview_ui(ir: dict) -> bool:
    """True when the IR's vendor-neutral UI classification is a WebView.

    The importers' INSPECT step sets `ui.kind` to `"webview"` when the source
    plugin's editor is a WebView (framework-specific markers — JUCE's
    WebBrowserComponent, iPlug2's IWebView/IGraphicsWebView — live only in the
    per-importer extractors). The shared emit core branches on this DATA alone
    and names no framework. Anything else (native / custom-paint / stock /
    none / unknown) takes the existing native scaffold path unchanged.
    """
    return (ir.get("ui", {}) or {}).get("kind") == "webview"


# --- file generators --------------------------------------------------------

def _gen_header(ir: dict, class_name: str, factory_name: str, namespace: str,
                params: list[tuple[str, int, str]], header_name: str,
                id_label: str, tool_label: str) -> str:
    webview = _is_webview_ui(ir)
    L: list[str] = []
    L.append("#pragma once")
    L.append("")
    L.append(f"// {ir.get('metadata', {}).get('name', class_name)} — Pulp migration scaffold")
    L.append(f"// Generated by the {tool_label}importer EMIT step (spike). This is a")
    L.append("// BUILDING starting point, not a finished plugin. Search for")
    L.append("// `TODO(import)` to find everything that still needs migration.")
    L.append("")
    L.append("#include <pulp/format/processor.hpp>")
    if webview:
        # WebView editor scaffold needs the view + WebView host headers.
        L.append("#include <pulp/view/plugin_view_host.hpp>")
        L.append("#include <pulp/view/view.hpp>")
        L.append("#include <pulp/view/web_view.hpp>")
        L.append("#include <memory>")
    L.append("#include <algorithm>")
    L.append("#include <cstdint>")
    L.append("#include <string>")
    L.append("")
    L.append(f"namespace {namespace} {{")
    L.append("")
    if params:
        L.append("// Stable parameter ids — the IR's proposed_pulp_id (FNV-1a of the")
        L.append(f"// source {id_label} id string), so preset/automation identity survives import.")
        L.append("enum ParamIDs : pulp::state::ParamID {")
        for enum, pid, sid in params:
            L.append(f"    {enum} = {pid}u,  // \"{_cpp_str(sid)}\"")
        L.append("};")
        L.append("")
    L.append(f"class {class_name} : public pulp::format::Processor {{")
    L.append("public:")
    L.extend(_emit_descriptor(ir, class_name))
    L.append("")
    L.append("    void define_parameters(pulp::state::StateStore& store) override;")
    L.append("    void prepare(const pulp::format::PrepareContext& context) override;")
    L.append("    void process(")
    L.append("        pulp::audio::BufferView<float>& audio_output,")
    L.append("        const pulp::audio::BufferView<const float>& audio_input,")
    L.append("        pulp::midi::MidiBuffer& midi_in,")
    L.append("        pulp::midi::MidiBuffer& midi_out,")
    L.append("        const pulp::format::ProcessContext& context) override;")
    if _source_native_f64(ir):
        L.append("")
        L.append("    void process_f64(")
        L.append("        pulp::audio::BufferView<double>& audio_output,")
        L.append("        const pulp::audio::BufferView<const double>& audio_input,")
        L.append("        pulp::midi::MidiBuffer& midi_in,")
        L.append("        pulp::midi::MidiBuffer& midi_out,")
        L.append("        const pulp::format::ProcessContext& context) override;")
    L.append("")
    L.append("    std::vector<uint8_t> serialize_plugin_state() const override;")
    L.append("    bool deserialize_plugin_state(std::span<const uint8_t> data) override;")
    if webview:
        L.append("")
        L.append("    // WebView editor: the source plugin's UI was a WebView, so the")
        L.append("    // imported editor hosts a Pulp WebViewPanel pointing at the")
        L.append("    // embedded `ui/` asset directory. See PluginProcessor.cpp.")
        L.append("    pulp::format::ViewSize view_size() const override;")
        L.append("    std::unique_ptr<pulp::view::View> create_view() override;")
        L.append("    void on_view_opened(pulp::view::View& root) override;")
        L.append("    void on_view_resized(pulp::view::View& root, uint32_t w, uint32_t h) override;")
        L.append("    void on_view_closed(pulp::view::View& root) override;")
    L.append("};")
    L.append("")
    L.append(f"std::unique_ptr<pulp::format::Processor> create_{factory_name}();")
    L.append("")
    L.append(f"}} // namespace {namespace}")
    L.append("")
    return "\n".join(L)


def _emit_state_skeleton(class_name: str, opaque: bool) -> list[str]:
    """The serialize/deserialize_plugin_state bodies — a working param-state
    save/restore skeleton (the APVTS / IParam -> Pulp state bridge).

    The skeleton round-trips the StateStore parameter payload via
    `state().serialize()` / `state().deserialize()` — a real, compiling
    round-trip the porter can run and trust — and labels any opaque/binary
    session blob the source plugin owned as a `// TODO(import)` that must be
    re-modelled by hand (binary DAW-session compatibility with the original is
    not supported, plan §7.4).
    """
    L: list[str] = []
    L.append(f"std::vector<uint8_t> {class_name}::serialize_plugin_state() const {{")
    L.append("    // Parameter state: the registered StateStore parameters are the")
    L.append("    // imported APVTS / IParam values. The format adapter already")
    L.append("    // persists them with the host session, so returning {} here is")
    L.append("    // sufficient for parameter recall. We additionally serialize the")
    L.append("    // StateStore payload explicitly so this hook is a self-contained,")
    L.append("    // round-trippable snapshot of the parameter state (verify with a")
    L.append("    // save -> deserialize_plugin_state -> compare round-trip test).")
    L.append("    std::vector<uint8_t> blob = state().serialize();")
    if opaque:
        L.append("    // TODO(import): the source plugin also wrote a hand-rolled binary blob")
        L.append("    // (custom getStateInformation / IByteChunk serializer).")
        L.append("    // Binary DAW-session compatibility with the original plugin is NOT "
                 "supported")
        L.append("    // (plan §7.4) — append your re-modelled plugin-owned state to `blob`")
        L.append("    // here once it has been ported.")
    else:
        L.append("    // TODO(import): if this plugin owns extra non-parameter state")
        L.append("    // (sample slots, learned curves, ...), append it to `blob` here.")
    L.append("    return blob;")
    L.append("}")
    L.append("")
    L.append(f"bool {class_name}::deserialize_plugin_state("
             "std::span<const uint8_t> data) {")
    L.append("    // Empty payload => legacy/parameter-only state; nothing to restore")
    L.append("    // beyond the adapter's automatic StateStore recall.")
    L.append("    if (data.empty())")
    L.append("        return true;")
    L.append("    // Restore the parameter snapshot written by serialize_plugin_state().")
    L.append("    if (!state().deserialize(data))")
    L.append("        return false;")
    if opaque:
        L.append("    // TODO(import): parse and restore the re-modelled plugin-owned state")
        L.append("    // appended above. No binary-compatible restore of the ORIGINAL")
        L.append("    // plugin's session blob is provided — see serialize_plugin_state.")
    L.append("    return true;")
    L.append("}")
    return L


# --- WebView editor emission ------------------------------------------------

# The default web origin root and entry filename for the embedded WebView. The
# importer cannot extract bundled binary web resources, so the scaffold ships a
# placeholder `ui/index.html` and serves the `ui/` directory through Pulp's
# directory-backed WebView resource fetcher. The porter replaces the placeholder
# with the real HTML/JS/CSS payload.
_WEBVIEW_UI_DIR = "ui"
_WEBVIEW_DEFAULT_ENTRY = "index.html"


def _webview_entry(ir: dict) -> str:
    """The HTML entry filename for the embedded WebView.

    Honesty rule (plan §16.3): use a literal filename ONLY when the INSPECT step
    statically resolved one from a string literal in the source
    (`ui.asset_hints.html_entry`). Otherwise default to `index.html` and leave a
    TODO — never guess a path the source did not contain.
    """
    hint = (((ir.get("ui", {}) or {}).get("asset_hints") or {}).get("html_entry"))
    if isinstance(hint, str) and hint.strip():
        # Strip any directory portion — the scaffold serves the ui/ dir as root.
        return hint.strip().replace("\\", "/").split("/")[-1]
    return _WEBVIEW_DEFAULT_ENTRY


def _emit_webview_view(ir: dict, class_name: str,
                       params: list[tuple[str, int, str]]) -> list[str]:
    """The WebView editor scaffold: a `create_view()` that hosts a Pulp
    WebViewPanel pointing at the embedded `ui/` asset dir, plus a native
    JS<->param bridge shim that maps the source's bridge onto Pulp's WebView
    bridge (param-by-string-key).

    Mirrors the real Pulp pattern in `examples/webview-plugin/` (PluginViewHost +
    attach_native_child_view + WebViewPanel). The bound page assets are NOT
    extractable statically, so the scaffold ships a placeholder ui/ dir and a
    loud `// TODO(import)` to copy the real payload in.
    """
    ui = ir.get("ui", {}) or {}
    hints = ui.get("asset_hints") or {}
    entry = _webview_entry(ir)
    has_entry_hint = bool(isinstance(hints.get("html_entry"), str)
                          and hints.get("html_entry"))

    L: list[str] = []
    L.append("// ---- WebView editor scaffold (UI kind: webview) "
             "------------------------")
    L.append("// The source plugin's editor was a WebView. This scaffold hosts a Pulp")
    L.append("// WebViewPanel inside the plugin editor subtree, serving the embedded")
    L.append(f"// `{_WEBVIEW_UI_DIR}/` directory as the web origin (same pattern as the")
    L.append("// Pulp `examples/webview-plugin/` reference).")
    L.append("//")
    L.append("// TODO(import): copy your WebView assets (HTML/JS/CSS) into "
             f"{_WEBVIEW_UI_DIR}/ —")
    L.append("//   the importer cannot extract bundled binary resources. A placeholder")
    L.append(f"//   {_WEBVIEW_UI_DIR}/{_WEBVIEW_DEFAULT_ENTRY} ships so the scaffold builds and "
             "renders something.")
    if has_entry_hint:
        L.append(f"// INSPECT found a literal HTML entry reference: \"{_cpp_str(entry)}\" "
                 "— wire your")
        L.append("//   copied assets so that file is the entry point.")
    else:
        L.append("// TODO(import): INSPECT could not statically resolve the HTML entry "
                 "filename")
        L.append(f"//   (no string literal found); defaulting to {_WEBVIEW_DEFAULT_ENTRY}. "
                 "Confirm + adjust.")
    L.append("")

    L.append("class WebViewEditorPane final : public pulp::view::View {")
    L.append("public:")
    L.append("    explicit WebViewEditorPane(pulp::state::StateStore& store) "
             ": store_(store) {")
    L.append("        pulp::view::WebViewOptions options;")
    L.append("        options.transparent_background = true;")
    L.append("        // Serve the embedded ui/ directory as the web origin. In a "
             "packaged")
    L.append("        // build, point this at the installed resource directory.")
    L.append(f'        options.custom_scheme_uri = "pulp://app/";')
    L.append("        options.fetch_resource =")
    L.append("            pulp::view::make_webview_directory_resource_fetcher(")
    L.append(f'                ui_root(), "{_cpp_str(entry)}");')
    L.append("        panel_ = pulp::view::WebViewPanel::create(options);")
    L.append("        if (!panel_) return;  // backend unavailable; native fallback")
    L.append("        install_bridge();")
    L.append("        panel_->set_ready_handler([this] {")
    L.append(f'            if (panel_) panel_->navigate("pulp://app/{_cpp_str(entry)}");')
    L.append("            push_all_params();")
    L.append("        });")
    L.append("    }")
    L.append("")
    L.append("    ~WebViewEditorPane() override { detach_if_needed(); }")
    L.append("")
    L.extend(_emit_webview_bridge(params))
    L.append("")
    L.append("    void attach_if_needed() {")
    L.append("        auto* host = plugin_view_host();")
    L.append("        if (attached_ || !host || !panel_ || !panel_->native_handle()) "
             "return;")
    L.append("        const auto size = host->get_size();")
    L.append("        attached_ = host->attach_native_child_view(")
    L.append("            panel_->native_handle(), 0.0f, 0.0f,")
    L.append("            static_cast<float>(size.width), "
             "static_cast<float>(size.height));")
    L.append("        if (attached_) sync_to_host();")
    L.append("    }")
    L.append("")
    L.append("    void sync_to_host() {")
    L.append("        auto* host = plugin_view_host();")
    L.append("        if (!attached_ || !host || !panel_ || !panel_->native_handle()) "
             "return;")
    L.append("        const auto size = host->get_size();")
    L.append("        host->set_native_child_view_bounds(")
    L.append("            panel_->native_handle(), 0.0f, 0.0f,")
    L.append("            static_cast<float>(size.width), "
             "static_cast<float>(size.height));")
    L.append("    }")
    L.append("")
    L.append("    void detach_if_needed() {")
    L.append("        auto* host = plugin_view_host();")
    L.append("        if (!attached_ || !host || !panel_ || !panel_->native_handle()) {")
    L.append("            attached_ = false;")
    L.append("            return;")
    L.append("        }")
    L.append("        host->detach_native_child_view(panel_->native_handle());")
    L.append("        attached_ = false;")
    L.append("    }")
    L.append("")
    L.append("private:")
    L.append("    static std::filesystem::path ui_root() {")
    L.append("        // TODO(import): resolve this to the installed ui/ resource dir for")
    L.append(f"        //   packaged builds. During development it is the scaffold's "
             f"{_WEBVIEW_UI_DIR}/.")
    L.append(f'        return std::filesystem::path("{_WEBVIEW_UI_DIR}");')
    L.append("    }")
    L.append("")
    L.append("    pulp::state::StateStore& store_;")
    L.append("    std::unique_ptr<pulp::view::WebViewPanel> panel_;")
    L.append("    bool attached_ = false;")
    L.append("};")
    L.append("")
    L.append("class WebViewEditorRoot final : public pulp::view::View {")
    L.append("public:")
    L.append("    explicit WebViewEditorRoot(pulp::state::StateStore& store) {")
    L.append("        auto pane = std::make_unique<WebViewEditorPane>(store);")
    L.append("        pane_ = pane.get();")
    L.append("        add_child(std::move(pane));")
    L.append("    }")
    L.append("    WebViewEditorPane& pane() { return *pane_; }")
    L.append("    void on_resized() override {")
    L.append("        if (pane_) pane_->set_bounds({0, 0, bounds().width, "
             "bounds().height});")
    L.append("    }")
    L.append("private:")
    L.append("    WebViewEditorPane* pane_ = nullptr;")
    L.append("};")
    L.append("// --------------------------------------------------------------"
             "------------")
    return L


def _emit_webview_bridge(params: list[tuple[str, int, str]]) -> list[str]:
    """The native JS<->param bridge shim: maps the source plugin's web param
    bridge onto Pulp's WebView message bridge, param-by-string-key.

    The contract mirrors Pulp's existing param-bridge: the page posts
    `{type:"param", payload:{key, value}}` to set a parameter, and native pushes
    `{type:"param", payload:{key, value}}` back when a value changes. The source
    framework's bespoke relay/bridge handlers (JUCE WebSliderRelay, iPlug2
    SendJSONFromDelegate, ...) are mapped onto this contract by the SAME string
    key the parameter carries — that key is the source parameter id string the
    IR preserved, so the JS side keeps using its original names.
    """
    L: list[str] = []
    L.append("    // Native <-> JS parameter bridge (param-by-string-key). The page")
    L.append("    // sets a parameter via window.pulp.postMessage(\"param\", "
             "{key, value}),")
    L.append("    // and native pushes value changes back the same way. The keys are the")
    L.append("    // source parameter id strings the import preserved.")
    L.append("    void install_bridge() {")
    L.append("        if (!panel_) return;")
    L.append("        panel_->set_message_handler(")
    L.append("            [this](const pulp::view::WebViewMessage& msg) -> std::string {")
    L.append('            if (msg.type == "param") {')
    L.append("                // TODO(import): parse {key, value} from msg.payload_json and")
    L.append("                //   route to the StateStore parameter with that key, e.g.:")
    L.append("                //     store_.set_parameter(key_to_id(key), value);")
    L.append("                return handle_param_message(msg.payload_json);")
    L.append("            }")
    L.append("            // TODO(import): the source plugin may have bespoke message")
    L.append("            //   handlers (custom JS<->native calls beyond parameters).")
    L.append("            //   Map each one onto a msg.type case here.")
    L.append('            return R"({"ok":true})";')
    L.append("        });")
    L.append("    }")
    L.append("")
    L.append("    // Map a source parameter key string to its stable Pulp ParamID.")
    L.append("    static bool key_to_id(const std::string& key, "
             "pulp::state::ParamID& out) {")
    if params:
        L.append("        // Keys are the source parameter id strings preserved by the import.")
        for enum, _pid, sid in params:
            L.append(f'        if (key == "{_cpp_str(sid)}") {{ out = {enum}; return true; }}')
    else:
        L.append("        (void)key; (void)out;")
        L.append("        // No statically-resolved parameters to map.")
    L.append("        return false;  // unknown key — bespoke / unresolved")
    L.append("    }")
    L.append("")
    L.append("    std::string handle_param_message(const std::string& payload_json) {")
    L.append("        // TODO(import): decode payload_json (a {\"key\":..,\"value\":..}")
    L.append("        //   object), look the key up with key_to_id(), and apply it via")
    L.append("        //   store_.set_parameter(id, value). Returning ok keeps the bridge")
    L.append("        //   responsive while the decode is wired up.")
    L.append("        (void)payload_json;")
    L.append('        return R"({"ok":true})";')
    L.append("    }")
    L.append("")
    L.append("    // Push every current parameter value to the page on load, so the web")
    L.append("    // UI initialises to the live plugin state.")
    L.append("    void push_all_params() {")
    L.append("        if (!panel_) return;")
    if params:
        for enum, _pid, sid in params:
            L.append("        // TODO(import): post the live value for "
                     f'"{_cpp_str(sid)}" to the page, e.g.:')
            L.append("        //   post_param(\"" + _cpp_str(sid) + f'", store_.get_parameter({enum}));')
    else:
        L.append("        // No statically-resolved parameters to push.")
    L.append("    }")
    L.append("")
    L.append("    void post_param(const std::string& key, float value) {")
    L.append("        if (!panel_) return;")
    L.append("        pulp::view::WebViewMessage m;")
    L.append('        m.type = "param";')
    L.append("        // TODO(import): serialize {key, value} as JSON for the payload.")
    L.append("        (void)key; (void)value;")
    L.append('        m.payload_json = "null";')
    L.append("        panel_->post_message(m);")
    L.append("    }")
    return L


def _emit_webview_view_hooks(class_name: str) -> list[str]:
    """The Processor view-lifecycle hook bodies for the WebView editor."""
    L: list[str] = []
    L.append(f"pulp::format::ViewSize {class_name}::view_size() const {{")
    L.append("    // TODO(import): match the source editor's natural / min / max size.")
    L.append("    return {720, 440, 480, 320, 1280, 800};")
    L.append("}")
    L.append("")
    L.append(f"std::unique_ptr<pulp::view::View> {class_name}::create_view() {{")
    L.append("    return std::make_unique<WebViewEditorRoot>(state());")
    L.append("}")
    L.append("")
    L.append(f"void {class_name}::on_view_opened(pulp::view::View& root) {{")
    L.append("    static_cast<WebViewEditorRoot&>(root).pane().attach_if_needed();")
    L.append("}")
    L.append("")
    L.append(f"void {class_name}::on_view_resized(pulp::view::View& root, "
             "uint32_t, uint32_t) {")
    L.append("    static_cast<WebViewEditorRoot&>(root).pane().sync_to_host();")
    L.append("}")
    L.append("")
    L.append(f"void {class_name}::on_view_closed(pulp::view::View& root) {{")
    L.append("    static_cast<WebViewEditorRoot&>(root).pane().detach_if_needed();")
    L.append("}")
    return L


def _gen_webview_index_html(ir: dict) -> str:
    """The placeholder ui/index.html the WebView serves until the real assets
    are copied in. Self-contained, dark-themed, and exercises the bridge so the
    porter can see whether native<->JS is wired before swapping in real assets."""
    name = _cpp_str((ir.get("metadata", {}) or {}).get("name") or "Imported Plugin")
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "  <head>\n"
        "    <meta charset=\"utf-8\">\n"
        "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "    <title>" + name + " — imported WebView UI (placeholder)</title>\n"
        "    <style>\n"
        "      :root { color-scheme: dark; font-family: system-ui, sans-serif; }\n"
        "      body { margin: 0; min-height: 100vh; display: grid; "
        "place-items: center;\n"
        "             background: #0f172a; color: #e2e8f0; }\n"
        "      .card { max-width: 480px; padding: 24px; border-radius: 16px;\n"
        "              border: 1px solid #334155; background: #1e293b; }\n"
        "      code { color: #7dd3fc; }\n"
        "    </style>\n"
        "  </head>\n"
        "  <body>\n"
        "    <main class=\"card\">\n"
        "      <h1>" + name + "</h1>\n"
        "      <p>This is a <strong>placeholder</strong> WebView UI emitted by the "
        "Pulp importer.</p>\n"
        "      <p>TODO(import): replace the files in <code>ui/</code> with your "
        "original WebView assets (HTML/JS/CSS). The importer cannot extract "
        "bundled binary resources.</p>\n"
        "      <p id=\"status\">bridge: checking…</p>\n"
        "    </main>\n"
        "    <script>\n"
        "      const status = document.getElementById('status');\n"
        "      // Pulp installs window.pulp on the native host. The bridge maps\n"
        "      // parameters by string key: post {type:'param', {key, value}} to set,\n"
        "      // and listen for the same shape pushed from native.\n"
        "      if (window.pulp) {\n"
        "        status.textContent = 'bridge: available';\n"
        "        window.pulp.on('param', (p) => {\n"
        "          // TODO(import): update your UI for the changed parameter p.key.\n"
        "        });\n"
        "      } else {\n"
        "        status.textContent = 'bridge: unavailable (browser preview)';\n"
        "      }\n"
        "    </script>\n"
        "  </body>\n"
        "</html>\n"
    )


def _gen_source(ir: dict, class_name: str, factory_name: str, namespace: str,
                params: list[tuple[str, int, str]], header_name: str,
                copied_cores: list[str], id_label: str,
                confidence_floor: float) -> str:
    state = ir.get("state_model", {})
    opaque = state.get("classification") == "opaque-custom"
    ir_params = ir.get("parameters", [])
    # map enum_name -> ir param dict by source id
    by_sid = {(p.get("source_id_string") or p.get("id")): p for p in ir_params}

    webview = _is_webview_ui(ir)

    L: list[str] = []
    L.append(f'#include "{header_name}"')
    if webview:
        L.append("")
        L.append("#include <filesystem>")
    L.append("")
    L.append(f"namespace {namespace} {{")
    L.append("")

    # WebView editor classes (anonymous namespace) — only when the source UI was
    # a WebView. The Processor's create_view()/lifecycle hook bodies below
    # reference these.
    if webview:
        L.append("namespace {")
        L.append("")
        L.extend(_emit_webview_view(ir, class_name, params))
        L.append("")
        L.append("}  // namespace")
        L.append("")

    # define_parameters
    L.append(f"void {class_name}::define_parameters("
             "pulp::state::StateStore& store) {")
    if params:
        for enum, _pid, sid in params:
            p = by_sid.get(sid, {})
            L.extend(_emit_param_registration(p, enum, id_label,
                                              confidence_floor))
    else:
        L.append("    // No statically-resolvable parameters were found.")
        L.append("    (void)store;")
    # unresolved constructs -> TODO
    for c in ir.get("constructs", []):
        ref = c.get("source_ref", {})
        L.append(f"    // TODO(import): unresolved parameter construct "
                 f"({c.get('construct_type')}) at {ref.get('file')}:{ref.get('line')} "
                 f"— {c.get('enumeration_status')}; cardinality not inferred. "
                 f"Enumerate and register these parameters manually.")
    L.append("}")
    L.append("")

    # prepare
    L.append(f"void {class_name}::prepare("
             "const pulp::format::PrepareContext& context) {")
    L.append("    (void)context;")
    L.append("    // TODO(import): allocate DSP resources at context.sample_rate / "
             "context.max_buffer_size.")
    L.append("}")
    L.append("")

    # process
    L.append(f"void {class_name}::process(")
    L.append("    pulp::audio::BufferView<float>& audio_output,")
    L.append("    const pulp::audio::BufferView<const float>& audio_input,")
    L.append("    pulp::midi::MidiBuffer& midi_in,")
    L.append("    pulp::midi::MidiBuffer& midi_out,")
    L.append("    const pulp::format::ProcessContext& context) {")
    L.append("    (void)midi_in;")
    L.append("    (void)midi_out;")
    L.append("    (void)context;")
    L.extend(_emit_process_body(ir, copied_cores))
    L.append("}")
    L.append("")

    if _source_native_f64(ir):
        L.append(f"void {class_name}::process_f64(")
        L.append("    pulp::audio::BufferView<double>& audio_output,")
        L.append("    const pulp::audio::BufferView<const double>& audio_input,")
        L.append("    pulp::midi::MidiBuffer& midi_in,")
        L.append("    pulp::midi::MidiBuffer& midi_out,")
        L.append("    const pulp::format::ProcessContext& context) {")
        L.append("    (void)midi_in;")
        L.append("    (void)midi_out;")
        L.append("    (void)context;")
        L.append("    // Source plugin advertised native double-precision audio. Keep this")
        L.append("    // override wired so Pulp hosts do not fall back through the f32 path.")
        L.extend(_emit_process_body(ir, copied_cores, sample_type="double"))
        L.append("}")
        L.append("")

    # serialize / deserialize — param-state save/restore skeleton.
    #
    # This is the source framework's parameter-state bridge (APVTS /
    # AudioProcessorValueTreeState on one side, IPlugAPIBase / IParam on the
    # other) mapped onto Pulp's state model. In Pulp the registered StateStore
    # parameters already round-trip through the format adapter automatically, so
    # the parameter half of the bridge is DONE the moment define_parameters()
    # ran. The hooks below are the *extra* plugin-owned-state half: a working,
    # compiling skeleton that round-trips the StateStore param payload itself
    # (honest, verifiable) plus a TODO for any opaque/binary session state the
    # adapter cannot model for you.
    L.extend(_emit_state_skeleton(class_name, opaque))
    L.append("")

    # WebView editor lifecycle hooks (only when the source UI was a WebView).
    if webview:
        L.extend(_emit_webview_view_hooks(class_name))
        L.append("")

    # factory
    L.append(f"std::unique_ptr<pulp::format::Processor> create_{factory_name}() {{")
    L.append(f"    return std::make_unique<{class_name}>();")
    L.append("}")
    L.append("")
    L.append(f"}} // namespace {namespace}")
    L.append("")
    return "\n".join(L)


def _gen_clap_entry(factory_name: str, namespace: str, header_name: str) -> str:
    return (
        f'// CLAP entry point for the imported scaffold.\n'
        f'#include "{header_name}"\n'
        f"#include <pulp/format/clap_entry.hpp>\n"
        f"\n"
        f"PULP_CLAP_PLUGIN({namespace}::create_{factory_name})\n"
    )


def _gen_cmake(ir: dict, target: str, namespace: str, factory_name: str,
               formats: list[str], sources: list[str], lower: str,
               sdk_version: str, tool_label: str) -> str:
    md = ir.get("metadata", {})
    midi = ir.get("midi", {})
    category = CATEGORY_MAP.get(md.get("pulp_category", "Effect"), "Effect")
    name = md.get("name") or target
    ver = md.get("version") or "0.0.0"

    src_list = "\n".join(f"        {s}" for s in sources)
    fmt_list = " ".join(formats)
    midi_flags = []
    if midi.get("accepts_midi"):
        midi_flags.append("    ACCEPTS_MIDI")
    if midi.get("produces_midi"):
        midi_flags.append("    PRODUCES_MIDI")
    midi_block = ("\n" + "\n".join(midi_flags)) if midi_flags else ""

    L: list[str] = []
    L.append(f"# {name} — Pulp migration scaffold (generated by the {tool_label}importer EMIT step)")
    L.append("#")
    L.append("# This builds against an installed Pulp SDK. It is a BUILDING scaffold,")
    L.append("# not a finished plugin — search the sources for `TODO(import)`.")
    L.append("")
    L.append("cmake_minimum_required(VERSION 3.24)")
    L.append(f"project({target} VERSION {ver} LANGUAGES CXX)")
    L.append("")
    L.append("set(CMAKE_CXX_STANDARD 20)")
    L.append("set(CMAKE_CXX_STANDARD_REQUIRED ON)")
    L.append("")
    L.extend(_gen_integration_cmake_prelude(ir))
    L.append(f"find_package(Pulp {sdk_version} REQUIRED)")
    L.append("")
    L.append(f"pulp_add_plugin({target}")
    L.append(f"    FORMATS         {fmt_list}")
    L.append(f'    PLUGIN_NAME     "{name}"')
    L.append(f'    BUNDLE_ID       "{md.get("bundle_id") or ("com.imported." + lower)}"')
    L.append(f'    MANUFACTURER    "{md.get("manufacturer") or "Unknown"}"')
    L.append(f'    VERSION         "{ver}"')
    L.append(f"    CATEGORY        {category}")
    if md.get("plugin_code"):
        L.append(f'    PLUGIN_CODE     "{md.get("plugin_code")}"')
    if md.get("manufacturer_code"):
        L.append(f'    MANUFACTURER_CODE "{md.get("manufacturer_code")}"')
    L.append("    SOURCES")
    L.append(src_list)
    L.append(f"{midi_block}")
    L.append(")")
    L.append("")
    L.extend(_gen_integration_target_links(ir, target))
    return "\n".join(L)


def _gen_migration_status(ir: dict, copied_cores: list[str],
                          emit_tool: str) -> dict:
    md = ir.get("metadata", {})
    dsp = ir.get("dsp", {})
    state = ir.get("state_model", {})
    integration_reqs = _integration_requirements(ir)
    todos: list[str] = []

    # Skewed / symmetric params emit an EXACT shaped ParamRange that round-trips
    # through Pulp's normalize/denormalize — not a migration blocker. But a curve
    # the extractor marked CLOSE (fidelity != "exact", e.g. an exp/log curve
    # emitted via ParamRange::with_centre) is only an APPROXIMATION and MUST stay
    # a visible verify-me task, never a silent "faithfully imported" claim.
    for p in ir.get("parameters", []):
        curve = p.get("source_curve") or {}
        shape = curve.get("shape")
        fidelity = curve.get("fidelity", "exact")
        if shape == "exp" or fidelity == "close":
            label = p.get("name") or p.get("source_id_string") or p.get("id") or "?"
            todos.append(f"verify shaped curve for parameter '{label}' — emitted "
                         f"CLOSE/PARTIAL (approximate; NOT a faithful round-trip)")
    # Unresolved constructs
    for c in ir.get("constructs", []):
        ref = c.get("source_ref", {})
        todos.append(f"unresolved construct ({c.get('construct_type')}) at "
                     f"{ref.get('file')}:{ref.get('line')} — "
                     f"{c.get('enumeration_status')}, cardinality not inferred")
    # DSP
    todos.append(f"migrate DSP — classification '{dsp.get('classification')}' "
                 f"(scope {dsp.get('reachability_scope')})")
    # State
    if state.get("classification") == "opaque-custom":
        todos.append("opaque-custom state: binary session compatibility with the "
                     "original plugin is NOT supported")

    for pkg in integration_reqs.get("packages", []) or []:
        if not isinstance(pkg, dict):
            continue
        pkg_id = str(pkg.get("id") or "").strip()
        if not pkg_id:
            continue
        tag = "required" if pkg.get("required", True) else "recommended"
        reason = str(pkg.get("reason") or "source project integration").strip()
        todos.append(f"enable Pulp package `{pkg_id}` ({tag}): {reason}")

    for opt in integration_reqs.get("cmake_options", []) or []:
        if not isinstance(opt, dict):
            continue
        name = str(opt.get("name") or "").strip()
        if not name:
            continue
        value = opt.get("value", True)
        reason = str(opt.get("reason") or "source project integration").strip()
        todos.append(f"configure `{name}={_cmake_value(value)}`: {reason}")

    for asset in integration_reqs.get("asset_inputs", []) or []:
        if not isinstance(asset, dict):
            continue
        path = str(asset.get("path") or "").strip()
        if not path:
            continue
        if asset.get("copy_policy") != "copy_to_scaffold":
            reason = str(asset.get("reason") or "source asset requires review").strip()
            todos.append(f"review source asset `{path}`: {reason}")

    # UI
    ui_kind = (ir.get("ui", {}) or {}).get("kind") or "native"
    if ui_kind == "webview":
        todos.append("WebView UI: copy your original WebView assets (HTML/JS/CSS) "
                     "into ui/ — the importer cannot extract bundled binary "
                     "resources. The emitted create_view() hosts a Pulp "
                     "WebViewPanel + a native param-by-string-key bridge shim; "
                     "wire up the payload decode and any bespoke message handlers.")

    audio_parity = "no"
    if dsp.get("classification") == "portable-core":
        audio_parity = "partial"  # core copied, wiring still TODO
    elif dsp.get("classification") in ("framework-bound-mappable", "framework-bound-midi"):
        audio_parity = "no"

    # UI parity: for a webview import the editor *shell* (WebViewPanel host +
    # bridge wiring) is scaffolded, but the actual web payload + bridge decode
    # are TODOs, so it is "partial" — better than the native/custom-paint path
    # ("no") which only emits a placeholder, but not a finished UI.
    ui_parity = "partial" if ui_kind == "webview" else "no"

    return {
        "status": "unresolved",
        "schema": "pulp.import.migration_status.draft1",
        "source": ir.get("source", {}),
        "plugin": md.get("name"),
        "emit_tool": emit_tool,
        "audio_precision": {
            "source_native_f64": _source_native_f64(ir),
            "sample_type": _audio_precision(ir).get("sample_type", "float"),
            "confidence": _audio_precision(ir).get("confidence"),
            "evidence": _audio_precision(ir).get("evidence", []),
        },
        "ui_kind": ui_kind,
        "verdict": {
            "builds": "yes",
            "audio_parity": audio_parity,
            "ui_parity": ui_parity,
            "session_compatibility": "no",
        },
        "dsp_classification": dsp.get("classification"),
        "state_classification": state.get("classification"),
        "copied_user_files": [
            {"file": c, "provenance": "copied-user-file"} for c in copied_cores
        ],
        "integration_requirements": integration_reqs,
        "constructs": ir.get("constructs", []),
        "bind_grid": _gen_param_bind_grid(ir),
        "todos": todos,
        "confidence_overall": ir.get("confidence_overall"),
    }


def _verdict_line(status: dict) -> str:
    v = status["verdict"]
    return (f"Builds: {v['builds']}. "
            f"Audio parity: {v['audio_parity']}. "
            f"UI parity: {v['ui_parity']}. "
            f"Session compatibility: {v['session_compatibility']}.")


def _gen_report(ir: dict, status: dict, copied_cores: list[str],
                render_report: Callable[[dict], str]) -> str:
    base = render_report(ir)
    L: list[str] = []
    # Extraction-health banner (see `extraction_health`): when the extraction is
    # not trustworthy — a framework header failed to resolve, build_ir errored,
    # or an explicit framework path yielded zero params with no explaining
    # construct — say so LOUDLY at the top. A scaffold built on a false-clean
    # extraction is worse than none; the report must not read as a clean import.
    health = ir.get("extraction_health") if isinstance(ir, dict) else None
    failed = bool(ir.get("extraction_failed")) or (
        isinstance(health, dict) and health.get("ok") is False)
    if failed:
        reasons = (health or {}).get("reasons") or []
        L.append("> ⚠️ **EXTRACTION FAILED — this import is NOT trustworthy.**")
        L.append(">")
        L.append("> The extractor could not produce a reliable ProjectIR, so the "
                 "scaffold below is built on incomplete data. Resolve these before "
                 "trusting anything in this report:")
        for r in reasons:
            L.append(f">   - {r}")
        L.append(">")
    L.append("> **EMIT verdict:** " + _verdict_line(status))
    L.append(">")
    L.append("> A Pulp migration **scaffold** was generated from this IR. It "
             "builds against the Pulp SDK and is an honest starting point — not "
             "a finished plugin. Search the generated sources for `TODO(import)`.")
    if copied_cores:
        L.append(">")
        L.append("> Framework-free DSP core(s) copied verbatim "
                 "(provenance: copied-user-file): "
                 + ", ".join(f"`{c}`" for c in copied_cores) + ".")
    L.append("")
    if status.get("todos"):
        L.append("## Migration TODOs (emitted)")
        L.append("")
        for t in status["todos"]:
            L.append(f"- [ ] {t}")
        L.append("")
    return "\n".join(L) + "\n" + base


# --- portable-core copy -----------------------------------------------------

def make_framework_free_predicate(framework_tokens: list[str]):
    """Build a predicate that decides whether a user header is framework-free.

    The substrate names no vendor: the caller passes its framework's source
    tells as DATA (e.g. the namespace qualifier `fw::` and the include stem
    `fw`). The returned predicate strips comments first (so a header whose
    *prose* mentions the framework — "no fw:: anywhere" in a comment — is not
    falsely disqualified) and then rejects only real *usage*: a `<token>::`
    qualifier or a `#include <token…>` / `#include "token…"`.

    `framework_tokens` is a list of namespace/include stem strings (case-
    sensitive, matched as identifier prefixes). Pass [] to treat every header as
    framework-free (no source tells to look for).
    """
    pats = []
    for tok in framework_tokens:
        t = re.escape(tok)
        pats.append(rf"\b{t}::")
        pats.append(rf"#\s*include\s*[<\"][^>\"]*{t}")
    rx = re.compile("|".join(pats)) if pats else None

    def is_framework_free(text: str) -> bool:
        if rx is None:
            return True
        code = re.sub(r"//[^\n]*", "", text)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        return rx.search(code) is None

    return is_framework_free


def _portable_core_files(ir: dict, source_dir: Path | None,
                         is_framework_free,
                         boundary_name_markers: list[str]) -> list[Path]:
    """Header files that look like the framework-free core the IR pointed at.

    The IR's dsp diagnostics name core classes/symbols but not files, so the
    spike copies any project header whose contents are framework-free (per the
    injected predicate). This is conservative: it only fires for the
    portable-core classification and only copies headers that are demonstrably
    framework-free.

    `boundary_name_markers` is DATA from the importer: filename substrings that
    identify framework-boundary / metadata headers which must NOT be treated as
    a portable DSP core even when they happen to be framework-free of source
    tells (e.g. a metadata config header full of `#define`s). The substrate
    names no such file itself — the importer passes its own markers.
    """
    if ir.get("dsp", {}).get("classification") != "portable-core":
        return []
    if source_dir is None or not source_dir.exists():
        return []
    out: list[Path] = []
    for hdr in sorted(source_dir.rglob("*.h")) + sorted(source_dir.rglob("*.hpp")):
        try:
            text = hdr.read_text(errors="ignore")
        except OSError:
            continue
        if any(marker in hdr.name for marker in boundary_name_markers):
            continue  # framework boundary / metadata files, not the portable core
        if not is_framework_free(text):
            continue  # not framework-free
        out.append(hdr)
    return out


def _safe_asset_destination(source_path: str) -> str | None:
    posix = source_path.replace("\\", "/")
    rel = PurePosixPath(posix)
    if rel.is_absolute():
        return None
    if any(part in ("", ".", "..") for part in rel.parts):
        return None
    return str(PurePosixPath("assets", "imported", *rel.parts))


def _integration_asset_files(ir: dict, source_dir: Path | None) -> list[FileSpec]:
    if source_dir is None or not source_dir.exists():
        return []
    reqs = _integration_requirements(ir)
    assets = [a for a in reqs.get("asset_inputs", []) if isinstance(a, dict)]
    if not assets:
        return []

    root = source_dir.resolve()
    out: list[FileSpec] = []
    seen: set[str] = set()
    for asset in assets:
        if asset.get("copy_policy") != "copy_to_scaffold":
            continue
        source_path = str(asset.get("path") or "").strip()
        dest = _safe_asset_destination(source_path)
        if not dest or dest in seen:
            continue
        src = (source_dir / source_path).resolve()
        try:
            src.relative_to(root)
        except ValueError:
            continue
        if not src.is_file():
            continue
        out.append(FileSpec(dest, provenance="copied-user-file",
                            copy_from=str(src), classification="asset"))
        seen.add(dest)
    return out


# --- production (path/content/provenance tuples) ----------------------------

class FileSpec:
    """One file the EMIT step proposes. Either `content` (generated/stub) or
    `copy_from` (a verbatim copy of a framework-free user file) is set, never
    both. `provenance` matches the SPI EmissionManifest vocabulary:
    "generated" | "copied-user-file" | "stub". `classification` is an optional
    hint (e.g. "source" / "build" / "report" / "manifest")."""

    __slots__ = ("path", "content", "provenance", "copy_from", "classification")

    def __init__(self, path: str, *, content: str | None = None,
                 provenance: str, copy_from: str | None = None,
                 classification: str | None = None):
        self.path = path
        self.content = content
        self.provenance = provenance
        self.copy_from = copy_from
        self.classification = classification

    def as_manifest_entry(self) -> dict:
        entry: dict = {"path": self.path, "provenance": self.provenance}
        if self.classification:
            entry["classification"] = self.classification
        if self.copy_from is not None:
            entry["copy_from"] = self.copy_from
        else:
            entry["content"] = self.content or ""
        return entry


def _default_render_report(ir: dict) -> str:
    """Minimal IR-only report when an importer does not inject its own.

    The shipped importers pass their framework-specific `render_report`; this
    fallback keeps the substrate usable (and testable) standalone without
    naming any vendor.
    """
    md = ir.get("metadata", {})
    src = ir.get("source", {})
    L = [f"# Import Report — {md.get('name') or src.get('project_dir') or 'project'}",
         "", f"- **Source framework:** {src.get('framework')}",
         f"- **Overall confidence:** {ir.get('confidence_overall')}",
         f"- **Parameters:** {len(ir.get('parameters', []))}", ""]
    return "\n".join(L) + "\n"


def produce(ir: dict, source_dir: Path | None = None,
            sdk_version: str | None = None, *,
            render_report: Callable[[dict], str] | None = None,
            framework_free_predicate=None,
            emit_tool: str = "pulp-importer-substrate-emit/0.0.0",
            id_label: str = "param",
            tool_label: str = "",
            boundary_name_markers: list[str] | None = None,
            confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR) -> dict:
    """Pure production step: turn a ProjectIR into the set of files a Pulp
    migration scaffold consists of, WITHOUT touching the output directory.

    Vendor-neutral: every generator consumes vendor-neutral IR fields. The
    framework-specific touch-points are injected as DATA / callables:
      - `render_report(ir) -> str` renders the importer's IMPORT_REPORT.md prose.
      - `framework_free_predicate(text) -> bool` decides whether a user header
        is framework-free enough to copy verbatim (build it with
        `make_framework_free_predicate(tokens)`). When omitted, every header is
        treated as framework-free.
      - `id_label` / `tool_label` are cosmetic prose labels for the generated
        comments (e.g. the source id-string origin and the "generated by the X
        importer" line). They default to the vendor-neutral "param" / "" so the
        substrate names no vendor; an importer passes its own label as DATA.
      - `emit_tool` is the migration_status emit-tool identity string (DATA).
      - `boundary_name_markers` is DATA: filename substrings of framework
        boundary/metadata headers that must NOT be copied verbatim as a portable
        DSP core (e.g. the importer's `PluginProcessor`/`PluginEditor` or
        metadata `config.h`). Defaults to [] so the substrate names no file.
      - `confidence_floor` gates concrete value emission (plan §§7.4/16.3): a
        parameter whose IR `confidence` is below the floor was inferred from a
        non-literal / runtime-computed source, so its resolved value is a guess.
        Below the floor the param is emitted as a labelled `// TODO(import):
        low-confidence …` stub instead of a concrete `add_parameter` call.
        Defaults to `DEFAULT_CONFIDENCE_FLOOR` (0.5).

    Returns a dict with:
      - "files": list[FileSpec] — every scaffold file, with content (generated/
        stub) or copy_from (verbatim portable-core copies).
      - "migration_status": the migration_status.json dict.
      - "formats" / "deferred_formats": the format split.
      - "unresolved": labelled-stub / unresolved-construct strings.
      - plus the same summary fields the CLI returns.

    Both `emit()` (writes to disk) and the SPI `emit` verb (serialises into an
    EmissionManifest) are thin shells over this function — there is exactly one
    place the scaffold is shaped, shared by every importer.
    """
    if render_report is None:
        render_report = _default_render_report
    if framework_free_predicate is None:
        framework_free_predicate = make_framework_free_predicate([])
    if boundary_name_markers is None:
        boundary_name_markers = []

    # A call already reported as a loop/factory construct must NOT also surface
    # as a high-confidence parameter (the S3 double-report). Filter phantoms up
    # front so EVERY downstream reader (enum, bind-grid, registration, status)
    # sees one consistent parameter list.
    ir = _suppress_construct_claimed_params(ir)

    md = ir.get("metadata", {})
    base_name = md.get("name") or ir.get("source", {}).get("project_dir") or "Imported"
    class_name = _pascal(base_name) + "Processor"
    target = _pascal(base_name)
    factory_name = _snake(base_name)
    namespace = "pulp_import_" + _snake(base_name)
    lower = _snake(base_name)
    header_name = "PluginProcessor.hpp"
    if sdk_version is None:
        sdk_version = ""

    # Identify framework-free portable-core files to copy verbatim.
    core_files = _portable_core_files(ir, source_dir, framework_free_predicate,
                                      boundary_name_markers)
    copied_cores: list[str] = [f"src/{cf.name}" for cf in core_files]
    integration_assets = _integration_asset_files(ir, source_dir)

    params = _param_id_enum(ir.get("parameters", []))

    files: list[FileSpec] = []

    # Verbatim portable-core copies (provenance: copied-user-file). The content
    # is NOT inlined — the SDK copies the file from `copy_from` so the user's own
    # DSP is never rewritten and stays exempt from the clean-room output scan.
    for cf in core_files:
        files.append(FileSpec(f"src/{cf.name}", provenance="copied-user-file",
                              copy_from=str(cf), classification="source"))
    files.extend(integration_assets)

    # Generated sources.
    files.append(FileSpec(f"src/{header_name}", provenance="generated",
                          classification="source",
                          content=_gen_header(ir, class_name, factory_name,
                                              namespace, params, header_name,
                                              id_label, tool_label)))
    files.append(FileSpec("src/PluginProcessor.cpp", provenance="generated",
                          classification="source",
                          content=_gen_source(ir, class_name, factory_name,
                                              namespace, params, header_name,
                                              copied_cores, id_label,
                                              confidence_floor)))
    files.append(FileSpec("src/clap_entry.cpp", provenance="generated",
                          classification="source",
                          content=_gen_clap_entry(factory_name, namespace,
                                                  header_name)))

    # WebView UI scaffold: when the source editor was a WebView, the generated
    # create_view() serves the `ui/` directory through a Pulp WebViewPanel. The
    # bundled web payload can't be extracted statically, so ship a placeholder
    # `ui/index.html` (a real, building page that proves the bridge) the porter
    # replaces with the original HTML/JS/CSS. The placeholder is "generated", not
    # a copied user file.
    webview = _is_webview_ui(ir)
    if webview:
        files.append(FileSpec(f"{_WEBVIEW_UI_DIR}/{_WEBVIEW_DEFAULT_ENTRY}",
                              provenance="generated", classification="asset",
                              content=_gen_webview_index_html(ir)))

    # Formats: emit CLAP only — it is self-contained (no external SDK), always
    # links (we generate its entry point), and is dlopen-testable. The source's
    # other formats (VST3/AU/Standalone) are DEFERRED, not emitted: Standalone
    # needs a generated main(), and VST3/AU need their developer-supplied SDKs +
    # entry points. Carrying them into FORMATS makes the scaffold fail to link
    # (Standalone: "_main not found"), which defeats the "builds: yes" MVP
    # contract. They are recorded as a migration TODO so the info isn't lost.
    formats = ["CLAP"]
    deferred_formats = [f for f in md.get("formats", [])
                        if FORMAT_MAP.get(f) and FORMAT_MAP[f] != "CLAP"]

    cmake_sources = ["src/PluginProcessor.cpp", "src/clap_entry.cpp"]
    files.append(FileSpec("CMakeLists.txt", provenance="generated",
                          classification="build",
                          content=_gen_cmake(ir, target, namespace, factory_name,
                                            formats, cmake_sources, lower,
                                            sdk_version, tool_label)))

    # Migration status + report.
    status = _gen_migration_status(ir, copied_cores, emit_tool)
    if integration_assets:
        status["copied_integration_assets"] = [
            {"file": f.path, "provenance": f.provenance}
            for f in integration_assets
        ]
    if deferred_formats:
        status.setdefault("todos", []).append(
            f"Enable additional plugin formats {deferred_formats}: Standalone needs a "
            "generated main(); VST3/AU need their developer-supplied SDKs + entry points. "
            "The MVP scaffold ships CLAP only (self-contained + loadable).")
        status["deferred_formats"] = deferred_formats

    files.append(FileSpec("migration_status.json", provenance="generated",
                          classification="manifest",
                          content=json.dumps(status, indent=2) + "\n"))
    files.append(FileSpec("IMPORT_REPORT.md", provenance="generated",
                          classification="report",
                          content=_gen_report(ir, status, copied_cores,
                                              render_report)))

    return {
        "files": files,
        "migration_status": status,
        "formats": formats,
        "deferred_formats": deferred_formats,
        "unresolved": list(status.get("todos", [])),
        "target": target,
        "class_name": class_name,
        "namespace": namespace,
        "params": [e for e, _, _ in params],
        "copied_cores": copied_cores,
        "copied_integration_assets": [f.path for f in integration_assets],
        "verdict": _verdict_line(status),
    }


# --- driver -----------------------------------------------------------------

def emit(ir: dict, out_dir: Path, source_dir: Path | None = None,
         sdk_version: str | None = None, *,
         render_report: Callable[[dict], str] | None = None,
         framework_free_predicate=None,
         emit_tool: str = "pulp-importer-substrate-emit/0.0.0",
         id_label: str = "param",
         tool_label: str = "",
         boundary_name_markers: list[str] | None = None,
         confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR) -> dict:
    """Materialise a scaffold on disk. Thin shell over `produce()`: writes each
    FileSpec (content -> write, copy_from -> verbatim copy) under `out_dir`."""
    prod = produce(ir, source_dir=source_dir, sdk_version=sdk_version,
                   render_report=render_report,
                   framework_free_predicate=framework_free_predicate,
                   emit_tool=emit_tool, id_label=id_label, tool_label=tool_label,
                   boundary_name_markers=boundary_name_markers,
                   confidence_floor=confidence_floor)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in prod["files"]:
        dest = out_dir / spec.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if spec.copy_from is not None:
            shutil.copy2(Path(spec.copy_from), dest)
        else:
            dest.write_text(spec.content or "")

    return {
        "out_dir": str(out_dir),
        "target": prod["target"],
        "class_name": prod["class_name"],
        "namespace": prod["namespace"],
        "formats": prod["formats"],
        "params": prod["params"],
        "copied_cores": prod["copied_cores"],
        "verdict": prod["verdict"],
    }
