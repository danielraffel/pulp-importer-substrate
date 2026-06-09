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
import re
import shutil
from pathlib import Path
from typing import Callable

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


# --- parameter emission -----------------------------------------------------

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
    skew = curve.get("skew", 1.0)
    symmetric = bool(curve.get("symmetric", False))
    shaped = (skew not in (None, 1.0)) or symmetric
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
    if shaped:
        # Pulp's ParamRange now carries the skew/symmetric shape directly, so
        # the curve round-trips through normalize/denormalize instead of being
        # downgraded to LINEAR+PARTIAL. The mapping matches the source curve's
        # behaviour (CLOSE-with-tolerance: float-precision, not bit-exact).
        lines.append(f"        // source curve skew={skew}"
                     f"{' (symmetric)' if symmetric else ''} emitted as a shaped "
                     f"ParamRange (CLOSE: round-trips within float tolerance)")

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
    if shaped:
        # 6-field aggregate: {min, max, default, step, skew, symmetric_skew}.
        sk = _cpp_float(skew if skew is not None else 1.0, 1.0)
        sym = "true" if symmetric else "false"
        lines.append(f"            .range = {{{mn}, {mx}, {df}, {st}, {sk}, {sym}}},")
    else:
        lines.append(f"            .range = {{{mn}, {mx}, {df}, {st}}},")
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

def _emit_process_body(ir: dict, copied_cores: list[str]) -> list[str]:
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
        L.append("                out[i] = 0.0f;")
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
        L.append("                out[i] = 0.0f;")
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
    L.append("        };")
    L.append("    }")
    return L


# --- file generators --------------------------------------------------------

def _gen_header(ir: dict, class_name: str, factory_name: str, namespace: str,
                params: list[tuple[str, int, str]], header_name: str,
                id_label: str, tool_label: str) -> str:
    L: list[str] = []
    L.append("#pragma once")
    L.append("")
    L.append(f"// {ir.get('metadata', {}).get('name', class_name)} — Pulp migration scaffold")
    L.append(f"// Generated by the {tool_label}importer EMIT step (spike). This is a")
    L.append("// BUILDING starting point, not a finished plugin. Search for")
    L.append("// `TODO(import)` to find everything that still needs migration.")
    L.append("")
    L.append("#include <pulp/format/processor.hpp>")
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
    L.append("")
    L.append("    std::vector<uint8_t> serialize_plugin_state() const override;")
    L.append("    bool deserialize_plugin_state(std::span<const uint8_t> data) override;")
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


def _gen_source(ir: dict, class_name: str, factory_name: str, namespace: str,
                params: list[tuple[str, int, str]], header_name: str,
                copied_cores: list[str], id_label: str,
                confidence_floor: float) -> str:
    state = ir.get("state_model", {})
    opaque = state.get("classification") == "opaque-custom"
    ir_params = ir.get("parameters", [])
    # map enum_name -> ir param dict by source id
    by_sid = {(p.get("source_id_string") or p.get("id")): p for p in ir_params}

    L: list[str] = []
    L.append(f'#include "{header_name}"')
    L.append("")
    L.append(f"namespace {namespace} {{")
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
    return "\n".join(L)


def _gen_migration_status(ir: dict, copied_cores: list[str],
                          emit_tool: str) -> dict:
    md = ir.get("metadata", {})
    dsp = ir.get("dsp", {})
    state = ir.get("state_model", {})
    todos: list[str] = []

    # Skewed / symmetric params now emit a shaped ParamRange that round-trips
    # through Pulp's normalize/denormalize (CLOSE-with-tolerance), so they are
    # no longer migration blockers — nothing is recorded as a TODO for them.
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

    audio_parity = "no"
    if dsp.get("classification") == "portable-core":
        audio_parity = "partial"  # core copied, wiring still TODO
    elif dsp.get("classification") in ("framework-bound-mappable", "framework-bound-midi"):
        audio_parity = "no"

    return {
        "status": "unresolved",
        "schema": "pulp.import.migration_status.draft1",
        "source": ir.get("source", {}),
        "plugin": md.get("name"),
        "emit_tool": emit_tool,
        "verdict": {
            "builds": "yes",
            "audio_parity": audio_parity,
            "ui_parity": "no",
            "session_compatibility": "no",
        },
        "dsp_classification": dsp.get("classification"),
        "state_classification": state.get("classification"),
        "copied_user_files": [
            {"file": c, "provenance": "copied-user-file"} for c in copied_cores
        ],
        "constructs": ir.get("constructs", []),
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

    params = _param_id_enum(ir.get("parameters", []))

    files: list[FileSpec] = []

    # Verbatim portable-core copies (provenance: copied-user-file). The content
    # is NOT inlined — the SDK copies the file from `copy_from` so the user's own
    # DSP is never rewritten and stays exempt from the clean-room output scan.
    for cf in core_files:
        files.append(FileSpec(f"src/{cf.name}", provenance="copied-user-file",
                              copy_from=str(cf), classification="source"))

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
