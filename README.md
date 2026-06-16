# pulp-importer-substrate

The **vendor-agnostic extraction core** shared by Pulp's per-framework
importer packages.

Each importer turns *the user's own* plugin source into a draft Pulp
`ProjectIR.json` by parsing it with libclang. A large fraction of that work —
configuring libclang, tokenizing literals, walking the AST, hashing param ids —
is identical regardless of the source framework. Before this package existed,
that core was **copy-pasted** into every importer, and a single bug (the
trailing-dot float in `numeric_seq`) had to be fixed in two places. This package
holds that core once.

It deliberately **names no vendor**: there is no `juce`, `iplug`, `steinberg`,
or `wdl` string anywhere in `pulp_importer_substrate/`. Framework names are
runtime DATA owned by the individual importers, never by the shared core.

## What's shared (the public surface)

Re-exported from `pulp_importer_substrate` (`from pulp_importer_substrate import *`):

| Name | Module | Purpose |
|------|--------|---------|
| `fnv1a_u32` | `ids` | stable string → uint32 proposed Pulp ParamID |
| `APPLE_LIBCLANG`, `_configure_libclang`, `system_include_args` | `libclang_setup` | point the binding at Apple libclang; recover the C++ system include search list from the real driver |
| `toks`, `first_string`, `all_strings`, `first_int`, `numeric_seq`, `first_bool` | `tokens` | pure token-stream helpers (this is where the trailing-dot float fix lives, in ONE place) |
| `walk`, `_cursor_kind`, `in_main_file`, `arg_is_computed`, `find_method`, `find_loops`, `in_loop`, `_LOOP_KINDS`, `_RUNTIME_REF_KINDS` | `cursors` | cursor traversal + generic AST predicates |
| `CATEGORY_TO_PULP` | `mappings` | category → Pulp plugin-category map |

### The `walk` choice (worth knowing)

`walk` is the **hardened** variant: when a real `--framework-path` parse surfaces
a `CursorKind` id the installed `clang.cindex` enum doesn't know, `walk` skips
yielding that cursor but still descends into its children, so the parse degrades
to a partial extraction instead of aborting. On a clean parse (every cursor kind
resolves) it is identical to a plain pre-order walk — which is why adopting it in
both importers is byte-identical against their existing goldens.

## The shared EMIT core (`pulp_importer_substrate.emit`)

The same de-duplication argument applies to the EMIT step. EMIT consumes a
**vendor-neutral** ProjectIR (already-resolved fields like `dsp.mappings`,
`parameters[].pulp_range` / `source_curve` / `proposed_pulp_id`, the state/DSP
classifications) and proposes a Pulp migration scaffold. Because the IR is
framework-neutral, the scaffold generators are framework-neutral too — so they
live here once and both importers share them.

`pulp_importer_substrate.emit` exposes:

| Name | Purpose |
|------|---------|
| `produce(ir, …)` | pure: ProjectIR → `{files: [FileSpec], migration_status, formats, deferred_formats, unresolved, …}`. The one place the scaffold is shaped. |
| `emit(ir, out_dir, …)` | thin shell over `produce()` that writes the FileSpecs to disk (content → write, `copy_from` → verbatim copy). |
| `FileSpec` | one proposed file: either inline `content` (generated/stub) or a `copy_from` verbatim portable-core copy; `as_manifest_entry()` serialises it for the SPI EmissionManifest. |
| `make_framework_free_predicate(tokens)` | builds the "is this header framework-free enough to copy verbatim?" predicate from the importer's source-tell tokens (DATA). |

It names **no vendor**. The framework-specific touch-points are injected as
DATA / callables by each importer:

- `render_report(ir) -> str` — the importer's own IMPORT_REPORT.md renderer
  (column sets and provenance prose differ per framework). The core appends an
  EMIT verdict block on top.
- `framework_free_predicate(text) -> bool` — built from the framework's source
  tells (its `fw::` namespace qualifier + include stems) via
  `make_framework_free_predicate`.
- `boundary_name_markers: [str]` — filename substrings of framework
  boundary/metadata headers that must never be copied as a DSP core
  (e.g. an importer's `PluginProcessor`/`PluginEditor`, or a metadata
  `config.h`).
- `id_label` / `tool_label` — cosmetic prose labels in the generated comments.
- `emit_tool` — the migration_status emit-tool identity string.

Each importer keeps a **thin shim** that injects its DATA and exposes the SPI
`emit` verb / CLI; the generator bodies are not duplicated. The `dsp_map.json`
itself stays per-importer because the *extractor* resolves it into the IR — emit
only ever reads the already-resolved `dsp.mappings`.

### UI branch (`ui.kind` DATA)

The emit core branches the editor scaffold on the IR's vendor-neutral
`ui.kind`:

- `ui.kind == "webview"` → a **Pulp webview-ui scaffold**: a `create_view()`
  hosting a `pulp::view::WebViewPanel` that serves an embedded `ui/` asset
  directory (via `make_webview_directory_resource_fetcher`), a placeholder
  `ui/index.html`, a `// TODO(import): copy your WebView assets …` note (bundled
  binary web resources can't be extracted), and a **native param-bridge shim**
  mapping the source's JS↔native param bridge onto Pulp's WebView bridge
  **param-by-string-key** (the preserved source parameter id strings). A literal
  HTML entry filename from `ui.asset_hints.html_entry` is used when the extractor
  resolved one; otherwise it defaults to `index.html` with a TODO.
- anything else (`"native"` / absent) → the existing native scaffold path,
  unchanged.

The framework-specific webview markers (JUCE's `WebBrowserComponent`, iPlug2's
`IWebView`/`IGraphicsWebView`) live **only in each importer's `classify_ui`** —
the core branches on the `kind` DATA and names no framework.

## What's deliberately NOT shared (stays per-importer)

These diverge between importers today; forcing them into the substrate would
either change behavior (breaking byte-identical goldens) or merge genuinely
different logic. They are **candidates for a later strategy-pattern extraction**,
not this slice:

- **`build_ir`** — the orchestration shell differs (JUCE derives buses from the
  constructor + category; iPlug2 derives them from `PLUG_CHANNEL_IO` metadata;
  the migration-task and diagnostic wording differs).
- **`main_cli`** — different CLI flags (`--juce-path` vs
  `--iplug-path`/`--igraphics-backend`/`--no-igraphics`).
- **`find_param_calls`** — different signatures (`find_param_calls(tu)` for JUCE,
  `find_param_calls(tu, main)` for iPlug2) and different call shapes
  (`make_unique<AudioParameter*>` vs `GetParam(k)->Init*`).
- **`extract_param`** — entirely framework-specific overload handling.
- **`classify_state` / `classify_dsp` / `classify_ui`** — different signatures
  and framework-specific symbol vocabularies (`juce::dsp::*` vs `WDL::*`,
  `createEditor` vs `IGraphics`/`IWebView`).
- **`build_constructs`** — byte-for-byte *almost* identical, but the
  `cardinality.reason` string differs (`non_literal_array_or_factory` vs
  `non_literal_loop_or_factory`), which is golden-visible. Left per-importer
  until the wording is reconciled.
- **`find_main_source` / `find_processor_class` / `lifecycle_hooks`** —
  framework-specific (`Processor`-named cpp vs `ProcessBlock`-containing cpp;
  `AudioProcessor` base vs `Plugin` base; different lifecycle hook names).
- **metadata scrapers** (`scrape_jucer`/`scrape_cmake` vs `config.h` `#define`
  scrape), **compile-arg synthesis** (`juce_compile_args` vs
  `iplug_compile_args`), and the **DSP/UI mapping tables**
  (`dsp_map.json` / `graphics_map.json`).

## Distribution / versioning

**Decision: standalone repo, pinned per SPI version** (recorded in the plan,
§21). Both importers depend on this package; they do not vendor a copy. Each
importer's `run_spike.sh` installs it editable (`pip install -e
../pulp-importer-substrate`) and the extractor also carries a `sys.path` fallback
to a sibling `../pulp-importer-substrate` checkout, so a clean checkout of an
importer repo plus this sibling works without a manual install step.

`SPI_VERSION` in `pulp_importer_substrate/__init__.py` pins the import-SPI
contract the shared surface targets. Bump it in lockstep with
`tools/import/schemas/import-spi-v0` when the shared extraction surface changes
shape, and pin the substrate version per SPI version in each importer.

## Tests

```bash
python3 tests/test_substrate.py   # extraction core (tokens / ids / mappings + libclang smoke)
python3 tests/test_emit.py        # emit core (scaffold generators + injection points)
```

The token helpers are pure, so they're unit-tested without libclang (including a
trailing-dot-float case for `numeric_seq` — the regression can now only live in
one place). A libclang-dependent smoke runs only if the binding is importable.
The emit core is pure too — `tests/test_emit.py` drives it with a synthetic
vendor-neutral IR (no libclang, no framework) and locks the scaffold file set,
shaped-vs-linear range emission, and the injected report/predicate/label
touch-points.

## Status

Spike-grade. NOT production code and NOT the shipped importer — it uses the
developer's local Apple libclang for convenience; the shipped tooling pins its
own LLVM/libclang and records the exact version.
