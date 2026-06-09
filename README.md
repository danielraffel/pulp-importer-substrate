# pulp-importer-substrate

The **vendor-agnostic extraction core** shared by Pulp's framework importers
(`pulp-import-juce`, `pulp-import-iplug`, …).

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
python3 tests/test_substrate.py
```

The token helpers are pure, so they're unit-tested without libclang (including a
trailing-dot-float case for `numeric_seq` — the regression can now only live in
one place). A libclang-dependent smoke runs only if the binding is importable.

## Status

Spike-grade. NOT production code and NOT the shipped importer — it uses the
developer's local Apple libclang for convenience; the shipped tooling pins its
own LLVM/libclang and records the exact version.
