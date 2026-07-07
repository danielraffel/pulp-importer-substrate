"""pulp-importer-substrate — the vendor-agnostic extraction core.

Shared, framework-neutral building blocks used by every Pulp framework
importer. It deliberately names NO vendor: framework-specific metadata
scrapers, parameter extractors, classification bodies, and DSP/UI mapping
tables stay in each importer repo. See README.md for the full contract of what
is shared and what is deliberately left per-importer.

The public surface is re-exported here so an importer can do
`from pulp_importer_substrate import *` (or import individual names) and have
the shared helpers behave exactly as the previously copy-pasted locals did.
"""
from __future__ import annotations

from .ids import fnv1a_u32
from .libclang_setup import (
    APPLE_LIBCLANG,
    _configure_libclang,
    system_include_args,
)
from .mappings import CATEGORY_TO_PULP
from .integrations import detect_tuning_integration_requirements
from .tokens import (
    all_strings,
    first_bool,
    first_int,
    first_string,
    numeric_seq,
    toks,
)
from .cursors import (
    _cursor_kind,
    _LOOP_KINDS,
    _RUNTIME_REF_KINDS,
    arg_is_computed,
    find_loops,
    find_method,
    in_loop,
    in_main_file,
    walk,
)

# Pinned to the import SPI contract version the substrate targets. Bump in
# lockstep with `tools/import/schemas/import-spi-v0` when the shared extraction
# surface changes shape; importers pin a substrate version per SPI version.
SPI_VERSION = "import-spi-v0"

__version__ = "0.0.0"

__all__ = [
    # ids
    "fnv1a_u32",
    # libclang setup
    "APPLE_LIBCLANG",
    "_configure_libclang",
    "system_include_args",
    # tokens
    "toks",
    "first_string",
    "all_strings",
    "first_int",
    "numeric_seq",
    "first_bool",
    # cursors / AST predicates
    "walk",
    "_cursor_kind",
    "in_main_file",
    "arg_is_computed",
    "find_method",
    "find_loops",
    "in_loop",
    "_LOOP_KINDS",
    "_RUNTIME_REF_KINDS",
    # mappings
    "CATEGORY_TO_PULP",
    # optional integrations
    "detect_tuning_integration_requirements",
    # metadata
    "SPI_VERSION",
    "__version__",
]
