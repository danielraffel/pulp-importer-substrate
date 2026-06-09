"""Framework-agnostic category mapping shared by every importer.

The source-framework category vocabulary (`effect` / `instrument` /
`midi_effect`) is normalized identically across importers, so the
category -> Pulp plugin-category map lives here once. How each importer
*derives* the category from its own metadata (project files vs config headers)
stays per-importer.
"""
from __future__ import annotations

CATEGORY_TO_PULP = {
    "effect": "Effect",
    "instrument": "Instrument",
    "midi_effect": "MidiEffect",
}
