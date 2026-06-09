"""Stable identifier hashing shared by every importer.

Pulp ParamIDs are uint32. Source frameworks identify params differently (string
ids, enum members, …), so each importer records the source identity AND a stable
proposed uint32 derived here, so preset/state migration semantics survive the
SDK assigning ids. The hash is framework-agnostic.
"""
from __future__ import annotations


def fnv1a_u32(s: str) -> int:
    """Stable string -> uint32 (proposed Pulp ParamID), via 32-bit FNV-1a."""
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h
