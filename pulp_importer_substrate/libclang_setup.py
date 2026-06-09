"""libclang configuration + system include resolution.

Vendor-agnostic. These helpers configure the Python `clang.cindex` binding
against the developer's local Apple libclang and recover the C++ system include
search list from the real driver — neither of which is framework-specific.

Spike-grade convenience: the shipped importer pins its own LLVM/libclang and
records the exact version; here we use whatever Apple toolchain is installed.
"""
from __future__ import annotations

import functools
import subprocess
from pathlib import Path

import clang.cindex as ci

APPLE_LIBCLANG = (
    "/Applications/Xcode.app/Contents/Developer/Toolchains/"
    "XcodeDefault.xctoolchain/usr/lib/libclang.dylib"
)


@functools.lru_cache(maxsize=1)
def _configure_libclang() -> None:
    if Path(APPLE_LIBCLANG).exists():
        ci.Config.set_library_file(APPLE_LIBCLANG)


@functools.lru_cache(maxsize=1)
def system_include_args() -> tuple[str, ...]:
    """Default C++ search paths from the real driver, passed as -isystem.

    libclang does not run the clang *driver* (which auto-detects the macOS SDK),
    so we hand it the same include search list `xcrun clang++` would use.
    """
    out: list[str] = []
    try:
        p = subprocess.run(
            ["xcrun", "clang++", "-std=c++20", "-x", "c++", "-E", "-v", "-"],
            input="", capture_output=True, text=True, check=False,
        )
        cap = False
        for ln in p.stderr.splitlines():
            if "search starts here" in ln:
                cap = True
                continue
            if "End of search list" in ln:
                break
            if cap:
                out += ["-isystem", ln.strip().split(" (")[0]]
        sdk = subprocess.run(
            ["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=False
        ).stdout.strip()
        if sdk:
            out = ["-isysroot", sdk] + out
    except Exception:
        pass
    return tuple(out)
