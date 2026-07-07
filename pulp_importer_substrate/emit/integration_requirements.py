"""CMake/status helpers for ProjectIR integration requirements."""
from __future__ import annotations

import re

_TUNING_PROVIDER_PACKAGES = {
    "mts-esp": "MTS_ESP",
    "sst-tuning-library": "SCALA",
}


def integration_requirements(ir: dict) -> dict:
    reqs = ir.get("integration_requirements") or {}
    return reqs if isinstance(reqs, dict) else {}


def cmake_value(value) -> str:
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if re.match(r"^[A-Za-z0-9_./:+-]+$", text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def gen_cmake_prelude(ir: dict) -> list[str]:
    reqs = integration_requirements(ir)
    packages = [p for p in reqs.get("packages", []) if isinstance(p, dict)]
    options = [o for o in reqs.get("cmake_options", []) if isinstance(o, dict)]
    if not packages and not options:
        return []

    lines: list[str] = []
    lines.append("# Importer-detected integration requirements.")
    if packages:
        lines.append("# Run these from the scaffold root if they are not already present:")
        for pkg in packages:
            pkg_id = str(pkg.get("id") or "").strip()
            if not pkg_id:
                continue
            tag = "required" if pkg.get("required", True) else "recommended"
            reason = str(pkg.get("reason") or "").strip()
            suffix = f" ({tag}: {reason})" if reason else f" ({tag})"
            lines.append(f"#   pulp add {pkg_id}{suffix}")
        lines.append("include(cmake/pulp-packages.cmake OPTIONAL)")
    if options:
        lines.append("# If this scaffold builds Pulp from source, these options enable")
        lines.append("# SDK-side provider code. Installed SDKs can attach optional")
        lines.append("# providers below after `pulp add` installs the packages.")
        for opt in options:
            name = str(opt.get("name") or "").strip()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                continue
            value = cmake_value(opt.get("value", True))
            reason = str(opt.get("reason") or "Required by imported project").strip()
            reason = reason.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'set({name} {value} CACHE BOOL "{reason}" FORCE)')
    lines.append("")
    return lines


def gen_target_links(ir: dict, plugin_target: str) -> list[str]:
    reqs = integration_requirements(ir)
    packages = [p for p in reqs.get("packages", []) if isinstance(p, dict)]
    if not packages:
        return []

    lines: list[str] = []
    tuning_providers: list[str] = []
    for pkg in packages:
        pkg_id = str(pkg.get("id") or "").strip()
        provider = _TUNING_PROVIDER_PACKAGES.get(pkg_id)
        if provider:
            tuning_providers.append(provider)
            continue
        for cmake_target in pkg.get("cmake_targets", []) or []:
            cmake_target = str(cmake_target).strip()
            if not cmake_target or '"' in cmake_target or "\n" in cmake_target:
                continue
            lines.append(f"if(TARGET {cmake_target})")
            lines.append(f"    target_link_libraries({plugin_target} PRIVATE {cmake_target})")
            lines.append("endif()")

    if tuning_providers:
        provider_list = " ".join(dict.fromkeys(tuning_providers))
        lines.insert(0, f"    pulp_enable_midi_tuning_provider({plugin_target} {provider_list})")
        lines.insert(0, "if(COMMAND pulp_enable_midi_tuning_provider)")
        lines.insert(2, "else()")
        lines.insert(
            3,
            '    message(FATAL_ERROR "This imported project uses microtuning '
            'features. Upgrade the Pulp SDK to one that provides '
            'pulp_enable_midi_tuning_provider().")'
        )
        lines.insert(4, "endif()")
    if lines:
        lines.insert(0, "# Attach optional integration packages when `pulp add` has installed them.")
        lines.append("")
    return lines
