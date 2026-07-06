"""Vendor-agnostic SPI verb-envelope + JSON-stdio CLI shell shared by importers.

Each Pulp framework importer speaks the SDK's project-import SPI over
JSON-stdio: one request envelope per line on stdin, one response envelope per
line on stdout, stderr is logs only. The envelope framing is identical across
importers — spi_version negotiation, the ok/error result shape, verb dispatch,
per-request exception isolation (an importer bug becomes an honest error
envelope, never a crash on the wire), and the stdin read loop. Only the verb
handlers and the framework / importer id strings differ per importer, and those
are injected as DATA:

    from pulp_importer_substrate.spi import main_cli

    def detect(payload): ...
    def analyze(payload): ...
    def emit(payload): ...

    if __name__ == "__main__":
        raise SystemExit(main_cli(
            {"detect": detect, "analyze": analyze, "emit": emit},
            framework_id=FRAMEWORK_ID, importer_id=IMPORTER_ID))

`handlers` maps a verb name to a `callable(payload: dict) -> dict`. The
`framework_id` and `importer_id` strings are the importer's own provenance
markers — the SDK never names them, the importer owns them — and are accepted
here so the shell owns the full SPI handshake surface even though the envelope
framing itself is id-agnostic. This module names NO vendor: every
framework-specific value is passed in.

This mirrors the same inject-a-callback seam the shared EMIT core uses
(`produce(..., render_report=, framework_free_predicate=)`): the substrate owns
the vendor-neutral machinery, the importer injects its framework-specific DATA.
"""
from __future__ import annotations

import json
import sys
from typing import Callable, Mapping, TextIO

__all__ = ["SPI_VERSION", "handle", "main_cli"]

# The SPI protocol version this shell speaks. An importer may override it per
# call, but every current importer speaks v0.
SPI_VERSION = 0


def handle(req: dict, handlers: Mapping[str, Callable[[dict], dict]],
           spi_version: int = SPI_VERSION) -> dict:
    """Dispatch one request envelope to a verb handler; return the response.

    The response always carries `spi_version` + the request `id`. Version
    mismatch and unknown/unimplemented verbs produce loud, well-formed error
    envelopes rather than crashing. A handler that raises is caught and framed
    as an `analyze_error` envelope so a bug in one request never takes down the
    stdio loop or crashes on the wire.
    """
    rid = req.get("id", "")
    verb = req.get("verb")
    resp = {"spi_version": spi_version, "id": rid}
    if req.get("spi_version") != spi_version:
        resp.update(ok=False, error={"code": "spi_version_mismatch",
                                     "message": f"importer speaks spi_version {spi_version}"})
        return resp
    fn = handlers.get(verb)
    if fn is None:
        resp.update(ok=False, error={"code": "unimplemented_verb",
                                     "message": f"verb '{verb}' is not implemented by this spike "
                                                "(plan/emit are SDK-driven)."})
        return resp
    try:
        resp.update(ok=True, result=fn(req.get("payload", {})))
    except Exception as e:  # honest failure envelope, never a crash on the wire
        resp.update(ok=False, error={"code": "analyze_error", "message": str(e)})
    return resp


def main_cli(handlers: Mapping[str, Callable[[dict], dict]],
             framework_id: str, importer_id: str, *,
             spi_version: int = SPI_VERSION,
             stdin: TextIO | None = None,
             stdout: TextIO | None = None) -> int:
    """Run the JSON-stdio SPI loop: read request envelopes, write responses.

    Reads one JSON request per line from `stdin` (defaults to `sys.stdin`) and
    writes one JSON response per line to `stdout` (defaults to `sys.stdout`),
    flushing after each so a piping caller sees responses promptly. A line that
    is not valid JSON produces a `bad_json` error envelope; blank lines are
    skipped. Returns 0 at EOF.

    `framework_id` / `importer_id` are the importer's provenance strings (see
    module docstring). They are part of the SPI handshake the importer owns;
    the envelope framing here is id-agnostic, so they are accepted for API
    completeness and left for the importer's own verb handlers to embed in
    their results.
    """
    del framework_id, importer_id  # importer-owned provenance; used by handlers
    ins = stdin if stdin is not None else sys.stdin
    outs = stdout if stdout is not None else sys.stdout
    for line in ins:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            outs.write(json.dumps(
                {"spi_version": spi_version, "id": "", "ok": False,
                 "error": {"code": "bad_json", "message": str(e)}}) + "\n")
            outs.flush()
            continue
        outs.write(json.dumps(handle(req, handlers, spi_version)) + "\n")
        outs.flush()
    return 0
