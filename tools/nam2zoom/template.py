"""Fill model data in a verified precompiled bank; never modify DSP instructions."""

import hashlib
import json
from pathlib import Path
import struct

from elf32 import parse_elf32
from offline_effect_audit import audit_bytes
from zd2 import compute_checksum, parse_zd2_bytes

from .compact import expected_parameters

WORDS = expected_parameters(3)
SOURCES = ("dsp/nam_a2_bank_zd2/bank_effect.c", "dsp/nam_a2_compact/compact_pair.c",
           "dsp/nam_a2_compact/compact_pair.h", "dsp/nam_a2_compact/compact_kernel.h")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fill_template(templates: Path, manifest: Path) -> Path:
    config = json.loads(manifest.read_text(encoding="ascii"))
    params = config["params"]
    count = config["const_blob"]["words"] // WORDS
    if not 1 <= count <= 5 or config["const_blob"]["words"] != count * WORDS:
        raise ValueError("invalid template model count")
    index = json.loads((templates / "index.json").read_text(encoding="ascii"))
    if index.get("format") != "nam2zoom.templates.v1":
        raise ValueError("unsupported bank template format")
    root = Path(__file__).resolve().parents[2]
    if index.get("dsp_source_sha256") != {name: digest((root / name).read_bytes()) for name in SOURCES}:
        raise ValueError("bank templates do not match the current DSP sources")
    record = index["banks"][str(count)]
    raw = (templates / f"bank-{count}.ZD2").read_bytes()
    icon = (templates / "N2ZBANK.ZIC").read_bytes()
    if digest(raw) != record["sha256"] or digest(icon) != index["icon_sha256"]:
        raise ValueError("bank template or icon hash mismatch")
    expected_params = ["Model", "Bass", "Mid", "Treble", "Vol", "Input", "Mix"]
    if ([p["name"] for p in params] != expected_params or config["dspload"] != 150
            or [p.get("default") for p in params] != [0, 50, 50, 50, 50, 50, 100]
            or [p.get("max", len(p.get("values", [])) - 1) for p in params]
            != [max(1, count - 1)] + [100] * 6):
        raise ValueError("manifest does not match the compiled template controls")
    zd2 = parse_zd2_bytes(raw)
    elf = parse_elf32(zd2.data_chunk.data)
    const = next(section for section in elf.sections if section.name == ".const")
    base = zd2.data_chunk.offset + 8 + const.offset
    labels = params[0]["values"]
    if len(labels) != max(2, count):
        raise ValueError("selector does not match template capacity")
    weights = (manifest.parent / config["const_blob"]["init"]).read_bytes()
    if len(weights) != count * WORDS * 4:
        raise ValueError("template weight size mismatch")
    if any(not (-float("inf") < value < float("inf"))
           for value in struct.unpack(f"<{count * WORDS}f", weights)):
        raise ValueError("template weights must be finite")
    labels_data = b"".join(label.encode("ascii").ljust(8, b"\0") for label in labels)
    if any(not 1 <= len(label) <= 5 for label in labels):
        raise ValueError("invalid template selector label")
    output = bytearray(raw)
    ranges = [(record["weights_offset"], weights), (record["labels_offset"], labels_data)]
    spans = []
    for offset, payload in ranges:
        if not isinstance(offset, int) or not base <= offset < offset + len(payload) <= base + const.size:
            raise ValueError("template patch range escapes the constant section")
        spans.append((offset, offset + len(payload)))
        output[offset:offset + len(payload)] = payload
    if max(spans[0][0], spans[1][0]) < min(spans[0][1], spans[1][1]):
        raise ValueError("template patch ranges overlap")
    struct.pack_into("<I", output, 8, compute_checksum(output))
    errors = audit_bytes(bytes(output), icon, {})
    if errors:
        raise ValueError("filled bank failed audit: " + "; ".join(errors))
    # This check also covers relocations and every non-model byte of the ELF.
    patched_elf = parse_zd2_bytes(bytes(output)).data_chunk.data
    for section in elf.sections:
        if section.name != ".const" and patched_elf[section.offset:section.offset + section.size] != elf.data[section.offset:section.offset + section.size]:
            raise ValueError("template patch modified a non-constant section")
    build = manifest.parent / "build"
    build.mkdir(exist_ok=False)
    effect = build / "N2ZBANK.ZD2"
    effect.write_bytes(output)
    effect.with_suffix(".ZIC").write_bytes(icon)
    print(f"Filled precompiled {count}-model template; DSP code unchanged", flush=True)
    return effect
