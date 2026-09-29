"""Small standard-library ZD2 parser used by the initial nam2zoom tools.

The parser is intentionally conservative: it understands the container enough
to inspect and round-trip files, while preserving undocumented bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import struct
import zlib
from pathlib import Path
from typing import Any


KNOWN_CHUNKS = {"ICON", "TXJ1", "TXE1", "INFO", "DATA", "INF2", "CCOE", "PRMJ", "PRME"}
TRAILER_SIZE = 16
FIXED_PREFIX_SIZE = 12 + 4 + 73 + 4 + 2 + 1 + 4


class ZD2Error(ValueError):
    pass


@dataclass
class ZD2Chunk:
    tag: str
    offset: int
    length: int
    data: bytes
    known: bool


@dataclass
class ZD2File:
    path: Path | None
    data: bytes
    length: int
    checksum: int
    target: int
    unknown_header: bytes
    version: str
    group: int
    effect_id: int
    name: str
    raw_name: bytes
    group_name: str
    raw_group_name: bytes
    unknown4: bytes
    chunks: list[ZD2Chunk]
    trailer: bytes

    @property
    def computed_checksum(self) -> int:
        return compute_checksum(self.data)

    @property
    def checksum_valid(self) -> bool:
        return self.checksum == self.computed_checksum

    @property
    def data_chunk(self) -> ZD2Chunk | None:
        return next((chunk for chunk in self.chunks if chunk.tag == "DATA"), None)

    @property
    def dspload(self) -> float | None:
        info = next((chunk for chunk in self.chunks if chunk.tag == "INFO"), None)
        if info is None or info.length < 4:
            return None
        return struct.unpack("<f", info.data[-4:])[0]


def compute_checksum(data: bytes) -> int:
    if len(data) < 28:
        raise ZD2Error("ZD2 data is too short to contain checksum range and trailer")
    return zlib.crc32(data[12:-TRAILER_SIZE]) ^ 0xFFFFFFFF


def parse_zd2(path: str | Path) -> ZD2File:
    zd2_path = Path(path)
    return parse_zd2_bytes(zd2_path.read_bytes(), zd2_path)


def parse_zd2_bytes(data: bytes, path: Path | None = None) -> ZD2File:
    if len(data) < FIXED_PREFIX_SIZE + 22 + TRAILER_SIZE:
        raise ZD2Error("File is too small to be a ZD2 container")
    if data[:4] != b"ZDLF":
        raise ZD2Error("Missing ZDLF magic")

    length, checksum, target = struct.unpack_from("<III", data, 4)
    unknown_header = data[16:89]
    version = _decode_ascii(data[89:93])
    if data[93:95] != b"\x00\x00":
        raise ZD2Error("Expected zero padding after version")
    group = data[95]
    effect_id = struct.unpack_from("<I", data, 96)[0]

    cursor = 100
    raw_name = data[cursor : cursor + 11]
    name = _decode_c_string(raw_name)
    cursor += 11

    raw_group_name = data[cursor : cursor + 11]
    group_name = _decode_c_string(raw_group_name)
    cursor += 11

    unknown4 = data[cursor : cursor + 3]
    cursor += 3
    if data[cursor : cursor + 3] != b"\x00\x00\x00":
        raise ZD2Error("Expected zero padding after group metadata")
    cursor += 3

    chunks: list[ZD2Chunk] = []
    chunk_end = len(data) - TRAILER_SIZE
    while cursor < chunk_end:
        if cursor + 8 > chunk_end:
            raise ZD2Error(f"Truncated chunk header at offset 0x{cursor:x}")
        tag_bytes = data[cursor : cursor + 4]
        try:
            tag = tag_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ZD2Error(f"Non-ASCII chunk tag at offset 0x{cursor:x}") from exc
        length_value = struct.unpack_from("<I", data, cursor + 4)[0]
        payload_start = cursor + 8
        payload_end = payload_start + length_value
        if payload_end > chunk_end:
            raise ZD2Error(f"Chunk {tag!r} overruns trailer")
        chunks.append(
            ZD2Chunk(
                tag=tag,
                offset=cursor,
                length=length_value,
                data=data[payload_start:payload_end],
                known=tag in KNOWN_CHUNKS,
            )
        )
        cursor = payload_end

    return ZD2File(
        path=path,
        data=data,
        length=length,
        checksum=checksum,
        target=target,
        unknown_header=unknown_header,
        version=version,
        group=group,
        effect_id=effect_id,
        name=name,
        raw_name=raw_name,
        group_name=group_name,
        raw_group_name=raw_group_name,
        unknown4=unknown4,
        chunks=chunks,
        trailer=data[-TRAILER_SIZE:],
    )


def build_zd2_from_manifest(directory: str | Path) -> bytes:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    out = bytearray()
    out += b"ZDLF"
    out += struct.pack("<I", int(manifest["length"]))
    out += b"\x00\x00\x00\x00"
    out += bytes.fromhex(manifest["target_hex"])
    out += bytes.fromhex(manifest["unknown_header_hex"])
    out += _encode_fixed_ascii(manifest["version"], 4)
    out += b"\x00\x00"
    out += bytes([int(manifest["group"])])
    out += struct.pack("<I", int(manifest["effect_id"]))
    out += bytes.fromhex(manifest["raw_name_hex"])
    out += bytes.fromhex(manifest["raw_group_name_hex"])
    out += bytes.fromhex(manifest["unknown4_hex"])
    out += b"\x00\x00\x00"

    for chunk in manifest["chunks"]:
        tag = chunk["tag"]
        if len(tag) != 4:
            raise ZD2Error(f"Chunk tag must be four ASCII characters: {tag!r}")
        payload = (root / chunk["file"]).read_bytes()
        out += tag.encode("ascii")
        out += struct.pack("<I", len(payload))
        out += payload

    out += bytes.fromhex(manifest["trailer_hex"])

    checksum = compute_checksum(out)
    struct.pack_into("<I", out, 8, checksum)
    return bytes(out)


def manifest_for(zd2: ZD2File) -> dict[str, Any]:
    return {
        "format": "nam2zoom.zd2.extract.v1",
        "source": str(zd2.path) if zd2.path else None,
        "length": zd2.length,
        "checksum": f"0x{zd2.checksum:08x}",
        "computed_checksum": f"0x{zd2.computed_checksum:08x}",
        "checksum_valid": zd2.checksum_valid,
        "target": zd2.target,
        "target_hex": zd2.target.to_bytes(4, "little").hex(),
        "unknown_header_hex": zd2.unknown_header.hex(),
        "version": zd2.version,
        "group": zd2.group,
        "effect_id": zd2.effect_id,
        "effect_id_hex": f"0x{zd2.effect_id:08x}",
        "name": zd2.name,
        "raw_name_hex": zd2.raw_name.hex(),
        "group_name": zd2.group_name,
        "raw_group_name_hex": zd2.raw_group_name.hex(),
        "unknown4_hex": zd2.unknown4.hex(),
        "dspload": zd2.dspload,
        "data_size": zd2.data_chunk.length if zd2.data_chunk else None,
        "chunks": [
            {
                "tag": chunk.tag,
                "offset": chunk.offset,
                "length": chunk.length,
                "known": chunk.known,
                "file": f"{index:02d}_{chunk.tag}.bin",
            }
            for index, chunk in enumerate(zd2.chunks)
        ],
        "trailer_hex": zd2.trailer.hex(),
    }


def target_labels(target: int) -> list[str]:
    labels = {
        0x0001: "g-series",
        0x0008: "seen on B6/G6/G11",
        0x0020: "seen on B2 Four",
        0x0080: "ms-50g+/ms-60b+/ms-70cdr+",
        0x0100: "ms-200d+",
        0x0200: "ms-80ir+",
    }
    return [label for bit, label in labels.items() if target & bit]


def _decode_ascii(raw: bytes) -> str:
    return raw.rstrip(b"\x00").decode("ascii", errors="replace")


def _decode_c_string(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")


def _encode_fixed_ascii(value: str, size: int) -> bytes:
    raw = value.encode("ascii")
    if len(raw) > size:
        raise ZD2Error(f"Value {value!r} is too long for {size} bytes")
    return raw.ljust(size, b"\x00")

