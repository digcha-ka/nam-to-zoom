"""Pure protocol helpers for read-only Zoom MS Plus operations.

MIDI libraries normally expose SysEx payloads without the leading F0 and
trailing F7 bytes. Every message accepted and returned here uses that form.
"""

from __future__ import annotations

from dataclasses import dataclass
import zlib


ZOOM_MANUFACTURER = 0x52
MS50G_PLUS_DEVICE = 0x6E
ZOOM_PREFIX = bytes((ZOOM_MANUFACTURER, 0x00, MS50G_PLUS_DEVICE))
IDENTITY_REQUEST = bytes((0x7E, 0x7F, 0x06, 0x01))


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Identity:
    device_id: int
    family_code: int
    model_number: int
    version: str


@dataclass(frozen=True)
class PatchInfo:
    count: int
    patch_size: int
    patches_per_bank: int


@dataclass(frozen=True)
class DataBlock:
    data: bytes
    checksum: int


def zoom_command(*body: int, device_id: int = MS50G_PLUS_DEVICE) -> bytes:
    payload = bytes((ZOOM_MANUFACTURER, 0x00, device_id, *body))
    _require_7bit(payload, "SysEx command")
    return payload


def pack_7bit(data: bytes) -> bytes:
    """Pack 8-bit bytes into groups of one high-bit mask plus seven bytes."""
    result = bytearray()
    for offset in range(0, len(data), 7):
        group = data[offset : offset + 7]
        mask = 0
        low = bytearray()
        for index, value in enumerate(group):
            if value & 0x80:
                mask |= 1 << (6 - index)
            low.append(value & 0x7F)
        result.append(mask)
        result.extend(low)
    return bytes(result)


def unpack_7bit(packet: bytes, expected_length: int | None = None) -> bytes:
    result = bytearray()
    cursor = 0
    while cursor < len(packet):
        mask = packet[cursor]
        cursor += 1
        group_size = min(7, len(packet) - cursor)
        for index in range(group_size):
            value = packet[cursor + index]
            if mask & (1 << (6 - index)):
                value |= 0x80
            result.append(value)
        cursor += group_size
    if expected_length is not None:
        if len(result) < expected_length:
            raise ProtocolError(
                f"7-bit payload decoded to {len(result)} bytes, expected {expected_length}"
            )
        return bytes(result[:expected_length])
    return bytes(result)


def decode_crc_7bit(raw: bytes) -> int:
    if len(raw) != 5:
        raise ProtocolError(f"CRC must contain five 7-bit bytes, got {len(raw)}")
    _require_7bit(raw, "CRC")
    return (
        raw[0]
        | (raw[1] << 7)
        | (raw[2] << 14)
        | (raw[3] << 21)
        | ((raw[4] & 0x0F) << 28)
    )


def parse_identity(payload: bytes) -> Identity:
    # Universal identity reply: 7E <device> 06 02 52 <family lo/hi>
    # <model lo/hi> <four ASCII version bytes>.
    if len(payload) < 13 or payload[0] != 0x7E or payload[2:5] != b"\x06\x02\x52":
        raise ProtocolError("Not a Zoom universal identity response")
    return Identity(
        device_id=payload[1],
        family_code=payload[5] | (payload[6] << 8),
        model_number=payload[7] | (payload[8] << 8),
        version=payload[9:13].decode("ascii", errors="replace"),
    )


def parse_patch_info(payload: bytes) -> PatchInfo:
    _require_zoom_response(payload, 0x43, minimum=12)
    info = PatchInfo(
        count=_u14(payload[4], payload[5]),
        patch_size=_u14(payload[6], payload[7]),
        patches_per_bank=_u14(payload[10], payload[11]),
    )
    if info.count <= 0 or info.patch_size <= 0 or info.patches_per_bank <= 0:
        raise ProtocolError(f"Invalid patch information: {info}")
    return info


def patch_download_request(location: int, info: PatchInfo) -> bytes:
    if not 1 <= location <= info.count:
        raise ProtocolError(f"Patch location must be between 1 and {info.count}")
    bank, program = divmod(location - 1, info.patches_per_bank)
    return zoom_command(
        0x46,
        0x00,
        0x00,
        bank & 0x7F,
        (bank >> 7) & 0x7F,
        program & 0x7F,
        (program >> 7) & 0x7F,
    )


def parse_patch_dump(payload: bytes) -> DataBlock:
    _require_zoom_response(payload, 0x45, minimum=17)
    length = _u14(payload[10], payload[11])
    return _parse_packed_block(payload, packed_offset=12, length=length)


def parse_file_listing(payload: bytes) -> str | None:
    _require_file_response(payload, minimum=5)
    if payload[4] != 0x04:
        return None
    if len(payload) < 15:
        raise ProtocolError("File listing response is truncated")
    # MS Plus directory replies reserve 12 bytes for an 8.3-style basename.
    # Byte 26 follows the filename and can be a nonzero status/attribute byte.
    raw = payload[14:26].split(b"\x00", 1)[0]
    if not raw:
        return None
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ProtocolError("File listing contains a non-ASCII filename") from exc


def parse_file_block(payload: bytes) -> DataBlock | None:
    _require_file_response(payload, minimum=10)
    if payload[4] != 0x04:
        return None
    length = _u14(payload[8], payload[9])
    if length == 0:
        return None
    return _parse_packed_block(payload, packed_offset=10, length=length)


def filename_request(operation: int, filename: str, *prefix: int) -> bytes:
    raw = filename.encode("ascii")
    if not raw or b"\x00" in raw or len(raw) > 64:
        raise ProtocolError("Filename must be 1-64 ASCII characters without NUL")
    if "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise ProtocolError("Only a device basename is allowed")
    return zoom_command(0x60, operation, *prefix) + raw + b"\x00"


def _parse_packed_block(payload: bytes, packed_offset: int, length: int) -> DataBlock:
    packed_length = length + ((length + 6) // 7)
    packed_end = packed_offset + packed_length
    crc_end = packed_end + 5
    if len(payload) < crc_end:
        raise ProtocolError(
            f"Packed response is truncated: need {crc_end} bytes, got {len(payload)}"
        )
    data = unpack_7bit(payload[packed_offset:packed_end], expected_length=length)
    checksum = decode_crc_7bit(payload[-5:])
    expected = zlib.crc32(data) ^ 0xFFFFFFFF
    if checksum != expected:
        raise ProtocolError(
            f"CRC mismatch: response 0x{checksum:08x}, computed 0x{expected:08x}"
        )
    return DataBlock(data=data, checksum=checksum)


def _require_zoom_response(payload: bytes, command: int, minimum: int) -> None:
    if len(payload) < minimum:
        raise ProtocolError(f"Response is too short: expected at least {minimum} bytes")
    if payload[:3] != ZOOM_PREFIX or payload[3] != command:
        raise ProtocolError(
            f"Unexpected response prefix {payload[:4].hex(' ')}, "
            f"expected {(ZOOM_PREFIX + bytes((command,))).hex(' ')}"
        )
    _require_7bit(payload, "SysEx response")


def _require_file_response(payload: bytes, minimum: int) -> None:
    _require_zoom_response(payload, 0x60, minimum)


def _require_7bit(payload: bytes, label: str) -> None:
    if any(value > 0x7F for value in payload):
        raise ProtocolError(f"{label} contains an invalid non-7-bit byte")


def _u14(low: int, high: int) -> int:
    return low | (high << 7)
