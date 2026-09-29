"""MIDI transport and read-only operations for Zoom MS Plus pedals."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import queue
import time
from typing import Callable

from msplus_protocol import (
    IDENTITY_REQUEST,
    MS_PLUS_DEVICE,
    ZOOM_PREFIX,
    DataBlock,
    Identity,
    PatchInfo,
    ProtocolError,
    filename_request,
    parse_file_block,
    parse_file_listing,
    parse_identity,
    parse_patch_dump,
    parse_patch_info,
    patch_download_request,
    zoom_command,
)


PORT_MARKERS = ("ZOOM MS Plus Series", "ZOOM G")


class MidiUnavailable(RuntimeError):
    pass


class MidiTimeout(TimeoutError):
    pass


@dataclass(frozen=True)
class PortPair:
    input_name: str
    output_name: str


def load_mido():
    try:
        import mido
    except ImportError as exc:
        raise MidiUnavailable(
            "MIDI support is not installed. Run: "
            "python -m pip install -r requirements-midi.txt"
        ) from exc
    return mido


def list_ports() -> tuple[list[str], list[str]]:
    mido = load_mido()
    return list(mido.get_input_names()), list(mido.get_output_names())


def choose_ports(
    input_names: list[str],
    output_names: list[str],
    input_name: str | None = None,
    output_name: str | None = None,
    index: int = 0,
) -> PortPair:
    inputs = [input_name] if input_name else _candidate_ports(input_names)
    outputs = [output_name] if output_name else _candidate_ports(output_names)
    if input_name and input_name not in input_names:
        raise MidiUnavailable(f"MIDI input not found: {input_name}")
    if output_name and output_name not in output_names:
        raise MidiUnavailable(f"MIDI output not found: {output_name}")
    if index < 0 or index >= len(inputs) or index >= len(outputs):
        raise MidiUnavailable(
            f"No matching Zoom MIDI port pair at index {index}; "
            f"found {len(inputs)} input(s) and {len(outputs)} output(s)"
        )
    return PortPair(inputs[index], outputs[index])


class MidiExchange(AbstractContextManager["MidiExchange"]):
    def __init__(self, ports: PortPair, timeout: float = 2.0):
        self.ports = ports
        self.timeout = timeout
        self._messages: queue.Queue[bytes] = queue.Queue()
        self._input = None
        self._output = None

    def __enter__(self) -> "MidiExchange":
        mido = load_mido()

        def receive(message) -> None:
            if message.type == "sysex":
                self._messages.put(bytes(message.data))

        self._input = mido.open_input(self.ports.input_name, callback=receive)
        try:
            self._output = mido.open_output(self.ports.output_name)
        except Exception:
            self._input.close()
            self._input = None
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._output is not None:
            self._output.close()
        if self._input is not None:
            self._input.close()
        self._output = None
        self._input = None

    def exchange(
        self,
        payload: bytes,
        accept: Callable[[bytes], bool] | None = None,
        timeout: float | None = None,
    ) -> bytes:
        if self._output is None:
            raise MidiUnavailable("MIDI ports are not open")
        self._discard_pending()
        mido = load_mido()
        self._output.send(mido.Message("sysex", data=payload))
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MidiTimeout(f"Timed out waiting for response to {payload.hex(' ')}")
            try:
                response = self._messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise MidiTimeout(
                    f"Timed out waiting for response to {payload.hex(' ')}"
                ) from exc
            if accept is None or accept(response):
                return response

    def _discard_pending(self) -> None:
        while True:
            try:
                self._messages.get_nowait()
            except queue.Empty:
                return


class ReadOnlyMSPlus(AbstractContextManager["ReadOnlyMSPlus"]):
    """High-level operations with no upload, delete, or mutation methods."""

    def __init__(self, transport: MidiExchange, device_id: int = MS_PLUS_DEVICE):
        self.transport = transport
        self.device_id = device_id
        self.pc_mode = False

    def __enter__(self) -> "ReadOnlyMSPlus":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.pc_mode:
            try:
                self.disable_pc_mode()
            except Exception:
                if exc is None:
                    raise

    def identify(self) -> Identity:
        response = self.transport.exchange(
            IDENTITY_REQUEST,
            lambda data: len(data) >= 5 and data[0] == 0x7E and data[2:5] == b"\x06\x02\x52",
        )
        return parse_identity(response)

    def enable_pc_mode(self) -> None:
        # Once the command is sent the pedal may have changed mode even if its
        # acknowledgement is lost, so make context exit attempt restoration.
        self.pc_mode = True
        self.transport.exchange(
            zoom_command(0x52, device_id=self.device_id), self._is_zoom_response
        )

    def disable_pc_mode(self) -> None:
        self.transport.exchange(
            zoom_command(0x53, device_id=self.device_id), self._is_zoom_response
        )
        self.pc_mode = False

    def patch_info(self) -> PatchInfo:
        response = self.transport.exchange(
            zoom_command(0x44, device_id=self.device_id), self._command_response(0x43)
        )
        return parse_patch_info(response)

    def download_patch(self, location: int, info: PatchInfo) -> DataBlock:
        response = self.transport.exchange(
            patch_download_request(location, info), self._command_response(0x45)
        )
        block = parse_patch_dump(response)
        if len(block.data) != info.patch_size:
            raise ProtocolError(
                f"Patch {location} has {len(block.data)} bytes; "
                f"device advertised {info.patch_size}"
            )
        return block

    def list_files(self, maximum: int = 4096) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for index in range(maximum):
            operation = 0x25 if index == 0 else 0x26
            response = self.transport.exchange(
                filename_request(operation, "*", 0x00, 0x00), self._command_response(0x60)
            )
            name = parse_file_listing(response)
            if name is None:
                return names
            if name in seen:
                raise ProtocolError(f"File listing repeated {name!r}; refusing an endless loop")
            seen.add(name)
            names.append(name)
        raise ProtocolError(f"File listing exceeded the safety limit of {maximum} entries")

    def download_file(self, filename: str) -> bytes:
        request = filename_request(
            0x20,
            filename,
            0x02,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
        )
        # zoom-zt2 performs this open-for-read exchange twice. Preserve that
        # observed sequence until hardware captures establish its semantics.
        opened = False
        try:
            self.transport.exchange(request, self._command_response(0x60))
            opened = True
            self.transport.exchange(request, self._command_response(0x60))

            result = bytearray()
            while True:
                self.transport.exchange(
                    zoom_command(0x60, 0x05, 0x00, device_id=self.device_id),
                    self._command_response(0x60),
                )
                self.transport.exchange(
                    zoom_command(
                        0x60,
                        0x22,
                        0x14,
                        0x2F,
                        0x60,
                        0x00,
                        0x0C,
                        0x00,
                        0x04,
                        0x00,
                        0x00,
                        0x00,
                        device_id=self.device_id,
                    ),
                    self._command_response(0x60),
                )
                response = self.transport.exchange(
                    zoom_command(0x60, 0x05, 0x00, device_id=self.device_id),
                    self._command_response(0x60),
                )
                block = parse_file_block(response)
                if block is None:
                    break
                result.extend(block.data)
        finally:
            if opened:
                self._close_file()
        return bytes(result)

    def _close_file(self) -> None:
        try:
            self.transport.exchange(
                zoom_command(0x60, 0x21, 0x40, 0, 0, 0, 0, device_id=self.device_id),
                self._command_response(0x60),
            )
        finally:
            self.transport.exchange(
                zoom_command(0x60, 0x09, device_id=self.device_id),
                self._command_response(0x60),
            )

    def _is_zoom_response(self, payload: bytes) -> bool:
        return len(payload) >= 3 and payload[:3] == bytes((0x52, 0x00, self.device_id))

    def _command_response(self, command: int) -> Callable[[bytes], bool]:
        prefix = bytes((0x52, 0x00, self.device_id, command))
        return lambda payload: payload.startswith(prefix)


def open_default_device(
    input_name: str | None = None,
    output_name: str | None = None,
    index: int = 0,
    timeout: float = 2.0,
) -> MidiExchange:
    inputs, outputs = list_ports()
    return MidiExchange(
        choose_ports(inputs, outputs, input_name, output_name, index), timeout=timeout
    )


def _candidate_ports(names: list[str]) -> list[str]:
    return [name for name in names if any(marker in name for marker in PORT_MARKERS)]
