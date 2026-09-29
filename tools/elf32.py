"""Minimal, bounds-checked ELF32 reader for factory Zoom DSP payloads."""

from __future__ import annotations

from dataclasses import dataclass
import struct


class ELFError(ValueError):
    pass


@dataclass(frozen=True)
class ELFHeader:
    osabi: int
    abi_version: int
    elf_type: int
    machine: int
    version: int
    entry: int
    program_offset: int
    section_offset: int
    flags: int
    header_size: int
    program_entry_size: int
    program_count: int
    section_entry_size: int
    section_count: int
    section_name_index: int


@dataclass(frozen=True)
class ProgramHeader:
    index: int
    type: int
    offset: int
    virtual_address: int
    physical_address: int
    file_size: int
    memory_size: int
    flags: int
    alignment: int


@dataclass(frozen=True)
class Section:
    index: int
    name: str
    type: int
    flags: int
    address: int
    offset: int
    size: int
    link: int
    info: int
    alignment: int
    entry_size: int


@dataclass(frozen=True)
class Symbol:
    table: str
    index: int
    name: str
    value: int
    size: int
    binding: int
    type: int
    visibility: int
    section_index: int


@dataclass(frozen=True)
class Relocation:
    section: str
    offset: int
    symbol_index: int
    type: int
    addend: int | None


@dataclass(frozen=True)
class DynamicEntry:
    tag: int
    value: int


@dataclass(frozen=True)
class ELFFile:
    data: bytes
    header: ELFHeader
    programs: list[ProgramHeader]
    sections: list[Section]
    symbols: list[Symbol]
    relocations: list[Relocation]
    dynamic_entries: list[DynamicEntry]

    @property
    def dynamic_symbols(self) -> list[Symbol]:
        return [symbol for symbol in self.symbols if symbol.table == ".dynsym"]

    @property
    def imports(self) -> list[Symbol]:
        return [symbol for symbol in self.dynamic_symbols if symbol.name and symbol.section_index == 0]

    @property
    def exports(self) -> list[Symbol]:
        return [
            symbol
            for symbol in self.dynamic_symbols
            if symbol.name and symbol.section_index != 0 and symbol.binding in (1, 2)
        ]


def parse_elf32(data: bytes) -> ELFFile:
    if len(data) < 52 or data[:4] != b"\x7fELF":
        raise ELFError("Missing ELF magic or truncated ELF32 header")
    if data[4] != 1:
        raise ELFError(f"Expected ELF32 class, got {data[4]}")
    if data[5] != 1:
        raise ELFError(f"Expected little-endian ELF data, got {data[5]}")
    if data[6] != 1:
        raise ELFError(f"Unsupported ELF identification version {data[6]}")

    fields = _unpack_from("<HHIIIIIHHHHHH", data, 16, "ELF header")
    header = ELFHeader(
        osabi=data[7],
        abi_version=data[8],
        elf_type=fields[0],
        machine=fields[1],
        version=fields[2],
        entry=fields[3],
        program_offset=fields[4],
        section_offset=fields[5],
        flags=fields[6],
        header_size=fields[7],
        program_entry_size=fields[8],
        program_count=fields[9],
        section_entry_size=fields[10],
        section_count=fields[11],
        section_name_index=fields[12],
    )
    if header.header_size != 52:
        raise ELFError(f"Unexpected ELF32 header size {header.header_size}")
    if header.section_entry_size != 40:
        raise ELFError(f"Unexpected ELF32 section entry size {header.section_entry_size}")
    if header.program_count and header.program_entry_size != 32:
        raise ELFError(f"Unexpected ELF32 program entry size {header.program_entry_size}")

    raw_sections = [
        _unpack_from(
            "<IIIIIIIIII",
            data,
            header.section_offset + index * header.section_entry_size,
            f"section header {index}",
        )
        for index in range(header.section_count)
    ]
    if not 0 <= header.section_name_index < len(raw_sections):
        raise ELFError("Section name table index is out of range")
    name_header = raw_sections[header.section_name_index]
    section_names = _slice(data, name_header[4], name_header[5], "section name table")

    sections = [
        Section(
            index=index,
            name=_string_at(section_names, raw[0]),
            type=raw[1],
            flags=raw[2],
            address=raw[3],
            offset=raw[4],
            size=raw[5],
            link=raw[6],
            info=raw[7],
            alignment=raw[8],
            entry_size=raw[9],
        )
        for index, raw in enumerate(raw_sections)
    ]

    programs = [
        ProgramHeader(index, *_unpack_from(
            "<IIIIIIII",
            data,
            header.program_offset + index * header.program_entry_size,
            f"program header {index}",
        ))
        for index in range(header.program_count)
    ]

    symbols: list[Symbol] = []
    for section in sections:
        if section.type not in (2, 11):
            continue
        if section.entry_size != 16 or not 0 <= section.link < len(sections):
            raise ELFError(f"Invalid symbol table metadata in {section.name}")
        strings_header = sections[section.link]
        strings = _slice(data, strings_header.offset, strings_header.size, strings_header.name)
        for index in range(section.size // section.entry_size):
            raw = _unpack_from(
                "<IIIBBH",
                data,
                section.offset + index * section.entry_size,
                f"symbol {index} in {section.name}",
            )
            symbols.append(
                Symbol(
                    table=section.name,
                    index=index,
                    name=_string_at(strings, raw[0]),
                    value=raw[1],
                    size=raw[2],
                    binding=raw[3] >> 4,
                    type=raw[3] & 0x0F,
                    visibility=raw[4] & 0x03,
                    section_index=raw[5],
                )
            )

    relocations: list[Relocation] = []
    for section in sections:
        if section.type not in (4, 9):
            continue
        expected_size = 12 if section.type == 4 else 8
        if section.entry_size != expected_size:
            raise ELFError(f"Invalid relocation entry size in {section.name}")
        for index in range(section.size // section.entry_size):
            offset = section.offset + index * section.entry_size
            if section.type == 4:
                reloc_offset, info, addend = _unpack_from(
                    "<IIi", data, offset, f"relocation {index} in {section.name}"
                )
            else:
                reloc_offset, info = _unpack_from(
                    "<II", data, offset, f"relocation {index} in {section.name}"
                )
                addend = None
            relocations.append(
                Relocation(
                    section=section.name,
                    offset=reloc_offset,
                    symbol_index=info >> 8,
                    type=info & 0xFF,
                    addend=addend,
                )
            )

    dynamic_entries: list[DynamicEntry] = []
    for section in sections:
        if section.type != 6:
            continue
        if section.entry_size != 8:
            raise ELFError(f"Invalid dynamic entry size in {section.name}")
        for index in range(section.size // section.entry_size):
            tag, value = _unpack_from(
                "<iI",
                data,
                section.offset + index * section.entry_size,
                f"dynamic entry {index}",
            )
            dynamic_entries.append(DynamicEntry(tag=tag, value=value))

    return ELFFile(data, header, programs, sections, symbols, relocations, dynamic_entries)


def _unpack_from(fmt: str, data: bytes, offset: int, label: str) -> tuple:
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise ELFError(f"{label} exceeds the ELF payload")
    return struct.unpack_from(fmt, data, offset)


def _slice(data: bytes, offset: int, size: int, label: str) -> bytes:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ELFError(f"{label} exceeds the ELF payload")
    return data[offset : offset + size]


def _string_at(table: bytes, offset: int) -> str:
    if offset == 0:
        return ""
    if not 0 <= offset < len(table):
        raise ELFError(f"String offset {offset} is outside its table")
    end = table.find(b"\x00", offset)
    if end < 0:
        raise ELFError("Unterminated string table entry")
    return table[offset:end].decode("ascii", errors="replace")
