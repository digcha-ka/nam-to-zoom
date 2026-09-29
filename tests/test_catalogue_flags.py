"""Offline regression tests for catalogue metadata set by Zoom's official tool."""

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STOMP = (ROOT / "backend/vendor/stomphacks" if (ROOT / "backend").is_dir()
                 else ROOT / ".tooling/stomphacks")
STOMP = Path(os.environ.get("NAM2ZOOM_STOMPHACKS", DEFAULT_STOMP))
sys.path.insert(0, str(STOMP / "tools"))
sys.path.insert(0, str(STOMP / "zoom-zt2"))

import flst_check
import zoomzt2


def catalogue():
    header = dict(name="BLANK.ZD2")
    groups = []
    for group in (1, 2, 7):
        effects = [dict(effect=f"FX{group}{index}.ZD2", version="1.10",
                        installed=index % 2, id=(group << 24) | (index + 1),
                        catalog_flags=flag)
                   for index, flag in enumerate((0, 1, 128, 255))]
        groups.append(dict(group=group, groupname=group, effects=effects))
    return flst_check.pad_to_native(zoomzt2.ZT2.build([header, groups]), 12506)


class CatalogueFlagsTests(unittest.TestCase):
    def test_metadata_round_trip(self):
        data = catalogue()
        valid, problems, entries = flst_check.validate_flst(data)
        self.assertTrue(valid, problems)
        self.assertEqual(len(entries), 12)
        rebuilt = flst_check.pad_to_native(
            zoomzt2.ZT2.build(zoomzt2.ZT2.parse(data)), len(data))
        self.assertEqual(rebuilt, data)

    def test_add_remove_preserves_all_original_bytes(self):
        data = catalogue()
        editor = zoomzt2.zoomzt2()
        added = flst_check.pad_to_native(
            editor.add_effect(data, "N2ZBANK.ZD2", "1.00", 0x07000F87), len(data))
        self.assertEqual(flst_check.expect_single_add(data, added, "N2ZBANK.ZD2"),
                         (True, []))
        parsed = zoomzt2.ZT2.parse(added)
        new = [e for g in parsed[1] for e in g.effects if e.effect == "N2ZBANK.ZD2"]
        self.assertEqual(new[0].catalog_flags, 0)
        removed = flst_check.pad_to_native(
            editor.remove_effect(added, "N2ZBANK.ZD2"), len(data))
        self.assertEqual(flst_check.expect_single_remove(added, removed, "N2ZBANK.ZD2"),
                         (True, []))
        self.assertEqual(removed, data)

    def test_invalid_structures_still_refused(self):
        data = catalogue()
        for offset in (0, 78, 104 + 12, 104 + 19 + 3, 104 + 23):
            with self.subTest(offset=offset):
                broken = bytearray(data)
                broken[offset] ^= 0x7F
                self.assertFalse(flst_check.validate_flst(broken)[0])
        broken = bytearray(data)
        broken[-1] = 1
        self.assertFalse(flst_check.validate_flst(broken)[0])
        self.assertFalse(flst_check.validate_flst(data[:8000])[0])


if __name__ == "__main__":
    unittest.main()
