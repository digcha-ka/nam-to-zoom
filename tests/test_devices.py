"""Device identity and deployment-profile checks."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from nam2zoom import deploy  # noqa: E402
from nam2zoom.devices import identify_profile, require_supported_device  # noqa: E402


class DeviceProfileTests(unittest.TestCase):
    def test_recognizes_reviewed_ms_plus_identities(self):
        ms50 = identify_profile(0x006E, 0x0023, "1.40")
        ms70 = identify_profile(0x006E, 0x0026, "1.20")
        self.assertEqual(ms50.name, "Zoom MS-50G+")
        self.assertTrue(ms50.bank_hardware_tested)
        self.assertEqual(ms70.name, "Zoom MS-70CDR+")
        self.assertFalse(ms70.bank_hardware_tested)
        self.assertEqual(ms70.patch_count, 100)

    def test_refuses_wrong_model_or_firmware(self):
        self.assertIsNone(identify_profile(0x006E, 0x0026, "1.10"))
        with self.assertRaisesRegex(ValueError, "unsupported pedal identity"):
            require_supported_device(0x006E, 0x0028, "1.20")

    def test_live_identity_parser_selects_ms70cdr_plus(self):
        output = (
            "Device ID: 0x6e\n"
            "Family: 0x006e\n"
            "Model: 0x0026\n"
            "Version: 1.20\n"
        )
        profile = deploy._profile_from_identify(output)
        self.assertEqual(profile.key, "ms70cdr-plus")

    def test_live_identity_parser_refuses_unknown_firmware(self):
        output = "Family: 0x006e\nModel: 0x0026\nVersion: 9.99\n"
        with self.assertRaisesRegex(deploy.DeployError, "unsupported pedal identity"):
            deploy._profile_from_identify(output)


if __name__ == "__main__":
    unittest.main()
