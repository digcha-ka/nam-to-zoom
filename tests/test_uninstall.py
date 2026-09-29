"""Offline checks for the guarded bank-removal orchestration."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from nam2zoom import deploy  # noqa: E402


class UninstallTests(unittest.TestCase):
    @staticmethod
    def installed_bank(session):
        files = session / "backup" / "files"
        files.mkdir(parents=True)
        (files / deploy.BANK_NAME).write_bytes(b"effect")
        (files / deploy.BANK_ICON).write_bytes(b"icon")
        (files / "FLST_SEQ.ZT2").write_bytes(b"before")
        return True

    def test_absent_bank_never_invokes_uninstall(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session"
            with patch.object(deploy, "_tool", return_value="Autosave OFF - CONFIRMED; CRC ok") as tool, \
                 patch.object(deploy, "require_stock_patch"), \
                 patch.object(deploy, "_run"), \
                 patch.object(deploy, "plan_backup", return_value=False):
                self.assertEqual(deploy.uninstall_bank(session), "not-installed")
                self.assertEqual([call.args[0] for call in tool.call_args_list],
                                 ["safe_connect.py"])

    def test_removal_dry_runs_then_verifies_files_and_list(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session"
            flst = Mock()
            flst.expect_single_remove.return_value = (True, [])
            with patch.object(deploy, "_tool", return_value="Autosave OFF - CONFIRMED; CRC ok") as tool, \
                 patch.object(deploy, "require_stock_patch"), \
                 patch.object(deploy, "_run"), \
                 patch.object(deploy, "plan_backup",
                              side_effect=lambda *_: self.installed_bank(session)), \
                 patch.object(deploy, "_read_effect_list", return_value=b"after"), \
                 patch.object(deploy, "_flst", return_value=flst):
                self.assertEqual(deploy.uninstall_bank(session), "uninstalled")
            commands = [call.args for call in tool.call_args_list]
            self.assertEqual(commands[1][:2], ("pedal_diy.py", "uninstall"))
            self.assertEqual(commands[1][-1], "--dry-run")
            self.assertEqual(commands[2][:2], ("pedal_diy.py", "uninstall"))
            self.assertNotIn("--dry-run", commands[2])
            self.assertEqual([call.kwargs.get("expected") for call in
                              tool.call_args_list[3:]], [1, 1])
            flst.expect_single_remove.assert_called_once_with(
                b"before", b"after", deploy.BANK_NAME)

    def test_failed_dry_run_prevents_write(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session"

            def tool(name, *args, **kwargs):
                if name == "safe_connect.py":
                    return "Autosave OFF - CONFIRMED; CRC ok"
                raise deploy.DeployError("dry run refused")

            with patch.object(deploy, "_tool", side_effect=tool) as invoked, \
                 patch.object(deploy, "require_stock_patch"), \
                 patch.object(deploy, "_run"), \
                 patch.object(deploy, "plan_backup",
                              side_effect=lambda *_: self.installed_bank(session)):
                with self.assertRaisesRegex(deploy.DeployError, "dry run refused"):
                    deploy.uninstall_bank(session)
                self.assertEqual(len(invoked.call_args_list), 2)


if __name__ == "__main__":
    unittest.main()
