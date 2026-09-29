"""Check the offline report does not hide helper frames behind local metrics."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_compact_pair import function_metrics


class PairBenchmarkTests(unittest.TestCase):
    def test_call_and_callp_include_nested_frames(self):
        text = """
;* FUNCTION NAME: helper
;* Local Frame Size : 0 Args + 60 Auto + 0 Save = 60 byte
 LDW .D2T2 *SP(4), B0
;* FUNCTION NAME: pair
;* Local Frame Size : 0 Args + 48 Auto + 48 Save = 96 byte
 CALL .S2 helper
;* FUNCTION NAME: block
;* Local Frame Size : 0 Args + 88 Auto + 56 Save = 144 byte
 CALLP .S2 helper
"""
        rows = function_metrics(text)
        self.assertEqual(rows["helper"]["static_sp_operand_occurrences"], 1)
        self.assertEqual(rows["pair"]["local_call_targets"], ["helper"])
        self.assertEqual(rows["pair"]["known_local_peak_frame_bytes"], 156)
        self.assertEqual(rows["block"]["known_local_peak_frame_bytes"], 204)

    def test_recursion_is_not_reported_as_bounded(self):
        with self.assertRaises(ValueError):
            function_metrics(";* FUNCTION NAME: loop\n CALL .S2 loop\n")


if __name__ == "__main__":
    unittest.main()
