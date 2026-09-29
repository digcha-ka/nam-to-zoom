"""Cab IR preprocessing and adaptation cache tests (training Python runtime)."""

import sys
import tempfile
import unittest
from unittest.mock import patch
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from nam2zoom.adapt import cache_key  # noqa: E402
from test_compact_shape import model  # noqa: E402

try:
    import numpy as np
    import soundfile as sf
    from nam2zoom.ir import MAX_TAPS, bake, fit_teacher_level, load_ir
except ImportError:
    np = sf = None


@unittest.skipIf(np is None, "IR tests require the training Python environment")
class IrTests(unittest.TestCase):
    def test_prepare_pair_bakes_ir_into_teacher(self):
        from nam2zoom.adapt import prepare_pair

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, di, ir = (root / name for name in ("amp.nam", "di.wav", "cab.wav"))
            source.write_text(json.dumps(model(3)), encoding="utf-8")
            dry = np.tile(np.array([0.0, 0.1, 0.0, -0.1], dtype="float32"),
                          44100 * 30 // 4)
            sf.write(di, dry, 44100, subtype="FLOAT")
            sf.write(ir, np.array([0.5, 0.25], dtype="float32"), 44100,
                     subtype="FLOAT")

            def render(command, **_):
                audio, rate = sf.read(command[-2], dtype="float32")
                sf.write(command[-1], audio, rate, subtype="FLOAT")

            with patch("nam2zoom.adapt._run", side_effect=render):
                dry_path, wet_path, _, level = prepare_pair(source, di, root, 44100, ir)
            actual_dry, _ = sf.read(dry_path, dtype="float32")
            actual_wet, _ = sf.read(wet_path, dtype="float32")
            expected = np.convolve(actual_dry[:32], [0.5, 0.25])[:32]
            np.testing.assert_allclose(actual_wet[:32], expected, atol=1e-6)
            self.assertEqual(level["ir_gain_db"], 0.0)

    def test_hot_ir_is_attenuated_without_clipping(self):
        audio = np.array([0.0, 4.4031, -2.0], dtype="float32")
        result, level = fit_teacher_level(audio)
        self.assertAlmostEqual(float(np.max(np.abs(result))), 0.95, places=5)
        self.assertLess(level["ir_gain_db"], -13.0)
        np.testing.assert_allclose(result / result[1], audio / audio[1], atol=1e-6)

    def test_resample_trim_and_bake(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cab.wav"
            impulse = np.zeros(4800, dtype="float32")
            impulse[48] = 0.5
            impulse[49] = 0.25
            sf.write(path, impulse, 48000, subtype="FLOAT")
            taps = load_ir(path)
            self.assertLess(len(taps), MAX_TAPS + 1)
            self.assertGreater(float(np.max(np.abs(taps[:24]))), 0.1)
            audio = np.zeros(100, dtype="float32")
            audio[10] = 1.0
            result = bake(audio, taps)
            np.testing.assert_allclose(result[10:20], taps[:10], atol=1e-6)
            self.assertEqual(len(result), len(audio))

    def test_reject_stereo_and_long_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cab.wav"
            sf.write(path, np.ones((128, 2), dtype="float32"), 44100)
            with self.assertRaisesRegex(ValueError, "mono"):
                load_ir(path)
            sf.write(path, np.ones(MAX_TAPS + 500, dtype="float32"), 44100)
            with self.assertRaisesRegex(ValueError, "energy beyond"):
                load_ir(path)

    def test_ir_content_changes_cache_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, di, ir = (root / name for name in ("amp.nam", "di.wav", "cab.wav"))
            source.write_bytes(b"source")
            di.write_bytes(b"di")
            ir.write_bytes(b"ir-one")
            plain = cache_key(source, di, 100)
            first = cache_key(source, di, 100, ir)
            ir.write_bytes(b"ir-two")
            self.assertNotEqual(first, cache_key(source, di, 100, ir))
            self.assertNotEqual(plain, first)
            self.assertEqual(plain, cache_key(source, di, 100))


if __name__ == "__main__":
    unittest.main()
