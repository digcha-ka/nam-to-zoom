"""Compare the production pair kernel with the scalar reference kernel."""

import _ctypes
import ctypes
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

from test_kernel_optimization import State, find_clang

ROOT = Path(__file__).resolve().parents[1]
DILATIONS = [1, 3, 7, 17, 41, 101, 239] * 2
SCALAR_FLOATS = 9948
PAIR_FLOATS = 5019


class KernelPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clang = find_clang()
        if not clang:
            raise unittest.SkipTest("host Clang compiler is unavailable")
        cls.temp = tempfile.TemporaryDirectory()
        path = Path(cls.temp.name) / ("pair.dll" if os.name == "nt" else "pair.so")
        command = [clang, "-O3", "-ffp-contract=off", "-DCOMPACT_CHANNELS=3",
                   "-DN2Z_OPTIMIZED_KERNEL", "-shared"]
        if os.name == "nt":
            command += ["-Wl,/export:compact_process", "-Wl,/export:compact_process_pair"]
        command += [str(ROOT / "dsp/nam_a2_compact/compact_kernel.c"),
                    str(ROOT / "dsp/nam_a2_compact/compact_pair.c"), "-o", str(path)]
        subprocess.run(command, check=True, capture_output=True)
        cls.library = ctypes.CDLL(str(path))
        pointer = ctypes.POINTER(ctypes.c_float)
        cls.scalar = cls.library.compact_process
        cls.scalar.argtypes = [pointer, pointer, ctypes.POINTER(State), ctypes.c_float]
        cls.scalar.restype = ctypes.c_float
        cls.pair = cls.library.compact_process_pair
        cls.pair.argtypes = [pointer, pointer, ctypes.POINTER(State), ctypes.c_float,
                            ctypes.c_float, pointer, pointer]
        cls.pair.restype = None

    @classmethod
    def tearDownClass(cls):
        del cls.scalar, cls.pair
        if os.name == "nt":
            _ctypes.FreeLibrary(cls.library._handle)
        del cls.library
        cls.temp.cleanup()

    def compare(self, weights, samples):
        self.assertEqual(len(samples) % 2, 0)
        w = (ctypes.c_float * 659)(*weights)
        # Sentinel floats surround the history supplied to each actual C kernel.
        scalar = (ctypes.c_float * (SCALAR_FLOATS + 2))()
        pair = (ctypes.c_float * (PAIR_FLOATS + 2))()
        scalar[0] = scalar[-1] = pair[0] = pair[-1] = 12345.0
        sp = ctypes.cast(ctypes.byref(scalar, 4), ctypes.POINTER(ctypes.c_float))
        pp = ctypes.cast(ctypes.byref(pair, 4), ctypes.POINTER(ctypes.c_float))
        ss, ps = State(), State()
        output = []
        for index in range(0, len(samples), 2):
            expected = [self.scalar(w, sp, ctypes.byref(ss), value)
                        for value in samples[index:index + 2]]
            y0, y1 = ctypes.c_float(), ctypes.c_float()
            self.pair(w, pp, ctypes.byref(ps), *samples[index:index + 2],
                      ctypes.byref(y0), ctypes.byref(y1))
            self.assertEqual(bytes(y0) + bytes(y1),
                             bytes(ctypes.c_float(expected[0])) +
                             bytes(ctypes.c_float(expected[1])), f"frame {index}")
            self.assertTrue(all(math.isfinite(value) for value in expected))
            output.extend(expected)
        self.assertEqual((scalar[0], scalar[-1], pair[0], pair[-1]), (12345.0,) * 4)
        # Compare the same recent input history, not physical layouts/positions.
        so = po = 0
        for layer, dilation in enumerate(DILATIONS + [None]):
            sl = 8 if dilation is None else 2 * dilation + 1
            pl = sl + 1
            spos = ss.head_pos if dilation is None else ss.layer_pos[layer]
            ppos = ps.head_pos if dilation is None else ps.layer_pos[layer]
            self.assertEqual(spos, len(samples) % sl)
            self.assertEqual(ppos, len(samples) % pl)
            for age in range(1, sl + 1):
                for channel in range(3):
                    self.assertEqual(sp[so + ((spos - age) % sl) * 3 + channel],
                                     pp[po + ((ppos - age) % pl) * 3 + channel])
            for slot in range(sl * 3):
                self.assertEqual(sp[so + slot], sp[so + sl * 3 + slot])
            so += sl * 6
            po += pl * 3
        self.assertEqual((so, po), (SCALAR_FLOATS, PAIR_FLOATS))
        return output

    def test_dense_models_biases_and_many_wraps(self):
        for seed in (4, 34, 94):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                weights = [rng.uniform(-0.08, 0.08) for _ in range(659)]
                weights[-1] = 1.0
                samples = [rng.uniform(-0.7, 0.7) if i % 61 == 0 else
                           math.sin(i * 0.011) * 0.2 for i in range(12000)]
                self.compare(weights, samples)

    def test_silence_impulses_resets_and_short_blocks(self):
        rng = random.Random(110)
        weights = [rng.uniform(-0.15, 0.15) for _ in range(659)]
        weights[-1] = 0.6
        for length in (2, 4, 8, 958, 4096):
            with self.subTest(length=length):
                samples = [0.0] * length
                if length > 2:
                    samples[1] = 0.8
                    samples[-2] = -0.5
                self.compare(weights, samples)

    def test_each_layer_and_head_tap_has_expected_delay(self):
        for layer, dilation in enumerate(DILATIONS):
            for tap in range(3):
                with self.subTest(layer=layer, tap=tap):
                    weights = [0.0] * 659
                    weights[0] = weights[3 + layer * 45 + tap] = 1.0
                    weights[633 + 7] = weights[658] = 1.0
                    samples = [0.0] * 1100
                    samples[0] = 0.5
                    actual = self.compare(weights, samples)
                    expected = [0.0] * len(samples)
                    expected[(2 - tap) * dilation] = 0.5
                    self.assertEqual(actual, expected)
        for tap in range(8):
            with self.subTest(head_tap=tap):
                weights = [0.0] * 659
                weights[33] = weights[633 + tap] = weights[658] = 1.0
                samples = [0.5] + [0.0] * 1099
                expected = [0.0] * len(samples)
                expected[7 - tap] = 0.5
                self.assertEqual(self.compare(weights, samples), expected)

    def test_local_trained_bank_when_available(self):
        path = ROOT / ".tooling/mix-build-20260928/weights.f32"
        if not path.is_file():
            self.skipTest("optional local trained bank is unavailable")
        raw = path.read_bytes()
        self.assertEqual(len(raw) % (659 * 4), 0)
        for offset in range(0, len(raw), 659 * 4):
            with self.subTest(slot=offset // (659 * 4)):
                weights = struct.unpack_from("<659f", raw, offset)
                self.compare(weights, [math.sin(i * 0.027) * 0.4 for i in range(4096)])


if __name__ == "__main__":
    unittest.main()
