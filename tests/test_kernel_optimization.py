"""Check the production 3-channel kernel against its reference implementation."""

import ctypes
import _ctypes
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dsp/nam_a2_compact/compact_kernel.c"
HISTORY_FLOATS = 2 * 1658 * 3
WEIGHTS = 659


class State(ctypes.Structure):
    _fields_ = [("layer_pos", ctypes.c_uint16 * 14),
                ("head_pos", ctypes.c_uint16)]


def find_clang():
    found = shutil.which("clang")
    if found:
        return found
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "")) / (
        "Microsoft Visual Studio/Installer/vswhere.exe")
    if vswhere.is_file():
        root = subprocess.check_output(
            [str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
            text=True).strip()
        candidate = Path(root) / "VC/Tools/Llvm/x64/bin/clang.exe"
        if candidate.is_file():
            return str(candidate)
    return None


class KernelOptimizationTests(unittest.TestCase):
    def test_same_audio_history_and_ring_state(self):
        clang = find_clang()
        if not clang:
            self.skipTest("host Clang compiler is unavailable")
        with tempfile.TemporaryDirectory() as temp:
            functions = []
            libraries = []
            for optimized in (False, True):
                path = Path(temp) / ("optimized.dll" if optimized else "reference.dll")
                command = [clang, "-O3", "-ffp-contract=off", "-DCOMPACT_CHANNELS=3"]
                if optimized:
                    command.append("-DN2Z_OPTIMIZED_KERNEL")
                if os.name == "nt":
                    command.append("-Wl,/export:compact_process")
                command += ["-shared", "-o", str(path), str(SOURCE)]
                subprocess.run(command, check=True, capture_output=True)
                library = ctypes.CDLL(str(path))
                libraries.append(library)
                function = library.compact_process
                function.argtypes = (ctypes.POINTER(ctypes.c_float),
                                     ctypes.POINTER(ctypes.c_float),
                                     ctypes.POINTER(State), ctypes.c_float)
                function.restype = ctypes.c_float
                functions.append(function)

            for seed in (4, 34, 94):
                rng = random.Random(seed)
                weights = (ctypes.c_float * WEIGHTS)(
                    *(rng.uniform(-0.08, 0.08) for _ in range(WEIGHTS)))
                weights[-1] = 1.0
                samples = [0.8 if i == 0 else (
                    rng.uniform(-0.5, 0.5) if i % 61 == 0 else
                    math.sin(i * 0.011) * 0.2) for i in range(12000)]
                runs = []
                for function in functions:
                    history = (ctypes.c_float * HISTORY_FLOATS)()
                    state = State()
                    output = [function(weights, history, ctypes.byref(state), sample)
                              for sample in samples]
                    runs.append((output, history, bytes(state)))
                original, optimized = runs[0][0], runs[1][0]
                energy = sum(sample * sample for sample in original)
                error = sum((a - b) ** 2 for a, b in zip(original, optimized))
                self.assertLess(error / energy, 1e-8)
                self.assertLess(max(abs(a - b) for a, b in zip(original, optimized)), 1e-4)
                self.assertLess(max(abs(a - b) for a, b in zip(runs[0][1], runs[1][1])), 1e-4)
                self.assertEqual(runs[0][2], runs[1][2])
            functions.clear()
            if os.name == "nt":
                for library in libraries:
                    _ctypes.FreeLibrary(library._handle)


if __name__ == "__main__":
    unittest.main()
