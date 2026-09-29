"""Exercise the bank's actual C blend helpers without pedal access."""

import ctypes
import _ctypes
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_kernel_optimization import find_clang

ROOT = Path(__file__).resolve().parents[1]


class BankMixTests(unittest.TestCase):
    def test_dry_wet_bypass_and_warmup(self):
        clang = find_clang()
        if not clang:
            self.skipTest("host Clang compiler is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            folder.joinpath("sh_params.h").write_text(
                "#define SH_STATE_BYTES 256\n#define SH_FRAMES 4\n"
                "#define SH_CH_B_OFFSET 4\n#define SH_CTX_EFF 0\n"
                "#define SH_STATE_GUARD_WORD 63\n#define SH_STATE_GUARD 123\n"
                "#define SH_COEFF_BYPASS 0\n#define SH_AUDIO_FN test_audio\n"
                + "".join(f"#define SH_PARAM_{name} {index}\n" for index, name in
                          enumerate(("MODEL", "BASS", "MID", "TREBLE", "VOL", "INPUT", "MIX"), 1)),
                encoding="ascii")
            folder.joinpath("bank_config.h").write_text(
                "#define BANK_MODEL_COUNT 1u\n#define BANK_SELECTOR_MAX 1u\n"
                "#define BANK_WORDS_PER_MODEL 659u\n#define N2Z_OPTIMIZED_KERNEL 1\n",
                encoding="ascii")
            source = (ROOT / "dsp/nam_a2_bank_zd2/bank_effect.c").as_posix()
            wrapper = folder / "wrapper.c"
            wrapper.write_text(
                f'#include "{source}"\n'
                'const uint32_t N2ZBankWeights[659] = {0};\n'
                'float test_blend(float dry, float wet, float fade, float mix) {\n'
                '    return blend_output(dry, wet, fade * mix);\n}\n'
                'void test_warmup(float *bus, float fade, float mix) {\n'
                '    mute_wet(bus, fade * mix);\n}\n', encoding="ascii")
            library_path = folder / ("mix.dll" if os.name == "nt" else "mix.so")
            command = [clang, "-O3", "-shared", "-Wno-unknown-pragmas", "-I", str(folder),
                       "-I", str(ROOT / "dsp/nam_a2_compact")]
            if os.name == "nt":
                command += ["-Wl,/export:test_blend", "-Wl,/export:test_warmup"]
            subprocess.run(command + [str(wrapper), "-o", str(library_path)],
                           check=True, capture_output=True)
            library = ctypes.CDLL(str(library_path))
            blend = library.test_blend
            blend.argtypes = [ctypes.c_float] * 4
            blend.restype = ctypes.c_float
            warmup = library.test_warmup
            warmup.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float]
            try:
                for dry, wet in ((0.4, -0.2), (-0.3, 0.7)):
                    for fade in (0.0, 0.25, 1.0):
                        for mix in (0.0, 0.5, 1.0):
                            gain = fade * mix
                            self.assertAlmostEqual(blend(dry, wet, fade, mix),
                                                   dry * (1 - gain) + wet * gain, places=6)
                for mix in (0.0, 0.5, 1.0):
                    bus = (ctypes.c_float * 8)(0.2, -0.4, 0.6, -0.8, 0.7, -0.5, 0.3, -0.1)
                    expected = [value * (1 - mix) for value in bus]
                    warmup(bus, 1.0, mix)
                    for actual, target in zip(bus, expected):
                        self.assertAlmostEqual(actual, target, places=6)
                bus = (ctypes.c_float * 8)(math.nan, math.inf, -math.inf, 9, -9, 0, 0, 0)
                warmup(bus, 1.0, 1.0)
                self.assertEqual(list(bus), [0.0] * 8)
            finally:
                del blend, warmup
                if os.name == "nt":
                    _ctypes.FreeLibrary(library._handle)


if __name__ == "__main__":
    unittest.main()
