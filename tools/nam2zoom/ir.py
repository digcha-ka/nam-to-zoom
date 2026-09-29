"""Prepare a mono cabinet IR for the compact teacher/student path."""

from __future__ import annotations

from math import gcd
from pathlib import Path

RATE = 44100
MAX_IR_BYTES = 10 * 1024 * 1024
MAX_IR_SECONDS = 0.5
MAX_LEADING_SAMPLES = RATE // 100
MAX_TAPS = 1536
MAX_DISCARDED_ENERGY = 0.01


def load_ir(path: Path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    if path.suffix.lower() != ".wav" or path.stat().st_size > MAX_IR_BYTES:
        raise ValueError("cab IR must be a WAV file no larger than 10 MiB")
    info = sf.info(path)
    if info.channels != 1 or info.samplerate not in (44100, 48000, 88200, 96000):
        raise ValueError("cab IR must be mono at 44.1, 48, 88.2, or 96 kHz")
    if not 0 < info.duration <= MAX_IR_SECONDS:
        raise ValueError("cab IR must be longer than zero and at most 500 ms")
    samples, _ = sf.read(path, dtype="float32")
    if not np.isfinite(samples).all():
        raise ValueError("cab IR contains non-finite samples")
    divisor = gcd(info.samplerate, RATE)
    taps = resample_poly(samples, RATE // divisor, info.samplerate // divisor)
    peak = float(np.max(np.abs(taps)))
    if peak < 1e-7:
        raise ValueError("cab IR is silent")
    onset = int(np.argmax(np.abs(taps) >= peak * 1e-4))
    if onset > MAX_LEADING_SAMPLES:
        raise ValueError("cab IR has more than 10 ms of leading silence; trim it first")
    taps = taps[onset:]
    if len(taps) > MAX_TAPS:
        energy = float(np.sum(taps.astype("float64") ** 2))
        discarded = float(np.sum(taps[MAX_TAPS:].astype("float64") ** 2))
        if discarded / energy > MAX_DISCARDED_ENERGY:
            raise ValueError("cab IR has too much energy beyond 35 ms for this compact model")
        print(f"Cab IR tail beyond {MAX_TAPS} samples contains "
              f"{100 * discarded / energy:.2f}% of its energy; trimming", flush=True)
        taps = taps[:MAX_TAPS]
    print(f"Cab IR: {info.samplerate} Hz -> {RATE} Hz, mono, "
          f"trimmed {onset} leading samples, using {len(taps)} taps", flush=True)
    return taps.astype("float32")


def bake(audio, taps):
    import numpy as np
    from scipy.signal import oaconvolve

    result = oaconvolve(audio, taps, mode="full")[:len(audio)]
    if not np.isfinite(result).all():
        raise ValueError("cab IR produced non-finite teacher audio")
    return result.astype("float32")


def fit_teacher_level(audio):
    """Keep a floating-point IR render below the trainer's clipping guard."""
    import numpy as np

    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("cab IR produced invalid teacher audio")
    gain = min(1.0, 0.95 / peak)
    gain_db = 20.0 * float(np.log10(gain))
    if gain < 1.0:
        print(f"Cab IR raised teacher peak to {peak:.4f}; applying "
              f"{gain_db:.2f} dB gain before training (new peak 0.9500)", flush=True)
        audio = audio * gain
    return audio.astype("float32"), {"ir_peak_before_gain": peak,
                                     "ir_gain_db": gain_db}
