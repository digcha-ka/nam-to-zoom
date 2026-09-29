"""Offline compact WaveNet shape and resource tests."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from nam2zoom.compact import (  # noqa: E402
    DILATIONS, KERNEL_SIZES, expected_parameters, inspect,
)


def model(channels):
    layers = len(KERNEL_SIZES)
    submodel = {
        "architecture": "WaveNet", "sample_rate": 44100,
        "config": {"head": None, "head_scale": 0.01, "layers": [{
            "input_size": 1, "condition_size": 1, "channels": channels,
            "bottleneck": channels, "kernel_sizes": list(KERNEL_SIZES),
            "dilations": list(DILATIONS),
            "activation": [{"type": "LeakyReLU", "negative_slope": 0.01}] * layers,
            "gating_mode": ["none"] * layers,
            "secondary_activation": [None] * layers,
            "layer1x1": {"active": True, "groups": 1},
            "head1x1": {"active": False},
            "head": {"out_channels": 1, "kernel_size": 8, "bias": True},
        }]},
        "weights": [0.0] * expected_parameters(channels),
    }
    return {"architecture": "SlimmableContainer", "sample_rate": 44100,
            "weights": [], "config": {"submodels": [
                {"max_value": 1.0, "model": submodel},
            ]}}


class CompactShapeTests(unittest.TestCase):
    def test_resource_counts(self):
        for channels, weights, history, terms in (
                (2, 328, 26528, 168), (3, 659, 39792, 378)):
            with self.subTest(channels=channels):
                parsed = inspect(model(channels))
                self.assertEqual(len(parsed.weights), weights)
                self.assertEqual(len(parsed.weight_bytes()), weights * 4)
                self.assertEqual(parsed.mirrored_history_bytes, history)
                self.assertEqual(parsed.convolution_terms_per_sample, terms)
                self.assertEqual(parsed.receptive_field, 1644)

    def test_reject_changed_shape(self):
        for key, value in (("dilations", [1] * 14),
                           ("kernel_sizes", [6] * 14),
                           ("channels", 4),
                           ("layer1x1", {"active": False})):
            with self.subTest(key=key):
                data = model(2)
                data["config"]["submodels"][0]["model"]["config"]["layers"][0][key] = value
                with self.assertRaises(ValueError):
                    inspect(data)

    def test_reject_nonfinite_and_wrong_rate(self):
        data = model(2)
        data["config"]["submodels"][0]["model"]["weights"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            inspect(data)
        data = model(3)
        data["sample_rate"] = 48000
        with self.assertRaisesRegex(ValueError, "44.1"):
            inspect(data)
        data = model(2)
        data["config"]["submodels"][0]["model"]["weights"].pop()
        with self.assertRaisesRegex(ValueError, "weight count"):
            inspect(data)


if __name__ == "__main__":
    unittest.main()
