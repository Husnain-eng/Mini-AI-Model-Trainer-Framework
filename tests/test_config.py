"""
tests/test_config.py
====================
Unit tests for ModelConfig.

Covers:
    - Default values
    - Custom values
    - __repr__ format
    - __eq__ and __hash__
    - Validation errors
    - to_dict / from_dict round-trip
    - save_json / from_json round-trip
    - summary()
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from framework.config import ModelConfig


class TestModelConfigDefaults(unittest.TestCase):
    """ModelConfig initialises correctly with default hyperparameters."""

    def setUp(self):
        self.cfg = ModelConfig("TestModel")

    def test_model_name(self):
        self.assertEqual(self.cfg.model_name, "TestModel")

    def test_default_learning_rate(self):
        self.assertAlmostEqual(self.cfg.learning_rate, 0.01)

    def test_default_epochs(self):
        self.assertEqual(self.cfg.epochs, 10)

    def test_default_batch_size(self):
        self.assertEqual(self.cfg.batch_size, 32)

    def test_default_seed(self):
        self.assertEqual(self.cfg.seed, 42)

    def test_default_extra_is_empty_dict(self):
        self.assertEqual(self.cfg.extra, {})


class TestModelConfigCustomValues(unittest.TestCase):
    """ModelConfig stores custom hyperparameters correctly."""

    def setUp(self):
        self.cfg = ModelConfig(
            "NeuralNet",
            learning_rate=0.001,
            epochs=50,
            batch_size=64,
            seed=7,
            extra={"dropout": 0.3},
        )

    def test_learning_rate(self):
        self.assertAlmostEqual(self.cfg.learning_rate, 0.001)

    def test_epochs(self):
        self.assertEqual(self.cfg.epochs, 50)

    def test_batch_size(self):
        self.assertEqual(self.cfg.batch_size, 64)

    def test_seed(self):
        self.assertEqual(self.cfg.seed, 7)

    def test_extra(self):
        self.assertEqual(self.cfg.extra["dropout"], 0.3)


class TestModelConfigRepr(unittest.TestCase):
    """__repr__ produces the expected string format."""

    def test_repr_contains_model_name(self):
        cfg = ModelConfig("LinearReg", learning_rate=0.01, epochs=10)
        r = repr(cfg)
        self.assertIn("LinearReg", r)

    def test_repr_contains_lr(self):
        cfg = ModelConfig("M", learning_rate=0.005, epochs=5)
        self.assertIn("0.005", repr(cfg))

    def test_repr_contains_epochs(self):
        cfg = ModelConfig("M", learning_rate=0.01, epochs=25)
        self.assertIn("25", repr(cfg))

    def test_repr_contains_config_prefix(self):
        cfg = ModelConfig("M")
        self.assertIn("[Config]", repr(cfg))


class TestModelConfigEquality(unittest.TestCase):
    """__eq__ and __hash__ behave correctly."""

    def test_equal_configs(self):
        a = ModelConfig("M", learning_rate=0.01, epochs=10, batch_size=32, seed=42)
        b = ModelConfig("M", learning_rate=0.01, epochs=10, batch_size=32, seed=42)
        self.assertEqual(a, b)

    def test_different_lr_not_equal(self):
        a = ModelConfig("M", learning_rate=0.01)
        b = ModelConfig("M", learning_rate=0.001)
        self.assertNotEqual(a, b)

    def test_same_hash_for_equal_configs(self):
        a = ModelConfig("M", learning_rate=0.01, epochs=10, batch_size=32, seed=42)
        b = ModelConfig("M", learning_rate=0.01, epochs=10, batch_size=32, seed=42)
        self.assertEqual(hash(a), hash(b))

    def test_usable_as_dict_key(self):
        cfg = ModelConfig("M")
        d = {cfg: "value"}
        self.assertEqual(d[cfg], "value")

    def test_not_equal_to_non_config(self):
        cfg = ModelConfig("M")
        result = cfg.__eq__("not a config")
        self.assertEqual(result, NotImplemented)


class TestModelConfigValidation(unittest.TestCase):
    """validate() raises ValueError for invalid hyperparameters."""

    def test_empty_model_name_raises(self):
        with self.assertRaises(ValueError):
            ModelConfig("")

    def test_non_string_model_name_raises(self):
        with self.assertRaises((ValueError, TypeError)):
            ModelConfig(None)

    def test_zero_learning_rate_raises(self):
        with self.assertRaises(ValueError):
            ModelConfig("M", learning_rate=0.0)

    def test_negative_learning_rate_raises(self):
        with self.assertRaises(ValueError):
            ModelConfig("M", learning_rate=-0.01)

    def test_learning_rate_gte_1_raises(self):
        with self.assertRaises(ValueError):
            ModelConfig("M", learning_rate=1.0)

    def test_zero_epochs_raises(self):
        with self.assertRaises(ValueError):
            ModelConfig("M", epochs=0)

    def test_negative_epochs_raises(self):
        with self.assertRaises(ValueError):
            ModelConfig("M", epochs=-5)

    def test_zero_batch_size_raises(self):
        with self.assertRaises(ValueError):
            ModelConfig("M", batch_size=0)


class TestModelConfigSerialization(unittest.TestCase):
    """to_dict / from_dict / save_json / from_json round-trips."""

    def setUp(self):
        self.cfg = ModelConfig("SVM", learning_rate=0.005, epochs=30, batch_size=16, seed=99)

    def test_to_dict_keys(self):
        d = self.cfg.to_dict()
        for key in ("model_name", "learning_rate", "epochs", "batch_size", "seed", "extra"):
            self.assertIn(key, d)

    def test_to_dict_values(self):
        d = self.cfg.to_dict()
        self.assertEqual(d["model_name"], "SVM")
        self.assertAlmostEqual(d["learning_rate"], 0.005)
        self.assertEqual(d["epochs"], 30)

    def test_from_dict_round_trip(self):
        restored = ModelConfig.from_dict(self.cfg.to_dict())
        self.assertEqual(self.cfg, restored)

    def test_from_dict_uses_defaults(self):
        d = {"model_name": "X"}
        cfg = ModelConfig.from_dict(d)
        self.assertAlmostEqual(cfg.learning_rate, 0.01)
        self.assertEqual(cfg.epochs, 10)

    def test_save_and_load_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.cfg.save_json(path)
            restored = ModelConfig.from_json(path)
            self.assertEqual(self.cfg, restored)
        finally:
            os.unlink(path)

    def test_summary_returns_dict(self):
        s = self.cfg.summary()
        self.assertIsInstance(s, dict)
        self.assertEqual(s["model_name"], "SVM")


if __name__ == "__main__":
    unittest.main(verbosity=2)
