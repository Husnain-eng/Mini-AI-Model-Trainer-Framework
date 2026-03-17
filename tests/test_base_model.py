"""
tests/test_base_model.py
========================
Unit tests for BaseModel (abstract class behaviour).

Covers:
    - Cannot instantiate BaseModel directly (TypeError)
    - model_count increments with each new instance (class attribute)
    - model_count is shared across different subclasses
    - is_trained defaults to False
    - describe() format
    - reset() clears training state
    - _mark_trained() sets is_trained + trained_at
    - __repr__ and __str__
"""

import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from framework.base_model import BaseModel
from framework.config import ModelConfig


# ── Minimal concrete stub for testing BaseModel behaviour ─────────────────────
class _StubModel(BaseModel):
    """Concrete stand-in that fulfils the abstract contract minimally."""

    def get_metric_name(self) -> str:
        return "StubMetric"

    def train(self, data: list) -> None:
        self._mark_trained()

    def evaluate(self, data: list) -> float:
        return 1.0


class TestBaseModelAbstraction(unittest.TestCase):
    """BaseModel cannot be instantiated directly (abstraction enforced)."""

    def test_direct_instantiation_raises_type_error(self):
        cfg = ModelConfig("X")
        with self.assertRaises(TypeError):
            BaseModel(cfg)   # type: ignore

    def test_missing_train_method_raises(self):
        class IncompleteModel(BaseModel):
            def evaluate(self, data): return 0.0
            def get_metric_name(self): return "M"

        cfg = ModelConfig("X")
        with self.assertRaises(TypeError):
            IncompleteModel(cfg)

    def test_missing_evaluate_raises(self):
        class IncompleteModel(BaseModel):
            def train(self, data): pass
            def get_metric_name(self): return "M"

        cfg = ModelConfig("X")
        with self.assertRaises(TypeError):
            IncompleteModel(cfg)


class TestBaseModelClassAttribute(unittest.TestCase):
    """model_count is a class attribute shared across all instances."""

    def setUp(self):
        # Reset so tests are independent of execution order
        BaseModel.model_count = 0

    def test_initial_count_is_zero(self):
        self.assertEqual(BaseModel.model_count, 0)

    def test_count_increments_on_creation(self):
        cfg = ModelConfig("A")
        _StubModel(cfg)
        self.assertEqual(BaseModel.model_count, 1)

    def test_count_increments_for_each_instance(self):
        cfg = ModelConfig("A")
        _StubModel(cfg)
        _StubModel(cfg)
        _StubModel(cfg)
        self.assertEqual(BaseModel.model_count, 3)

    def test_count_shared_across_subclasses(self):
        from framework.models import LinearRegressionModel, NeuralNetworkModel
        lr_cfg = ModelConfig("LR")
        nn_cfg = ModelConfig("NN")
        LinearRegressionModel(lr_cfg)
        NeuralNetworkModel(nn_cfg, layers=[4, 2, 1])
        self.assertEqual(BaseModel.model_count, 2)

    def test_count_accessible_on_instance(self):
        cfg = ModelConfig("A")
        m = _StubModel(cfg)
        # Should be accessible via instance (reads class attr)
        self.assertGreaterEqual(m.model_count, 1)


class TestBaseModelInstanceAttributes(unittest.TestCase):
    """Instance attributes are correctly initialised."""

    def setUp(self):
        BaseModel.model_count = 0
        self.cfg = ModelConfig("TestModel", learning_rate=0.01, epochs=5)
        self.model = _StubModel(self.cfg)

    def test_config_is_stored(self):
        self.assertIs(self.model.config, self.cfg)

    def test_is_trained_defaults_false(self):
        self.assertFalse(self.model.is_trained)

    def test_trained_at_defaults_none(self):
        self.assertIsNone(self.model.trained_at)

    def test_history_defaults_empty_list(self):
        self.assertEqual(self.model.history, [])


class TestBaseModelDescribe(unittest.TestCase):
    """describe() produces the expected human-readable string."""

    def setUp(self):
        BaseModel.model_count = 0
        self.cfg = ModelConfig("Describer")
        self.model = _StubModel(self.cfg)

    def test_describe_contains_model_name(self):
        self.assertIn("Describer", self.model.describe())

    def test_describe_shows_untrained(self):
        self.assertIn("untrained", self.model.describe())

    def test_describe_shows_trained_after_train(self):
        self.model.train([1, 2, 3])
        self.assertIn("trained", self.model.describe())

    def test_describe_shows_timestamp_after_train(self):
        self.model.train([1])
        desc = self.model.describe()
        # Should contain a year (timestamp present)
        self.assertRegex(desc, r"\d{4}-\d{2}-\d{2}")


class TestBaseModelReset(unittest.TestCase):
    """reset() clears training state."""

    def setUp(self):
        BaseModel.model_count = 0
        self.cfg = ModelConfig("Resetter")
        self.model = _StubModel(self.cfg)

    def test_reset_clears_is_trained(self):
        self.model.train([1])
        self.model.reset()
        self.assertFalse(self.model.is_trained)

    def test_reset_clears_trained_at(self):
        self.model.train([1])
        self.model.reset()
        self.assertIsNone(self.model.trained_at)

    def test_reset_clears_history(self):
        self.model._log_epoch(1, 0.5, 0.9)
        self.model.reset()
        self.assertEqual(self.model.history, [])


class TestBaseModelMarkTrained(unittest.TestCase):
    """_mark_trained() helper sets is_trained and trained_at."""

    def setUp(self):
        BaseModel.model_count = 0
        self.model = _StubModel(ModelConfig("Marker"))

    def test_mark_trained_sets_flag(self):
        self.model._mark_trained()
        self.assertTrue(self.model.is_trained)

    def test_mark_trained_sets_timestamp(self):
        before = datetime.now()
        self.model._mark_trained()
        after = datetime.now()
        self.assertIsNotNone(self.model.trained_at)
        self.assertGreaterEqual(self.model.trained_at, before)
        self.assertLessEqual(self.model.trained_at, after)


class TestBaseModelLogEpoch(unittest.TestCase):
    """_log_epoch() appends to self.history correctly."""

    def setUp(self):
        BaseModel.model_count = 0
        self.model = _StubModel(ModelConfig("Logger"))

    def test_log_epoch_appends(self):
        self.model._log_epoch(1, 0.5, 0.8)
        self.assertEqual(len(self.model.history), 1)

    def test_log_epoch_record_keys(self):
        self.model._log_epoch(1, 0.5, 0.8)
        record = self.model.history[0]
        self.assertIn("epoch", record)
        self.assertIn("loss", record)
        self.assertIn("StubMetric", record)

    def test_log_epoch_values(self):
        self.model._log_epoch(3, 0.123456789, 0.987654321)
        record = self.model.history[0]
        self.assertEqual(record["epoch"], 3)
        self.assertAlmostEqual(record["loss"], 0.123457, places=4)


class TestBaseModelMagicMethods(unittest.TestCase):
    """__repr__ and __str__ return useful strings."""

    def setUp(self):
        BaseModel.model_count = 0
        self.model = _StubModel(ModelConfig("Magic"))

    def test_repr_contains_class_name(self):
        self.assertIn("_StubModel", repr(self.model))

    def test_repr_contains_trained_status(self):
        self.assertIn("trained=False", repr(self.model))

    def test_str_delegates_to_describe(self):
        self.assertEqual(str(self.model), self.model.describe())


if __name__ == "__main__":
    unittest.main(verbosity=2)
