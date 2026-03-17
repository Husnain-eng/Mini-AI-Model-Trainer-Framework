"""
tests/test_models.py
====================
Unit tests for LinearRegressionModel, NeuralNetworkModel, SVMModel.

Covers:
    - Inheritance chain (isinstance checks)
    - super() call — config and model_count set correctly
    - Child-specific attributes exist
    - train() sets is_trained, populates history
    - evaluate() returns a float in expected range
    - get_metric_name() returns correct string per model
    - Method overriding — different models return different metric names
    - Polymorphism — list of BaseModel refs, same calling code
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from framework.base_model import BaseModel
from framework.config import ModelConfig
from framework.models import LinearRegressionModel, NeuralNetworkModel, SVMModel


# ── Shared fixtures ────────────────────────────────────────────────────────────
def _lr_cfg() -> ModelConfig:
    return ModelConfig("LinearReg", learning_rate=0.01, epochs=5)

def _nn_cfg() -> ModelConfig:
    return ModelConfig("NeuralNet", learning_rate=0.001, epochs=5)

def _svm_cfg() -> ModelConfig:
    return ModelConfig("SVM", learning_rate=0.005, epochs=5)

_DATA = [1.0, 2.0, 3.0, 4.0, 5.0]


class TestLinearRegressionModel(unittest.TestCase):

    def setUp(self):
        BaseModel.model_count = 0
        self.model = LinearRegressionModel(_lr_cfg())

    # ── Inheritance / super() ──────────────────────────────────────────────────
    def test_isinstance_base_model(self):
        self.assertIsInstance(self.model, BaseModel)

    def test_config_set_by_super(self):
        self.assertEqual(self.model.config.model_name, "LinearReg")

    def test_model_count_incremented_by_super(self):
        self.assertEqual(BaseModel.model_count, 1)

    def test_is_trained_false_before_train(self):
        self.assertFalse(self.model.is_trained)

    # ── Child-specific attributes ──────────────────────────────────────────────
    def test_weights_initially_empty(self):
        self.assertEqual(self.model.weights, [])

    def test_bias_initially_zero(self):
        self.assertAlmostEqual(self.model.bias, 0.0)

    # ── train() ───────────────────────────────────────────────────────────────
    def test_train_sets_is_trained(self):
        self.model.train(_DATA)
        self.assertTrue(self.model.is_trained)

    def test_train_sets_trained_at(self):
        self.model.train(_DATA)
        self.assertIsNotNone(self.model.trained_at)

    def test_train_populates_weights(self):
        self.model.train(_DATA)
        self.assertEqual(len(self.model.weights), len(_DATA))

    def test_train_populates_history(self):
        self.model.train(_DATA)
        self.assertGreater(len(self.model.history), 0)

    def test_train_history_has_correct_epochs(self):
        self.model.train(_DATA)
        self.assertEqual(len(self.model.history), self.model.config.epochs)

    def test_train_history_record_keys(self):
        self.model.train(_DATA)
        record = self.model.history[0]
        self.assertIn("epoch", record)
        self.assertIn("loss", record)
        self.assertIn("MSE", record)

    # ── evaluate() ────────────────────────────────────────────────────────────
    def test_evaluate_returns_float(self):
        result = self.model.evaluate(_DATA)
        self.assertIsInstance(result, float)

    def test_evaluate_mse_positive(self):
        result = self.model.evaluate(_DATA)
        self.assertGreater(result, 0)

    def test_evaluate_mse_reasonable_range(self):
        result = self.model.evaluate(_DATA)
        self.assertLess(result, 1.0)   # MSE should be < 1 for normalised data

    def test_evaluate_deterministic(self):
        """Same seed → same evaluation result every time."""
        r1 = self.model.evaluate(_DATA)
        r2 = self.model.evaluate(_DATA)
        self.assertAlmostEqual(r1, r2)

    # ── get_metric_name() ─────────────────────────────────────────────────────
    def test_metric_name_is_mse(self):
        self.assertEqual(self.model.get_metric_name(), "MSE")

    # ── get_parameters() ──────────────────────────────────────────────────────
    def test_get_parameters_returns_dict(self):
        self.model.train(_DATA)
        params = self.model.get_parameters()
        self.assertIn("weights", params)
        self.assertIn("bias", params)


class TestNeuralNetworkModel(unittest.TestCase):

    def setUp(self):
        BaseModel.model_count = 0
        self.layers = [64, 32, 1]
        self.model = NeuralNetworkModel(_nn_cfg(), layers=self.layers)

    # ── Inheritance / super() ──────────────────────────────────────────────────
    def test_isinstance_base_model(self):
        self.assertIsInstance(self.model, BaseModel)

    def test_config_set_by_super(self):
        self.assertEqual(self.model.config.model_name, "NeuralNet")

    def test_model_count_incremented(self):
        self.assertEqual(BaseModel.model_count, 1)

    # ── Child-specific attributes ──────────────────────────────────────────────
    def test_layers_stored(self):
        self.assertEqual(self.model.layers, [64, 32, 1])

    def test_activation_default(self):
        self.assertEqual(self.model.activation, "relu")

    def test_custom_activation(self):
        m = NeuralNetworkModel(_nn_cfg(), layers=[4], activation="sigmoid")
        self.assertEqual(m.activation, "sigmoid")

    # ── train() ───────────────────────────────────────────────────────────────
    def test_train_sets_is_trained(self):
        self.model.train(_DATA)
        self.assertTrue(self.model.is_trained)

    def test_train_populates_history(self):
        self.model.train(_DATA)
        self.assertEqual(len(self.model.history), self.model.config.epochs)

    def test_train_history_has_accuracy_key(self):
        self.model.train(_DATA)
        self.assertIn("Accuracy", self.model.history[0])

    # ── evaluate() ────────────────────────────────────────────────────────────
    def test_evaluate_returns_float(self):
        result = self.model.evaluate(_DATA)
        self.assertIsInstance(result, float)

    def test_accuracy_in_valid_range(self):
        result = self.model.evaluate(_DATA)
        self.assertGreaterEqual(result, 91.5)
        self.assertLessEqual(result, 100.0)

    def test_evaluate_deterministic(self):
        r1 = self.model.evaluate(_DATA)
        r2 = self.model.evaluate(_DATA)
        self.assertAlmostEqual(r1, r2)

    # ── get_metric_name() — METHOD OVERRIDING ────────────────────────────────
    def test_metric_name_is_accuracy(self):
        self.assertEqual(self.model.get_metric_name(), "Accuracy")

    def test_metric_name_differs_from_linear_regression(self):
        lr = LinearRegressionModel(_lr_cfg())
        self.assertNotEqual(self.model.get_metric_name(), lr.get_metric_name())

    # ── total_parameters() ────────────────────────────────────────────────────
    def test_total_parameters_positive(self):
        self.assertGreater(self.model.total_parameters(), 0)


class TestSVMModel(unittest.TestCase):

    def setUp(self):
        BaseModel.model_count = 0
        self.model = SVMModel(_svm_cfg(), kernel="rbf", C=1.0)

    def test_isinstance_base_model(self):
        self.assertIsInstance(self.model, BaseModel)

    def test_kernel_stored(self):
        self.assertEqual(self.model.kernel, "rbf")

    def test_c_stored(self):
        self.assertAlmostEqual(self.model.C, 1.0)

    def test_train_sets_is_trained(self):
        self.model.train(_DATA)
        self.assertTrue(self.model.is_trained)

    def test_evaluate_f1_in_range(self):
        result = self.model.evaluate(_DATA)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_metric_name_is_f1(self):
        self.assertEqual(self.model.get_metric_name(), "F1Score")


class TestPolymorphism(unittest.TestCase):
    """All models respond identically to the same interface calls (polymorphism)."""

    def setUp(self):
        BaseModel.model_count = 0
        self.models: list[BaseModel] = [
            LinearRegressionModel(_lr_cfg()),
            NeuralNetworkModel(_nn_cfg(), layers=[8, 4, 1]),
            SVMModel(_svm_cfg(), kernel="linear"),
        ]

    def test_all_models_are_base_model_instances(self):
        for m in self.models:
            self.assertIsInstance(m, BaseModel)

    def test_all_models_train_without_error(self):
        for m in self.models:
            try:
                m.train(_DATA)
            except Exception as e:
                self.fail(f"{m.config.model_name}.train() raised {e}")

    def test_all_models_evaluate_return_float(self):
        for m in self.models:
            result = m.evaluate(_DATA)
            self.assertIsInstance(result, float, msg=f"Failed for {m.config.model_name}")

    def test_all_models_set_is_trained(self):
        for m in self.models:
            m.train(_DATA)
            self.assertTrue(m.is_trained, msg=f"Failed for {m.config.model_name}")

    def test_model_count_counts_all_subclasses(self):
        # 3 models created in setUp
        self.assertEqual(BaseModel.model_count, 3)

    def test_get_metric_name_returns_different_values(self):
        names = {m.get_metric_name() for m in self.models}
        # Each model type should have a unique metric name
        self.assertGreater(len(names), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
