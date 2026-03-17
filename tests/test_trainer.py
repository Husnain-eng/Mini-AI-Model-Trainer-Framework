"""
tests/test_trainer.py
=====================
Unit tests for Trainer, MultiTrainer, and TrainingResult.

Covers:
    - Trainer.run() trains and evaluates via polymorphic dispatch
    - Trainer.run() returns a TrainingResult
    - TrainingResult fields are populated correctly
    - MultiTrainer.run_all() processes every model
    - MultiTrainer.get_best() returns the correct result
    - Aggregation: DataLoader outlives Trainer (independence verified)
    - Polymorphism: same Trainer.run() code works for all model types
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from framework.base_model import BaseModel
from framework.config import ModelConfig
from framework.data_loader import DataLoader
from framework.models import LinearRegressionModel, NeuralNetworkModel, SVMModel
from framework.trainer import Trainer, MultiTrainer, TrainingResult


_DATA = [1.0, 2.5, 3.1, 4.8, 5.0, 6.3, 7.1]


def _lr(epochs=3) -> LinearRegressionModel:
    return LinearRegressionModel(ModelConfig("LR", learning_rate=0.01, epochs=epochs))

def _nn(epochs=3) -> NeuralNetworkModel:
    return NeuralNetworkModel(ModelConfig("NN", learning_rate=0.001, epochs=epochs), layers=[8, 4, 1])

def _svm(epochs=3) -> SVMModel:
    return SVMModel(ModelConfig("SVM", learning_rate=0.005, epochs=epochs))

def _loader() -> DataLoader:
    dl = DataLoader("TestSet")
    dl.load(_DATA)
    return dl


class TestTrainerRun(unittest.TestCase):

    def setUp(self):
        BaseModel.model_count = 0

    def test_run_returns_training_result(self):
        t = Trainer(_lr(), _loader())
        result = t.run()
        self.assertIsInstance(result, TrainingResult)

    def test_run_trains_model(self):
        model = _lr()
        Trainer(model, _loader()).run()
        self.assertTrue(model.is_trained)

    def test_run_result_model_name_matches(self):
        result = Trainer(_lr(), _loader()).run()
        self.assertEqual(result.model_name, "LR")

    def test_run_result_metric_value_is_float(self):
        result = Trainer(_lr(), _loader()).run()
        self.assertIsInstance(result.metric_value, float)

    def test_run_result_duration_positive(self):
        result = Trainer(_lr(), _loader()).run()
        self.assertGreater(result.duration_s, 0)

    def test_run_result_history_populated(self):
        result = Trainer(_lr(epochs=5), _loader()).run()
        self.assertEqual(result.epochs_logged, 5)

    def test_run_result_stored_in_trainer(self):
        t = Trainer(_lr(), _loader())
        result = t.run()
        self.assertIs(t.result, result)

    def test_run_result_none_before_run(self):
        t = Trainer(_lr(), _loader())
        self.assertIsNone(t.result)


class TestTrainerPolymorphism(unittest.TestCase):
    """Same Trainer.run() code works for every model type (polymorphism)."""

    def setUp(self):
        BaseModel.model_count = 0
        self.loader = _loader()

    def _run_model(self, model: BaseModel) -> TrainingResult:
        return Trainer(model, self.loader).run()

    def test_runs_linear_regression(self):
        result = self._run_model(_lr())
        self.assertEqual(result.metric_name, "MSE")

    def test_runs_neural_network(self):
        result = self._run_model(_nn())
        self.assertEqual(result.metric_name, "Accuracy")

    def test_runs_svm(self):
        result = self._run_model(_svm())
        self.assertEqual(result.metric_name, "F1Score")

    def test_all_return_training_result(self):
        for model in [_lr(), _nn(), _svm()]:
            result = Trainer(model, self.loader).run()
            self.assertIsInstance(result, TrainingResult)

    def test_no_isinstance_check_in_run_method(self):
        """Trainer.run() never calls isinstance — it relies purely on the interface."""
        import inspect
        from framework import trainer as trainer_module
        src = inspect.getsource(trainer_module.Trainer.run)
        self.assertNotIn("isinstance", src)


class TestTrainerAggregation(unittest.TestCase):
    """DataLoader is aggregated — it exists independently of Trainer."""

    def setUp(self):
        BaseModel.model_count = 0

    def test_data_loader_outlives_trainer(self):
        loader = _loader()
        t = Trainer(_lr(), loader)
        del t   # Trainer gone
        # DataLoader still intact
        self.assertEqual(len(loader), len(_DATA))

    def test_same_loader_used_by_multiple_trainers(self):
        loader = _loader()
        r1 = Trainer(_lr(), loader).run()
        r2 = Trainer(_nn(), loader).run()
        # Both trainers used the same loader without corrupting it
        self.assertEqual(len(loader), len(_DATA))
        self.assertIsInstance(r1, TrainingResult)
        self.assertIsInstance(r2, TrainingResult)

    def test_model_outlives_trainer(self):
        model = _lr()
        t = Trainer(model, _loader())
        t.run()
        del t   # Trainer gone
        # Model still trained
        self.assertTrue(model.is_trained)


class TestTrainingResult(unittest.TestCase):

    def setUp(self):
        BaseModel.model_count = 0
        self.result = Trainer(_lr(epochs=4), _loader()).run()

    def test_to_dict_keys(self):
        d = self.result.to_dict()
        for key in ("model_name", "metric_name", "metric_value", "duration_s", "trained_at", "epochs_logged"):
            self.assertIn(key, d)

    def test_to_dict_model_name(self):
        self.assertEqual(self.result.to_dict()["model_name"], "LR")

    def test_repr_contains_model_name(self):
        self.assertIn("LR", repr(self.result))

    def test_repr_contains_metric(self):
        self.assertIn("MSE", repr(self.result))

    def test_epochs_logged_correct(self):
        self.assertEqual(self.result.epochs_logged, 4)


class TestMultiTrainer(unittest.TestCase):

    def setUp(self):
        BaseModel.model_count = 0
        self.models = [_lr(epochs=3), _nn(epochs=3), _svm(epochs=3)]
        self.loader = _loader()
        self.mt = MultiTrainer(self.models, self.loader)

    def test_run_all_returns_list(self):
        results = self.mt.run_all()
        self.assertIsInstance(results, list)

    def test_run_all_trains_all_models(self):
        self.mt.run_all()
        for model in self.models:
            self.assertTrue(model.is_trained)

    def test_run_all_result_count_matches_model_count(self):
        results = self.mt.run_all()
        self.assertEqual(len(results), 3)

    def test_run_all_results_are_training_results(self):
        results = self.mt.run_all()
        for r in results:
            self.assertIsInstance(r, TrainingResult)

    def test_get_best_higher_is_better(self):
        results = self.mt.run_all()
        best = self.mt.get_best(higher_is_better=True)
        self.assertIsInstance(best, TrainingResult)
        # best.metric_value should be >= all others
        for r in results:
            self.assertGreaterEqual(best.metric_value, r.metric_value)

    def test_get_best_lower_is_better(self):
        results = self.mt.run_all()
        best = self.mt.get_best(higher_is_better=False)
        for r in results:
            self.assertLessEqual(best.metric_value, r.metric_value)

    def test_get_best_none_before_run(self):
        mt = MultiTrainer(self.models, self.loader)
        self.assertIsNone(mt.get_best())

    def test_run_all_called_twice_replaces_results(self):
        self.mt.run_all()
        first_count = len(self.mt.results)
        self.mt.run_all()
        self.assertEqual(len(self.mt.results), first_count)


if __name__ == "__main__":
    unittest.main(verbosity=2)
