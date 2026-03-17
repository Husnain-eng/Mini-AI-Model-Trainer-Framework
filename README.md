# Mini AI Model Trainer Framework

> A production-quality Python OOP framework that simulates ML training pipelines —  
> demonstrating every core object-oriented concept in one coherent codebase.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [OOP Concepts Map](#oop-concepts-map)
5. [Class Reference](#class-reference)
6. [CLI Usage](#cli-usage)
7. [Running Tests](#running-tests)
8. [Extending the Framework](#extending-the-framework)
9. [Configuration Files](#configuration-files)

---

## Overview

This framework simulates the core architecture of production ML libraries (PyTorch, scikit-learn)
at a teaching scale. It is fully functional Python — not pseudocode — and demonstrates:

| OOP Concept        | Implemented In                                        |
|--------------------|-------------------------------------------------------|
| Class Attribute    | `BaseModel.model_count`                               |
| Instance Attribute | `ModelConfig.learning_rate`, `model.config`, etc.     |
| Abstraction (ABC)  | `BaseModel` with `@abstractmethod` train/evaluate     |
| Single Inheritance | `LinearRegressionModel`, `NeuralNetworkModel`, `SVMModel` → `BaseModel` |
| Method Overriding  | `train()`, `evaluate()`, `get_metric_name()` per model|
| `super()`          | Every child `__init__` calls `super().__init__(config)`|
| Polymorphism       | `Trainer.run()` / `MultiTrainer.run_all()` — no `isinstance` |
| Composition        | `BaseModel` owns a `ModelConfig` instance             |
| Aggregation        | `Trainer` receives `DataLoader` + `BaseModel` externally |
| Magic Methods      | `__repr__`, `__eq__`, `__hash__`, `__len__`, `__iter__` |
| Instance Methods   | `train()`, `evaluate()`, `run()`, `describe()`, `stats()`, … |

---

## Project Structure

```
mini_ai_trainer/
│
├── framework/                   # Core library
│   ├── __init__.py              # Public API
│   ├── config.py                # ModelConfig  (Composition object)
│   ├── base_model.py            # BaseModel    (Abstract base class)
│   ├── models.py                # LinearRegressionModel, NeuralNetworkModel, SVMModel
│   ├── data_loader.py           # DataLoader   (Aggregation target)
│   ├── trainer.py               # Trainer, MultiTrainer, TrainingResult
│   └── logger.py                # Structured logger (stdout + JSON file)
│
├── tests/                       # Full unit test suite
│   ├── test_config.py           # 21 tests — ModelConfig
│   ├── test_base_model.py       # 27 tests — BaseModel (abstraction, class attr)
│   ├── test_models.py           # 46 tests — concrete models + polymorphism
│   ├── test_data_loader.py      # 27 tests — DataLoader
│   └── test_trainer.py          # 26 tests — Trainer, MultiTrainer, TrainingResult
│
├── configs/                     # JSON hyperparameter files
│   ├── linear_regression.json
│   ├── neural_network.json
│   └── svm.json
│
├── logs/                        # Auto-created on first run
│   └── ai_trainer.log           # JSON-line structured logs
│
├── scripts/
│   └── run_pipeline.py          # Full CLI with argparse
│
├── main.py                      # Entry point — runs full pipeline demo
└── README.md
```

---

## Quick Start

```bash
# 1. Clone / enter the project
cd mini_ai_trainer

# 2. No external dependencies for core framework — stdlib only!
#    (pytest needed only for tests)
pip install pytest

# 3. Run the full pipeline demo
python main.py

# 4. Run all 167 unit tests
python -m pytest tests/ -v
```

### Expected Output (main.py)

```
════════════════════════════════════════════════════════════
  STEP 1 — Model Configurations
════════════════════════════════════════════════════════════
[Config] LinearRegression     | lr=0.01   | epochs=10   | batch=32
[Config] NeuralNetwork        | lr=0.001  | epochs=20   | batch=64
[Config] SupportVectorMachine | lr=0.005  | epochs=15   | batch=16

════════════════════════════════════════════════════════════
  STEP 2 — Model Instantiation
════════════════════════════════════════════════════════════
  Models created so far : 3
  LinearRegression (untrained) | ...
  NeuralNetwork (untrained)    | ...
  SupportVectorMachine (untrained) | ...

════════════════════════════════════════════════════════════
  STEP 4 — Training Pipeline (Polymorphism)
════════════════════════════════════════════════════════════

  Training  →  LinearRegression
  Samples : 15  |  Epochs : 10  |  LR : 0.01
  Epoch [  1/10]  loss=0.97168  MSE=0.48584  ██
  ...
  ✓ Training complete — LinearRegression
  MSE = 0.0181  (lower is better)

  Training  →  NeuralNetwork
  Architecture : 64 → 32 → 1
  ...
  ✓ Training complete — NeuralNetwork [64, 32, 1]
  Accuracy = 88.12%  (higher is better)
  ...

═════════════════════════════════════════════════════════════════
                         TRAINING REPORT
═════════════════════════════════════════════════════════════════
  Model                  Metric            Value      Time  Epochs
  LinearRegression       MSE              0.0181    0.000s      10
  NeuralNetwork          Accuracy        88.1200    0.000s      20
  SupportVectorMachine   F1Score          0.8922    0.000s      15
═════════════════════════════════════════════════════════════════
```

---

## OOP Concepts Map

### Composition vs Aggregation

```
COMPOSITION (strong ownership — lives and dies with the owner):
    ModelConfig  ←────────────  BaseModel
    config is created before the model and passed in,
    but logically it belongs to exactly one model for its lifetime.

AGGREGATION (weak ownership — outlives the user):
    DataLoader  ────────────►  Trainer
    BaseModel   ────────────►  Trainer
    Trainer receives both externally. Delete the Trainer — both still exist.
```

### Polymorphism in Action

```python
# Trainer.run() never checks what type of model it has.
# Python's runtime dispatch calls the correct train()/evaluate().

models: list[BaseModel] = [
    LinearRegressionModel(lr_config),    # MSE
    NeuralNetworkModel(nn_config, ...),  # Accuracy
    SVMModel(svm_config, ...),           # F1 Score
]

for model in models:                     # ← same loop, three different behaviours
    Trainer(model=model, data_loader=loader).run()
```

### super() Chain

```python
class LinearRegressionModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)   # ← sets self.config, increments model_count,
                                   #   sets is_trained=False, initialises history
        self.weights = []          # ← then child adds its own attributes
        self.bias = 0.0
```

---

## Class Reference

### `ModelConfig`
| Member | Type | Description |
|--------|------|-------------|
| `model_name` | `str` | Human-readable name |
| `learning_rate` | `float` | Step size (0 < lr < 1) |
| `epochs` | `int` | Training iterations ≥ 1 |
| `batch_size` | `int` | Samples per update ≥ 1 |
| `seed` | `int` | Reproducibility seed |
| `validate()` | method | Raises `ValueError` on bad params |
| `to_dict()` / `from_dict()` | method / classmethod | Serialisation |
| `save_json()` / `from_json()` | method / classmethod | File I/O |
| `__repr__` | magic | `[Config] Name \| lr=… \| epochs=… \| batch=…` |
| `__eq__` / `__hash__` | magic | Value equality, usable as dict key |

### `BaseModel` (ABC)
| Member | Type | Description |
|--------|------|-------------|
| `model_count` | **class attr** | Total instances ever created |
| `config` | instance attr | Composed `ModelConfig` |
| `is_trained` | instance attr | `False` until `train()` completes |
| `history` | instance attr | Per-epoch `[{epoch, loss, metric}]` |
| `train(data)` | **abstract** | Must override in subclass |
| `evaluate(data)` | **abstract** | Must override in subclass |
| `get_metric_name()` | **abstract** | Returns `'MSE'`, `'Accuracy'`, etc. |
| `describe()` | concrete | One-line status string |
| `reset()` | concrete | Clears training state |

### Concrete Models
| Class | Extra Attrs | Metric |
|-------|-------------|--------|
| `LinearRegressionModel` | `weights`, `bias` | MSE (lower is better) |
| `NeuralNetworkModel` | `layers`, `activation` | Accuracy % (higher) |
| `SVMModel` | `kernel`, `C` | F1 Score 0–1 (higher) |

### `DataLoader`
| Member | Description |
|--------|-------------|
| `load(data)` | Load from Python list, returns `self` |
| `load_csv(path)` | Parse numeric CSV |
| `get_batch(size)` | Full dataset or random sub-sample |
| `split(train, val, test)` | Partition dataset, no overlap |
| `get_split(name)` | Retrieve `'train'`, `'val'`, `'test'` |
| `stats()` | count, min, max, mean, std_dev |
| `__len__` / `__iter__` | `len(loader)`, `for x in loader` |

### `Trainer` / `MultiTrainer`
| Class | Key Method | Description |
|-------|------------|-------------|
| `Trainer` | `run()` | Train + evaluate one model, returns `TrainingResult` |
| `MultiTrainer` | `run_all()` | Run all models polymorphically |
| `MultiTrainer` | `print_report()` | Formatted comparison table |
| `MultiTrainer` | `get_best(higher_is_better)` | Best result by metric |

---

## CLI Usage

```bash
# Train all models (default)
python scripts/run_pipeline.py

# Train only Linear Regression with custom hyperparameters
python scripts/run_pipeline.py --model lr --lr 0.005 --epochs 25

# Train Neural Network with custom architecture
python scripts/run_pipeline.py --model nn --layers 128 64 32 1 --activation sigmoid

# Train SVM with linear kernel
python scripts/run_pipeline.py --model svm --kernel linear --C 0.5

# Load hyperparameters from a JSON config file
python scripts/run_pipeline.py --model lr --config configs/linear_regression.json

# Save current hyperparameters to JSON
python scripts/run_pipeline.py --model nn --save-config configs/my_custom_nn.json
```

---

## Running Tests

```bash
# All 167 tests
python -m pytest tests/ -v

# One test file
python -m pytest tests/test_models.py -v

# One specific test
python -m pytest tests/test_models.py::TestPolymorphism -v

# With coverage (requires pytest-cov)
pip install pytest-cov
python -m pytest tests/ --cov=framework --cov-report=term-missing
```

### Test Summary

| File | Tests | What Is Covered |
|------|-------|-----------------|
| `test_config.py` | 21 | Defaults, validation, `__repr__`, equality, serialisation |
| `test_base_model.py` | 27 | Abstraction, class attr, `describe()`, `reset()`, helpers |
| `test_models.py` | 46 | Inheritance, `super()`, overriding, polymorphism |
| `test_data_loader.py` | 27 | `load()`, `split()`, `stats()`, magic methods, CSV |
| `test_trainer.py` | 46 | `run()`, aggregation, `MultiTrainer`, `TrainingResult` |
| **Total** | **167** | |

---

## Extending the Framework

Adding a new model requires **zero changes to existing code**.

```python
# Step 1 — Create a config
rf_config = ModelConfig("RandomForest", learning_rate=0.01, epochs=50)

# Step 2 — Subclass BaseModel, override the three abstract methods
class RandomForestModel(BaseModel):
    def __init__(self, config: ModelConfig, n_trees: int = 100):
        super().__init__(config)       # ← always call super()
        self.n_trees = n_trees         # child-specific attribute

    def get_metric_name(self) -> str:
        return "AUC"

    def train(self, data: list) -> None:
        print(f"Training {self.n_trees} trees...")
        self._mark_trained()           # sets is_trained + timestamp

    def evaluate(self, data: list) -> float:
        return 0.94

# Step 3 — Pass to Trainer — polymorphism handles the rest
loader = DataLoader("MyData")
loader.load([...])
Trainer(RandomForestModel(rf_config), loader).run()
```

---

## Configuration Files

JSON configs can be loaded via CLI or `ModelConfig.from_json()`:

```json
{
  "model_name": "LinearRegression",
  "learning_rate": 0.01,
  "epochs": 10,
  "batch_size": 32,
  "seed": 42,
  "extra": {}
}
```

```python
cfg = ModelConfig.from_json("configs/linear_regression.json")
model = LinearRegressionModel(cfg)
```

---

## Requirements

- Python 3.10+  
- No third-party packages required for the framework itself  
- `pytest` for running the test suite only

---

*Mini AI Model Trainer Framework — v1.0.0 — ML Engineering Team*
# Mini-AI-Model-Trainer-Framework
# Mini-AI-Model-Trainer-Framework
