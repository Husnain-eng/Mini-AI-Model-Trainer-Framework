"""
framework/trainer.py
====================
Trainer — Orchestrates the full train → evaluate pipeline.

OOP Concepts demonstrated:
    - Aggregation  : holds references to BaseModel + DataLoader (doesn't own them)
    - Polymorphism : run() works identically for any BaseModel subclass
    - Instance methods : run(), run_all(), get_results(), print_report()
    - Magic methods    : __repr__
"""

from __future__ import annotations
import time
from datetime import datetime
from typing import Optional

from framework.base_model import BaseModel
from framework.data_loader import DataLoader


class TrainingResult:
    """
    Immutable record of a single model's training + evaluation run.

    Attributes
    ----------
    model_name   : Name from ModelConfig.
    metric_name  : e.g. 'MSE', 'Accuracy', 'F1Score'
    metric_value : Scalar result from evaluate().
    duration_s   : Wall-clock seconds for the full run.
    trained_at   : Datetime when training completed.
    history      : Per-epoch log from the model.
    """

    def __init__(
        self,
        model_name: str,
        metric_name: str,
        metric_value: float,
        duration_s: float,
        trained_at: Optional[datetime],
        history: list[dict],
    ) -> None:
        self.model_name = model_name
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.duration_s = round(duration_s, 4)
        self.trained_at = trained_at
        self.history = history

    @property
    def epochs_logged(self) -> int:
        return len(self.history)

    def to_dict(self) -> dict:
        return {
            "model_name":   self.model_name,
            "metric_name":  self.metric_name,
            "metric_value": self.metric_value,
            "duration_s":   self.duration_s,
            "trained_at":   str(self.trained_at),
            "epochs_logged": self.epochs_logged,
        }

    def __repr__(self) -> str:
        return (
            f"TrainingResult({self.model_name!r} | "
            f"{self.metric_name}={self.metric_value} | "
            f"{self.duration_s}s)"
        )


class Trainer:
    """
    Orchestrates the full pipeline for a single model:
        DataLoader.get_batch()  →  model.train()  →  model.evaluate()

    AGGREGATION: Trainer holds references to model and data_loader but
    does NOT own them — they are created externally and can outlive any Trainer.

    POLYMORPHISM: run() calls model.train() and model.evaluate() through
    the BaseModel interface. Python's runtime dispatch picks the correct
    concrete implementation without any isinstance checks.

    Attributes
    ----------
    model       : Any concrete BaseModel subclass.
    data_loader : A DataLoader instance (passed in, not created here).
    result      : TrainingResult after run() completes (None before).
    """

    def __init__(self, model: BaseModel, data_loader: DataLoader) -> None:
        # AGGREGATION — references only, no ownership
        self.model: BaseModel = model
        self.data_loader: DataLoader = data_loader
        self.result: Optional[TrainingResult] = None

    def run(self) -> TrainingResult:
        """
        Execute the full pipeline for self.model.

        Steps
        -----
        1. Fetch data batch from DataLoader.
        2. Call model.train(data)   — polymorphic dispatch.
        3. Call model.evaluate(data) — polymorphic dispatch.
        4. Wrap results in a TrainingResult and store in self.result.

        Returns
        -------
        TrainingResult
        """
        data = self.data_loader.get_batch()

        t0 = time.perf_counter()
        self.model.train(data)          # POLYMORPHISM — dispatches to subclass
        metric_value = self.model.evaluate(data)  # POLYMORPHISM
        duration = time.perf_counter() - t0

        self.result = TrainingResult(
            model_name=self.model.config.model_name,
            metric_name=self.model.get_metric_name(),
            metric_value=metric_value,
            duration_s=duration,
            trained_at=self.model.trained_at,
            history=list(self.model.history),
        )
        return self.result

    def __repr__(self) -> str:
        return (
            f"Trainer(model={self.model.config.model_name!r}, "
            f"loader={self.data_loader.name!r}, "
            f"result={'pending' if self.result is None else self.result.metric_value})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MultiTrainer — runs a list of models over the same DataLoader
# ─────────────────────────────────────────────────────────────────────────────
class MultiTrainer:
    """
    Runs multiple models through the same pipeline in sequence.

    POLYMORPHISM: Each model in self.models is a different concrete subclass
    of BaseModel. run_all() treats them identically via the shared interface.

    AGGREGATION: Both the model list and data_loader are passed in externally.
    """

    def __init__(self, models: list[BaseModel], data_loader: DataLoader) -> None:
        self.models: list[BaseModel] = models          # AGGREGATION
        self.data_loader: DataLoader = data_loader     # AGGREGATION
        self.results: list[TrainingResult] = []

    def run_all(self) -> list[TrainingResult]:
        """Train and evaluate every model. Returns all TrainingResult records."""
        self.results.clear()
        for model in self.models:          # POLYMORPHISM — no type checks needed
            trainer = Trainer(model=model, data_loader=self.data_loader)
            result = trainer.run()
            self.results.append(result)
        return self.results

    def print_report(self) -> None:
        """Print a formatted comparison table of all results."""
        if not self.results:
            print("No results yet. Call run_all() first.")
            return

        width = 65
        print(f"\n{'═'*width}")
        print(f"  {'TRAINING REPORT':^{width-4}}")
        print(f"{'═'*width}")
        print(f"  {'Model':<22} {'Metric':<12} {'Value':>10}  {'Time':>8}  Epochs")
        print(f"  {'─'*22} {'─'*12} {'─'*10}  {'─'*8}  {'─'*6}")

        for r in self.results:
            epochs = len(r.history)
            print(
                f"  {r.model_name:<22} "
                f"{r.metric_name:<12} "
                f"{r.metric_value:>10.4f}  "
                f"{r.duration_s:>7.3f}s  "
                f"{epochs:>6}"
            )

        print(f"{'═'*width}")
        print(f"  Models trained : {len(self.results)}")
        print(f"  Total models created (all time) : {BaseModel.model_count}")
        print(f"{'═'*width}\n")

    def get_best(self, higher_is_better: bool = True) -> Optional[TrainingResult]:
        """Return the result with the best metric value."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.metric_value) if higher_is_better \
               else min(self.results, key=lambda r: r.metric_value)
