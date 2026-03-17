"""
framework/base_model.py
=======================
BaseModel — Abstract Base Class that defines the contract for all ML models.

OOP Concepts demonstrated:
    - Abstraction     : ABC + @abstractmethod (train, evaluate)
    - Class attribute : model_count — shared across ALL instances
    - Composition     : owns a ModelConfig (created externally, lives with model)
    - Instance method : describe(), reset(), get_metric_name()
    - Magic methods   : __repr__, __str__
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from framework.config import ModelConfig


class BaseModel(ABC):
    """
    Abstract base class for all ML models in the framework.

    Subclasses MUST override:
        - train(data)    → None
        - evaluate(data) → float

    Subclasses SHOULD call super().__init__(config) to ensure
    model_count is incremented and shared attributes are set.

    Class Attributes
    ----------------
    model_count : int
        Total number of BaseModel instances ever created (any subclass).

    Instance Attributes
    -------------------
    config      : ModelConfig   — hyperparameters (COMPOSITION)
    is_trained  : bool          — True after train() completes
    trained_at  : datetime|None — timestamp of last successful train()
    history     : list[dict]    — per-epoch loss/metric log populated by subclasses
    """

    # ── Class attribute ───────────────────────────────────────────────────────
    model_count: int = 0  # shared by ALL instances of BaseModel and subclasses

    # ── Initialisation ────────────────────────────────────────────────────────
    def __init__(self, config: ModelConfig) -> None:
        # COMPOSITION — BaseModel owns its ModelConfig for its entire lifetime
        self.config: ModelConfig = config
        self.is_trained: bool = False
        self.trained_at: Optional[datetime] = None
        self.history: list[dict] = []          # training log (filled by subclasses)

        BaseModel.model_count += 1             # update shared class attribute

    # ── Abstract interface (contract for all subclasses) ──────────────────────
    @abstractmethod
    def train(self, data: list) -> None:
        """
        Train the model on the provided dataset.

        Subclasses must set self.is_trained = True upon completion.

        Parameters
        ----------
        data : list
            Raw training samples (format is subclass-specific).
        """
        ...

    @abstractmethod
    def evaluate(self, data: list) -> float:
        """
        Evaluate the trained model and return a scalar performance metric.

        Parameters
        ----------
        data : list
            Evaluation samples.

        Returns
        -------
        float
            A scalar metric (e.g. MSE for regression, accuracy for classifiers).
        """
        ...

    @abstractmethod
    def get_metric_name(self) -> str:
        """Return the name of the primary evaluation metric (e.g. 'MSE', 'Accuracy')."""
        ...

    # ── Concrete shared methods ───────────────────────────────────────────────
    def describe(self) -> str:
        """Return a one-line human-readable model description."""
        status = "trained" if self.is_trained else "untrained"
        ts = f" @ {self.trained_at:%Y-%m-%d %H:%M:%S}" if self.trained_at else ""
        return f"{self.config.model_name} ({status}{ts}) | {self.config}"

    def reset(self) -> None:
        """Reset training state so the model can be re-trained from scratch."""
        self.is_trained = False
        self.trained_at = None
        self.history.clear()

    def _mark_trained(self) -> None:
        """
        Helper for subclasses: mark the model as trained and record timestamp.
        Call this at the end of a successful train() implementation.
        """
        self.is_trained = True
        self.trained_at = datetime.now()

    def _log_epoch(self, epoch: int, loss: float, metric: float) -> None:
        """
        Helper for subclasses: append an epoch record to self.history.

        Parameters
        ----------
        epoch  : 1-based epoch number
        loss   : training loss for this epoch
        metric : evaluation metric for this epoch
        """
        self.history.append({"epoch": epoch, "loss": round(loss, 6), self.get_metric_name(): round(metric, 6)})

    # ── Magic methods ─────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config!r}, trained={self.is_trained})"

    def __str__(self) -> str:
        return self.describe()
