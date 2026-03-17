"""
framework/config.py
===================
ModelConfig — Composition object owned by every BaseModel.

OOP Concepts demonstrated:
    - Instance attributes  : model_name, learning_rate, epochs, batch_size, seed
    - Magic methods        : __repr__, __eq__, __hash__
    - Instance methods     : summary(), validate(), to_dict(), from_dict()
    - Class method         : from_dict() (alternative constructor)
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


class ModelConfig:
    """
    Stores and validates hyperparameters for an ML model.

    This class is used via COMPOSITION — it is created externally and
    injected into a BaseModel instance. The model owns the config for
    its entire lifetime.

    Attributes
    ----------
    model_name    : Human-readable model identifier.
    learning_rate : Step size for gradient updates (default 0.01).
    epochs        : Number of full passes over the training data (default 10).
    batch_size    : Samples per gradient-update step (default 32).
    seed          : Random seed for reproducibility (default 42).
    extra         : Optional dict for model-specific hyperparameters.
    """

    # ── Construction ─────────────────────────────────────────────────────────
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 0.01,
        epochs: int = 10,
        batch_size: int = 32,
        seed: int = 42,
        extra: Optional[dict] = None,
    ) -> None:
        self.model_name: str = model_name
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.seed: int = seed
        self.extra: dict = extra or {}

        self.validate()  # always validate on creation

    # ── Validation ────────────────────────────────────────────────────────────
    def validate(self) -> None:
        """Raise ValueError if any hyperparameter is out of range."""
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError("model_name must be a non-empty string.")
        if not (0 < self.learning_rate < 1):
            raise ValueError(f"learning_rate must be in (0, 1), got {self.learning_rate}.")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Return config as a plain dictionary (JSON-serialisable)."""
        return {
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Alternative constructor — build a ModelConfig from a dictionary."""
        return cls(
            model_name=data["model_name"],
            learning_rate=data.get("learning_rate", 0.01),
            epochs=data.get("epochs", 10),
            batch_size=data.get("batch_size", 32),
            seed=data.get("seed", 42),
            extra=data.get("extra", {}),
        )

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """Load a ModelConfig from a JSON file."""
        with open(path, "r") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def save_json(self, path: str) -> None:
        """Persist this config to a JSON file."""
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def summary(self) -> dict:
        """Return a human-friendly summary dictionary."""
        return self.to_dict()

    # ── Magic Methods ─────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"[Config] {self.model_name:<20} "
            f"| lr={self.learning_rate:<6} "
            f"| epochs={self.epochs:<4} "
            f"| batch={self.batch_size}"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash((self.model_name, self.learning_rate, self.epochs, self.batch_size, self.seed))
