"""
framework/__init__.py
=====================
Public API of the Mini AI Model Trainer Framework.

Import directly from `framework`:

    from framework import ModelConfig, BaseModel
    from framework import LinearRegressionModel, NeuralNetworkModel, SVMModel
    from framework import DataLoader, Trainer, MultiTrainer
"""

from framework.config import ModelConfig
from framework.base_model import BaseModel
from framework.models import LinearRegressionModel, NeuralNetworkModel, SVMModel
from framework.data_loader import DataLoader
from framework.trainer import Trainer, MultiTrainer, TrainingResult
from framework.logger import get_logger

__all__ = [
    "ModelConfig",
    "BaseModel",
    "LinearRegressionModel",
    "NeuralNetworkModel",
    "SVMModel",
    "DataLoader",
    "Trainer",
    "MultiTrainer",
    "TrainingResult",
    "get_logger",
]

__version__ = "1.0.0"
__author__  = "ML Engineering Team"
