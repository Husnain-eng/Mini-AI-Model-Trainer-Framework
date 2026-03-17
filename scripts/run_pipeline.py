"""
scripts/run_pipeline.py
=======================
Command-line interface for the Mini AI Model Trainer Framework.

Usage examples
--------------
  # Run all models (default)
  python scripts/run_pipeline.py

  # Run a specific model type
  python scripts/run_pipeline.py --model lr
  python scripts/run_pipeline.py --model nn --layers 128 64 32 1
  python scripts/run_pipeline.py --model svm --kernel linear --C 0.5

  # Tune hyperparameters
  python scripts/run_pipeline.py --model lr --lr 0.005 --epochs 20

  # Load config from JSON
  python scripts/run_pipeline.py --config configs/linear_regression.json

  # Save config to JSON
  python scripts/run_pipeline.py --model lr --save-config configs/my_lr.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from framework import (
    ModelConfig,
    BaseModel,
    LinearRegressionModel,
    NeuralNetworkModel,
    SVMModel,
    DataLoader,
    Trainer,
    MultiTrainer,
    get_logger,
)

logger = get_logger("cli", log_dir="logs")

_SAMPLE_DATA = [
    1.2, 3.4, 2.1, 5.6, 4.8, 6.3, 7.0,
    8.1, 9.5, 2.9, 4.1, 5.5, 6.8, 3.3, 7.7,
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Mini AI Model Trainer — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=["lr", "nn", "svm", "all"],
        default="all",
        help="Which model type to train.",
    )
    p.add_argument("--lr",     type=float, default=0.01,  metavar="LEARNING_RATE")
    p.add_argument("--epochs", type=int,   default=10,    metavar="N")
    p.add_argument("--batch",  type=int,   default=32,    metavar="SIZE")
    p.add_argument("--seed",   type=int,   default=42)

    # NN-specific
    p.add_argument("--layers", type=int, nargs="+", default=[64, 32, 1], metavar="N")
    p.add_argument("--activation", default="relu", choices=["relu", "sigmoid", "tanh"])

    # SVM-specific
    p.add_argument("--kernel", default="rbf", choices=["rbf", "linear", "poly"])
    p.add_argument("--C", type=float, default=1.0, metavar="REGULARISATION")

    # Config I/O
    p.add_argument("--config",      default=None, metavar="PATH", help="Load config from JSON.")
    p.add_argument("--save-config", default=None, metavar="PATH", help="Save config to JSON.")

    return p


def run(args: argparse.Namespace) -> None:
    logger.info(f"CLI args: {vars(args)}")

    # ── DataLoader ─────────────────────────────────────────────────────────────
    loader = DataLoader("CLIDataset")
    loader.load(_SAMPLE_DATA)
    loader.split(0.7, 0.15, 0.15, seed=args.seed)

    # ── Build model(s) ─────────────────────────────────────────────────────────
    models: list[BaseModel] = []

    def make_config(name: str) -> ModelConfig:
        if args.config:
            cfg = ModelConfig.from_json(args.config)
            logger.info(f"Config loaded from {args.config}")
            return cfg
        return ModelConfig(name, learning_rate=args.lr, epochs=args.epochs,
                           batch_size=args.batch, seed=args.seed)

    if args.model in ("lr", "all"):
        cfg = make_config("LinearRegression")
        if args.save_config:
            cfg.save_json(args.save_config)
            logger.info(f"Config saved to {args.save_config}")
        models.append(LinearRegressionModel(cfg))

    if args.model in ("nn", "all"):
        cfg = make_config("NeuralNetwork")
        models.append(NeuralNetworkModel(cfg, layers=args.layers, activation=args.activation))

    if args.model in ("svm", "all"):
        cfg = make_config("SupportVectorMachine")
        models.append(SVMModel(cfg, kernel=args.kernel, C=args.C))

    # ── Run pipeline ───────────────────────────────────────────────────────────
    if len(models) == 1:
        t = Trainer(models[0], loader)
        result = t.run()
        print(f"\n  Result: {result}")
    else:
        mt = MultiTrainer(models, loader)
        mt.run_all()
        mt.print_report()

    logger.info(f"CLI run complete. Total models ever created: {BaseModel.model_count}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
