"""
main.py
=======
Entry point for the Mini AI Model Trainer Framework.

Demonstrates the full pipeline:
    1. Build ModelConfig objects        (composition)
    2. Instantiate concrete models      (inheritance + super())
    3. Load dataset via DataLoader      (aggregation)
    4. Run each model via MultiTrainer  (polymorphism)
    5. Print the comparison report
"""

import sys
import os

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(__file__))

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

logger = get_logger("ai_trainer", log_dir="logs")


def main() -> None:
    logger.info("Mini AI Model Trainer Framework v1.0.0 — starting")

    # ── 1. ModelConfig — COMPOSITION objects ──────────────────────────────────
    #    Each config is created once and passed (composed) into a model.
    print("\n" + "═" * 60)
    print("  STEP 1 — Model Configurations")
    print("═" * 60)

    lr_config = ModelConfig(
        model_name="LinearRegression",
        learning_rate=0.01,
        epochs=10,
        batch_size=32,
        seed=42,
    )
    nn_config = ModelConfig(
        model_name="NeuralNetwork",
        learning_rate=0.001,
        epochs=20,
        batch_size=64,
        seed=7,
    )
    svm_config = ModelConfig(
        model_name="SupportVectorMachine",
        learning_rate=0.005,
        epochs=15,
        batch_size=16,
        seed=99,
    )

    print(lr_config)   # __repr__ in action
    print(nn_config)
    print(svm_config)

    logger.info("Configs created: LR / NN / SVM")

    # ── 2. Concrete models — INHERITANCE + super() ────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 2 — Model Instantiation")
    print("═" * 60)

    lr_model  = LinearRegressionModel(lr_config)
    nn_model  = NeuralNetworkModel(nn_config, layers=[64, 32, 1], activation="relu")
    svm_model = SVMModel(svm_config, kernel="rbf", C=1.5)

    # Class attribute — shared across ALL BaseModel instances
    print(f"\n  Models created so far : {BaseModel.model_count}")
    print(f"  {lr_model.describe()}")
    print(f"  {nn_model.describe()}")
    print(f"  {svm_model.describe()}")

    logger.info(f"Models instantiated (total: {BaseModel.model_count})")

    # ── 3. DataLoader — AGGREGATION target ────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 3 — Data Loading")
    print("═" * 60)

    loader = DataLoader(name="HousePrices")
    loader.load([
        1.2, 3.4, 2.1, 5.6, 4.8,
        6.3, 7.0, 8.1, 9.5, 2.9,
        4.1, 5.5, 6.8, 3.3, 7.7,
    ])
    loader.split(train=0.7, val=0.15, test=0.15, seed=42)

    stats = loader.stats()
    print(f"\n  Dataset stats: {stats}")
    logger.info(f"DataLoader ready: {loader}")

    # ── 4. MultiTrainer — POLYMORPHISM ────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 4 — Training Pipeline (Polymorphism)")
    print("═" * 60)

    # MultiTrainer treats every model identically through BaseModel interface.
    # No isinstance checks — pure polymorphic dispatch.
    multi = MultiTrainer(
        models=[lr_model, nn_model, svm_model],
        data_loader=loader,
    )
    results = multi.run_all()

    logger.info(f"All models trained. Results: {[r.to_dict() for r in results]}")

    # ── 5. Report ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 5 — Results Summary")
    print("═" * 60)
    multi.print_report()

    # ── 6. Best model (demo of result querying) ────────────────────────────────
    best_high = multi.get_best(higher_is_better=True)
    best_low  = multi.get_best(higher_is_better=False)
    print(f"  Best model (higher metric) : {best_high.model_name}  ({best_high.metric_name}={best_high.metric_value})")
    print(f"  Best model (lower metric)  : {best_low.model_name}   ({best_low.metric_name}={best_low.metric_value})\n")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
