"""
framework/data_loader.py
========================
DataLoader — Independent dataset container used via AGGREGATION in Trainer.

OOP Concepts demonstrated:
    - Instance attributes : name, dataset, _splits
    - Instance methods    : load(), load_csv(), get_batch(), split(), stats()
    - Magic methods       : __repr__, __len__, __iter__
    - Aggregation         : passed into Trainer externally (Trainer does NOT own it)
"""

from __future__ import annotations
import csv
import os
import random
from typing import Iterator, Optional, Tuple


class DataLoader:
    """
    Holds, validates, and serves dataset samples to models.

    DataLoader is an AGGREGATION target — it is created independently and
    injected into a Trainer. It can outlive any Trainer that uses it.

    Attributes
    ----------
    name    : Human-readable dataset label.
    dataset : The raw list of samples.
    """

    def __init__(self, name: str = "DefaultDataset") -> None:
        self.name: str = name
        self.dataset: list = []
        self._splits: dict[str, list] = {}   # train / val / test splits

    # ── Loading ───────────────────────────────────────────────────────────────
    def load(self, data: list) -> "DataLoader":
        """Load a Python list as the dataset. Returns self for chaining."""
        if not isinstance(data, list):
            raise TypeError(f"Expected list, got {type(data).__name__}.")
        self.dataset = list(data)
        print(f"[DataLoader] '{self.name}' — loaded {len(data)} samples from memory.")
        return self

    def load_csv(self, path: str, skip_header: bool = True) -> "DataLoader":
        """Load numeric data from a CSV file."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV not found: {path}")

        rows = []
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            if skip_header:
                next(reader, None)
            for row in reader:
                try:
                    rows.append([float(x) for x in row])
                except ValueError:
                    continue  # skip non-numeric rows

        self.dataset = rows
        print(f"[DataLoader] '{self.name}' — loaded {len(rows)} rows from '{path}'.")
        return self

    # ── Access ────────────────────────────────────────────────────────────────
    def get_batch(self, size: Optional[int] = None) -> list:
        """
        Return a batch of samples.

        Parameters
        ----------
        size : If None, return the entire dataset.
               If int, return a random sub-sample of that size.
        """
        if not self.dataset:
            raise RuntimeError("DataLoader is empty. Call load() first.")
        if size is None or size >= len(self.dataset):
            return list(self.dataset)
        return random.sample(self.dataset, size)

    # ── Splits ────────────────────────────────────────────────────────────────
    def split(
        self,
        train: float = 0.7,
        val: float = 0.15,
        test: float = 0.15,
        shuffle: bool = True,
        seed: int = 42,
    ) -> "DataLoader":
        """
        Split dataset into train / val / test partitions.

        Ratios must sum to 1.0 (within floating-point tolerance).
        """
        if abs(train + val + test - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0 (got {train+val+test:.4f}).")
        if not self.dataset:
            raise RuntimeError("Cannot split an empty dataset.")

        data = list(self.dataset)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(data)

        n = len(data)
        t_end = int(n * train)
        v_end = t_end + int(n * val)

        self._splits = {
            "train": data[:t_end],
            "val":   data[t_end:v_end],
            "test":  data[v_end:],
        }
        print(
            f"[DataLoader] '{self.name}' split — "
            f"train={len(self._splits['train'])} "
            f"val={len(self._splits['val'])} "
            f"test={len(self._splits['test'])}"
        )
        return self

    def get_split(self, name: str) -> list:
        """Return a named split ('train', 'val', 'test')."""
        if name not in self._splits:
            raise KeyError(f"Split '{name}' not found. Run split() first.")
        return self._splits[name]

    # ── Statistics ────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        """
        Compute basic statistics for numeric datasets.

        Returns a dict with count, min, max, mean, std_dev.
        Works on flat numeric lists; skips non-numeric items.
        """
        flat = []
        for item in self.dataset:
            if isinstance(item, (int, float)):
                flat.append(item)
            elif isinstance(item, (list, tuple)):
                flat.extend(x for x in item if isinstance(x, (int, float)))

        if not flat:
            return {"count": 0, "note": "No numeric data found."}

        n = len(flat)
        mean = sum(flat) / n
        variance = sum((x - mean) ** 2 for x in flat) / n
        return {
            "count":   n,
            "min":     round(min(flat), 6),
            "max":     round(max(flat), 6),
            "mean":    round(mean, 6),
            "std_dev": round(variance ** 0.5, 6),
        }

    # ── Magic methods ─────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        splits_info = f", splits={list(self._splits.keys())}" if self._splits else ""
        return f"DataLoader(name={self.name!r}, samples={len(self.dataset)}{splits_info})"

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator:
        return iter(self.dataset)
