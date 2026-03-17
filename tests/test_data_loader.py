"""
tests/test_data_loader.py
=========================
Unit tests for DataLoader.

Covers:
    - load() stores data and returns self
    - get_batch() returns full dataset or a sub-sample
    - split() partitions correctly and validates ratios
    - get_split() returns the named partition
    - stats() computes correct statistics
    - __len__ and __iter__
    - __repr__ format
    - Error conditions (empty dataset, bad ratios, missing split)
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from framework.data_loader import DataLoader


class TestDataLoaderLoad(unittest.TestCase):

    def test_load_stores_data(self):
        dl = DataLoader("Test")
        dl.load([1, 2, 3])
        self.assertEqual(dl.dataset, [1, 2, 3])

    def test_load_returns_self_for_chaining(self):
        dl = DataLoader()
        result = dl.load([1, 2])
        self.assertIs(result, dl)

    def test_load_replaces_existing_data(self):
        dl = DataLoader()
        dl.load([1, 2, 3])
        dl.load([9, 8])
        self.assertEqual(dl.dataset, [9, 8])

    def test_load_empty_list(self):
        dl = DataLoader()
        dl.load([])
        self.assertEqual(dl.dataset, [])

    def test_load_non_list_raises(self):
        dl = DataLoader()
        with self.assertRaises(TypeError):
            dl.load((1, 2, 3))   # tuple, not list

    def test_load_accepts_nested_lists(self):
        dl = DataLoader()
        dl.load([[1, 2], [3, 4]])
        self.assertEqual(len(dl.dataset), 2)


class TestDataLoaderGetBatch(unittest.TestCase):

    def setUp(self):
        self.dl = DataLoader("Test")
        self.dl.load([10, 20, 30, 40, 50])

    def test_get_batch_none_returns_all(self):
        batch = self.dl.get_batch()
        self.assertEqual(sorted(batch), [10, 20, 30, 40, 50])

    def test_get_batch_size_returns_subset(self):
        batch = self.dl.get_batch(size=3)
        self.assertEqual(len(batch), 3)

    def test_get_batch_subset_items_are_in_dataset(self):
        batch = self.dl.get_batch(size=3)
        for item in batch:
            self.assertIn(item, self.dl.dataset)

    def test_get_batch_size_larger_than_dataset_returns_all(self):
        batch = self.dl.get_batch(size=100)
        self.assertEqual(len(batch), 5)

    def test_get_batch_empty_dataset_raises(self):
        dl = DataLoader()
        with self.assertRaises(RuntimeError):
            dl.get_batch()

    def test_get_batch_does_not_mutate_dataset(self):
        original = list(self.dl.dataset)
        self.dl.get_batch(size=3)
        self.assertEqual(self.dl.dataset, original)


class TestDataLoaderSplit(unittest.TestCase):

    def setUp(self):
        self.dl = DataLoader("Test")
        self.dl.load(list(range(100)))

    def test_split_creates_three_partitions(self):
        self.dl.split(0.7, 0.15, 0.15)
        for key in ("train", "val", "test"):
            self.assertIn(key, self.dl._splits)

    def test_split_total_count_matches_dataset(self):
        self.dl.split(0.7, 0.15, 0.15)
        total = (
            len(self.dl._splits["train"]) +
            len(self.dl._splits["val"])   +
            len(self.dl._splits["test"])
        )
        self.assertEqual(total, 100)

    def test_split_ratios_sum_violation_raises(self):
        with self.assertRaises(ValueError):
            self.dl.split(0.6, 0.2, 0.1)   # sums to 0.9

    def test_split_empty_dataset_raises(self):
        dl = DataLoader()
        with self.assertRaises(RuntimeError):
            dl.split()

    def test_split_returns_self(self):
        result = self.dl.split()
        self.assertIs(result, self.dl)

    def test_get_split_train(self):
        self.dl.split(0.8, 0.1, 0.1)
        train = self.dl.get_split("train")
        self.assertAlmostEqual(len(train), 80, delta=2)

    def test_get_split_invalid_name_raises(self):
        self.dl.split()
        with self.assertRaises(KeyError):
            self.dl.get_split("nonexistent")

    def test_split_no_overlap_between_partitions(self):
        self.dl.split(0.7, 0.15, 0.15, shuffle=False)
        train_set = set(map(str, self.dl._splits["train"]))
        val_set   = set(map(str, self.dl._splits["val"]))
        test_set  = set(map(str, self.dl._splits["test"]))
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)


class TestDataLoaderStats(unittest.TestCase):

    def test_stats_count(self):
        dl = DataLoader()
        dl.load([1.0, 2.0, 3.0, 4.0, 5.0])
        s = dl.stats()
        self.assertEqual(s["count"], 5)

    def test_stats_min_max(self):
        dl = DataLoader()
        dl.load([3.0, 1.0, 4.0, 1.5, 9.0])
        s = dl.stats()
        self.assertAlmostEqual(s["min"], 1.0)
        self.assertAlmostEqual(s["max"], 9.0)

    def test_stats_mean(self):
        dl = DataLoader()
        dl.load([2.0, 4.0, 6.0])
        s = dl.stats()
        self.assertAlmostEqual(s["mean"], 4.0)

    def test_stats_std_dev_of_constant(self):
        dl = DataLoader()
        dl.load([5.0, 5.0, 5.0])
        s = dl.stats()
        self.assertAlmostEqual(s["std_dev"], 0.0)

    def test_stats_empty_numeric_returns_count_zero(self):
        dl = DataLoader()
        dl.load(["a", "b"])
        s = dl.stats()
        self.assertEqual(s["count"], 0)

    def test_stats_nested_list_flattens(self):
        dl = DataLoader()
        dl.load([[1.0, 2.0], [3.0, 4.0]])
        s = dl.stats()
        self.assertEqual(s["count"], 4)


class TestDataLoaderMagicMethods(unittest.TestCase):

    def setUp(self):
        self.dl = DataLoader("MagicTest")
        self.dl.load([1, 2, 3, 4, 5])

    def test_len(self):
        self.assertEqual(len(self.dl), 5)

    def test_iter(self):
        items = list(self.dl)
        self.assertEqual(items, [1, 2, 3, 4, 5])

    def test_repr_contains_name(self):
        self.assertIn("MagicTest", repr(self.dl))

    def test_repr_contains_sample_count(self):
        self.assertIn("5", repr(self.dl))


class TestDataLoaderLoadCSV(unittest.TestCase):

    def test_load_csv_reads_numeric_data(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("x,y\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
            path = f.name

        try:
            dl = DataLoader("CSV")
            dl.load_csv(path, skip_header=True)
            self.assertEqual(len(dl.dataset), 3)
        finally:
            os.unlink(path)

    def test_load_csv_missing_file_raises(self):
        dl = DataLoader()
        with self.assertRaises(FileNotFoundError):
            dl.load_csv("/nonexistent/path.csv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
