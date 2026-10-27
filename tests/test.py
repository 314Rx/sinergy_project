"""
Юнит-тесты ключевых компонентов.
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from src import analyze, clean, load, ml, validate


class Testload(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    def test_load_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.csv")
            self.df.to_csv(path, index=False)
            loaded = load.load_csv(path)
            pd.testing.assert_frame_equal(loaded, self.df)

    def test_load_excel(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.xlsx")
            self.df.to_excel(path, index=False)
            loaded = load.load_excel(path)
            pd.testing.assert_frame_equal(loaded, self.df)


class Testvalidate(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3, 100, 2],  # 100 — выброс
            "y": [1.0, np.nan, 3.0, 4.0, 5.0],
        })

    def test_check_duplicates(self):
        report = validate.check_duplicates(self.df)
        self.assertEqual(report["duplicates"], 0)

    def test_check_missing(self):
        report = validate.check_missing(self.df)
        self.assertEqual(report["y"]["count"], 1)

    def test_detect_outliers_iqr(self):
        report = validate.detect_outliers_iqr(self.df, ["x"])
        self.assertEqual(report["x"]["count"], 1)

    def test_detect_outliers_zscore(self):
        df = pd.DataFrame({"x": [2.0] * 19 + [100.0]})  # 20 точек, одна аномалия
        report = validate.detect_outliers_zscore(df, ["x"])
        self.assertEqual(report["x"]["count"], 1)

    def test_check_dtypes_mismatch(self):
        report = validate.check_dtypes(self.df, {"x": "int64", "z": "float64"})
        self.assertEqual(len(report["mismatches"]), 1)
        self.assertEqual(report["mismatches"][0]["column"], "z")


class Testclean(unittest.TestCase):
    def test_handle_missing_median(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = clean.handle_missing(df, strategy="median")
        self.assertFalse(result["x"].isna().any())
        self.assertEqual(result["x"].iloc[1], 2.0)

    def test_handle_missing_drop(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = clean.handle_missing(df, strategy="drop")
        self.assertEqual(len(result), 2)

    def test_drop_duplicates(self):
        df = pd.DataFrame({"x": [1, 1, 2]})
        self.assertEqual(len(clean.drop_duplicates(df)), 2)

    def test_encode_onehot(self):
        df = pd.DataFrame({"city": ["a", "b", "a"], "v": [1, 2, 3]})
        result = clean.encode_categorical(df, ["city"], method="onehot")
        self.assertIn("city_b", result.columns)
        self.assertNotIn("city", result.columns)

    def test_encode_label(self):
        df = pd.DataFrame({"city": ["a", "b", "a"]})
        result = clean.encode_categorical(df, ["city"], method="label")
        self.assertTrue(set(result["city"]).issubset({0, 1}))

    def test_scale_minmax(self):
        df = pd.DataFrame({"x": [0.0, 5.0, 10.0]})
        result = clean.scale_numeric(df, method="minmax")
        self.assertAlmostEqual(result["x"].min(), 0.0)
        self.assertAlmostEqual(result["x"].max(), 1.0)

    def test_parse_dates(self):
        df = pd.DataFrame({"d": ["2026-01-01", "01.02.2026"]})
        result = clean.parse_dates(df, ["d"], dayfirst=False)
        self.assertEqual(str(result["d"].dtype), "datetime64[us]")


class Testanalyze(unittest.TestCase):
    def test_basic_stats(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        stats = analyze.basic_stats(df)
        self.assertAlmostEqual(stats.loc["x", "mean"], 2.5)

    def test_decomposition_columns(self):
        idx = pd.date_range("2026-01-01", periods=24, freq="MS")
        series = pd.Series(range(24), index=idx, dtype=float)
        result = analyze.time_series_decomposition(series, period=12)
        for col in ("original", "trend", "seasonal", "residual"):
            self.assertIn(col, result.columns)

    def test_detect_anomalies(self):
        series = pd.Series([1, 2, 2, 1, 2, 100])
        mask = analyze.detect_anomalies(series, method="iqr")
        self.assertTrue(bool(mask.iloc[-1]))


class TestML(unittest.TestCase):
    def test_train_regression(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"x": rng.rand(100)})
        y = 3 * X["x"] + 1 + rng.normal(0, 0.01, 100)
        result = ml.train_regression(X, y)
        self.assertGreater(result["metrics"]["R2"], 0.9)
        self.assertIn("RMSE", result["metrics"])

    def test_train_classifier(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"x": rng.rand(200)})
        y = pd.Series((X["x"] > 0.5).astype(int))
        result = ml.train_classifier(X, y)
        self.assertGreater(result["metrics"]["accuracy"], 0.9)
        self.assertIn("ROC_AUC", result["metrics"])


if __name__ == "__main__":
    unittest.main()
