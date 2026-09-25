import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mos_trans.modeling.compare_risk import (
    CompareConfig, _fit_model, _lightgbm_frames, _predict_fitted, predict, quantile_risk,
)


class RiskComparisonTests(unittest.TestCase):
    def test_quantile_probability_monotone_with_delay(self):
        quantiles = np.asarray([[0, 100, 200], [100, 200, 300]], dtype=float)
        probability = quantile_risk(quantiles, np.asarray([0, 0]))
        self.assertLess(probability[0], probability[1])
        np.testing.assert_allclose(probability, [0.3, 0.7])

    def test_unseen_lightgbm_categories_become_missing(self):
        fit = pd.DataFrame({"tr_id": ["1"], "target_stop_id": ["a"],
                            "schedule_target_address": ["A"], "cur_dev_s": [1]})
        other = pd.DataFrame({"tr_id": ["2"], "target_stop_id": ["b"],
                              "schedule_target_address": ["B"], "cur_dev_s": [2]})
        _, transformed = _lightgbm_frames(fit, other, list(fit.columns))
        self.assertTrue(transformed.tr_id.isna().all())
        self.assertTrue(transformed.target_stop_id.isna().all())

    def test_checkpoint_predict_roundtrip(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "data/processed/train_features.parquet"
        if not path.is_file():
            self.skipTest("Local dataset absent")
        train = pd.read_parquet(path).iloc[:200].copy()
        config = CompareConfig(lightgbm_trees=5, threads=1)
        artifact = _fit_model("lightgbm_base", train, config)
        before = _predict_fitted(artifact, train.iloc[:10])
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "risk.joblib"
            joblib.dump(artifact, model_path)
            after = predict(model_path, train.iloc[:10])
            with self.assertRaises(ValueError):
                predict(model_path, train.iloc[:1].assign(time_fact_begin="2026-01-01"))
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
