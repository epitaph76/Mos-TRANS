import contextlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from mos_trans.modeling.catboost import run, select_feature_columns


class RegressionSubmissionTests(unittest.TestCase):
    def test_factual_arrival_rejected(self):
        with self.assertRaises(ValueError):
            select_feature_columns(pd.DataFrame({"time_fact_begin": ["2026-01-01"]}))

    def test_local_test_then_refit_train_and_test_for_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self._frame(20, "train")
            test = self._frame(8, "test")
            validate = self._frame(3, "validate").assign(target_delay_s=np.nan)
            for name, frame in (("train", train), ("test", test), ("validate", validate)):
                frame.to_parquet(root / f"{name}_features.parquet")
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("sample_submission.csv",
                                  "sample_id;prediction\nvalidate_2;0\nvalidate_0;0\nvalidate_1;0\n")

            def small_model(iterations=3000, verbose=100):
                return CatBoostRegressor(loss_function="MAE", eval_metric="MAE",
                                         iterations=min(iterations, 6), depth=2, learning_rate=.1,
                                         random_seed=42, verbose=False, allow_writing_files=False)

            with patch("mos_trans.modeling.catboost.create_model", side_effect=small_model):
                with contextlib.redirect_stdout(io.StringIO()):
                    report = run(root, root / "out", archive)
            self.assertEqual(report["final_model_training_rows"], 28)
            self.assertEqual(report["final_model_training_splits"], ["train", "test"])
            self.assertTrue((root / "out/catboost_residual_local_test.cbm").is_file())
            self.assertTrue((root / "out/catboost_residual.cbm").is_file())
            submission = pd.read_csv(root / "out/submission.csv", sep=";")
            self.assertEqual(submission.columns.tolist(), ["sample_id", "prediction"])
            self.assertEqual(submission.sample_id.tolist(), ["validate_2", "validate_0", "validate_1"])
            self.assertTrue(np.isfinite(submission.prediction).all())

    @staticmethod
    def _frame(n: int, split: str) -> pd.DataFrame:
        value = np.arange(n)
        return pd.DataFrame({"sample_id": [f"{split}_{i}" for i in value],
                             "tr_id": [str(100 + i % 3) for i in value],
                             "cur_dev_s": value.astype(float),
                             "target_delay_s": value.astype(float) + 2 * (value % 4),
                             "target_stop_id": [f"stop_{i % 3}" for i in value],
                             "schedule_target_address": [f"addr_{i % 3}" for i in value],
                             "realtime_speed_last": value.astype(float)})


if __name__ == "__main__":
    unittest.main()
