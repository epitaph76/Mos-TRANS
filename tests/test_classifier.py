import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from mos_trans.modeling.classifier import (
    CLASS_NAMES, Config, _model, family_map, family_split, feature_columns,
    labels, model_input, predict,
)


class ClassifierTests(unittest.TestCase):
    def test_boundaries(self):
        actual = labels([-61, -60, 0, 59.99, 60, 299.99, 300])
        np.testing.assert_array_equal(actual, [0, 1, 1, 1, 2, 2, 3])
        self.assertEqual(len(CLASS_NAMES), 4)

    def test_future_column_rejected(self):
        frame = pd.DataFrame({"sample_id": ["a"], "tr_id": ["1"], "cur_dev_s": [0],
                              "target_stop_id": ["stop"], "schedule_target_address": ["address"],
                              "time_fact_begin": ["future"]})
        with self.assertRaises(ValueError):
            feature_columns(frame)
        with self.assertRaises(ValueError):
            model_input(frame, ["tr_id", "target_stop_id", "schedule_target_address", "time_fact_begin"])

    def test_synthetic_parent_stays_out_of_training(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "dataset.zip"
            header = "tr_id,time_begin,geom\n"
            rows = []
            for i in range(13):
                parent = str(100 + i)
                for vehicle in (parent, "900" + parent):
                    rows.append(f"{vehicle},2026-01-01 10:00:00,POINT ({i} 55)\n")
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("train/schedule.csv", header + "".join(rows))
            families = family_map(archive_path, {str(100 + i) for i in range(13)})
            frame = pd.DataFrame({"tr_id": list(families)})
            for seed in (42, 7, 123):
                train, validation, held = family_split(frame, families, seed)
                self.assertEqual(len(held), 3)
                self.assertTrue(all(families[frame.tr_id[i]] not in held for i in train))
                self.assertTrue(all(frame.tr_id[i] in held for i in validation))

    def test_prediction_after_model_reload(self):
        # Use the production feature schema and a tiny, deliberately separable input.
        from mos_trans.modeling.classifier import FEATURES, CATEGORICAL
        frame = pd.DataFrame({name: [0.0] * 16 for name in FEATURES if name not in (*CATEGORICAL, "tr_id")})
        frame["sample_id"] = [str(i) for i in range(16)]
        frame["tr_id"] = [str(100 + i % 4) for i in range(16)]
        frame["target_stop_id"] = [f"s{i % 4}" for i in range(16)]
        frame["schedule_target_address"] = [f"a{i % 4}" for i in range(16)]
        frame["cur_dev_s"] = list(range(16))
        columns = feature_columns(frame)
        model = _model(Config(iterations=5), 5, 42)
        model.fit(model_input(frame, columns), np.arange(16) % 4, cat_features=list(CATEGORICAL))
        expected = model.predict_proba(model_input(frame, columns))
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.cbm"
            metadata_path = Path(directory) / "metrics.json"
            model.save_model(str(model_path))
            metadata_path.write_text(json.dumps({"features": columns}), encoding="utf-8")
            actual = predict(model_path, metadata_path, frame)
        np.testing.assert_allclose(actual[[f"p_{name}" for name in CLASS_NAMES]].to_numpy(), expected)


if __name__ == "__main__":
    unittest.main()
