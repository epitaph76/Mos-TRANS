import unittest

import numpy as np
import pandas as pd

from mos_trans.modeling.history_risk import augment


class HistoryRiskTests(unittest.TestCase):
    def test_only_prior_same_vehicle_observations_are_used(self):
        points = pd.DataFrame({"tr_id": ["1", "1", "2", "1"],
                               "T": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:05",
                                                    "2026-01-01 10:03", "2026-01-01 10:10"]),
                               "cur_dev_s": [20, 35, 999, 50]})
        result = augment(points)
        self.assertTrue(np.isnan(result.loc[0, "previous_dev_s"]))
        self.assertEqual(result.loc[1, "previous_dev_s"], 20)
        self.assertEqual(result.loc[1, "dev_change_s"], 15)
        self.assertTrue(np.isnan(result.loc[2, "previous_dev_s"]))
        self.assertEqual(result.loc[3, "dev_change_10m_s"], 30)
        changed_future = augment(points.assign(cur_dev_s=[20, 35, 999, 1000]))
        self.assertEqual(changed_future.loc[1, "dev_change_s"], 15)

    def test_factual_time_is_rejected(self):
        points = pd.DataFrame({"tr_id": ["1"], "T": [pd.Timestamp("2026-01-01 10:00")],
                               "cur_dev_s": [0], "time_fact_begin": [pd.Timestamp("2026-01-01 10:05")]})
        with self.assertRaises(ValueError):
            augment(points)


if __name__ == "__main__":
    unittest.main()
