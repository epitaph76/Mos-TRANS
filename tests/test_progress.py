import unittest

import numpy as np
import pandas as pd

from mos_trans.modeling.progress import Route, VehiclePackets, _xy, augment


class ProgressTests(unittest.TestCase):
    def setUp(self):
        points = np.asarray([_xy(0, 0, 0), _xy(.01, 0, 0)])
        self.routes = {"1": Route(["a", "b"], np.asarray([pd.Timestamp("2026-01-01 09:50").value,
                                                          pd.Timestamp("2026-01-01 10:15").value]),
                                  points, np.asarray([0, points[1, 0]]), {"a": 0, "b": 1}, 0)}
        self.point = pd.DataFrame({"sample_id": ["s"], "tr_id": ["1"],
                                   "T": [pd.Timestamp("2026-01-01 10:00")], "target_stop_id": ["b"],
                                   "schedule_target_lon": [.01], "schedule_target_lat": [0.]})

    def test_late_and_future_packets_do_not_change_progress(self):
        traffic = pd.DataFrame({
            "packet_id": ["old", "current", "late", "future", "no-receipt"],
            "tr_id": ["1"] * 5,
            "event_time": pd.to_datetime(["2026-01-01 09:57:55", "2026-01-01 09:59:55",
                                          "2026-01-01 09:59:59", "2026-01-01 10:00:01",
                                          "2026-01-01 09:59:58"]),
            "available_time": pd.to_datetime(["2026-01-01 09:57:55", "2026-01-01 09:59:55",
                                              "2026-01-01 10:01:00", "2026-01-01 10:00:01", None]),
            "gps_valid": [True] * 5,
            "lon": [.001, .005, .009, .009, .009], "lat": [0.] * 5,
            "speed": [10.] * 5,
        })
        result = augment(self.point, traffic, self.routes).iloc[0]
        self.assertAlmostEqual(result.progress_route_remaining_m, .005 * 111_195, delta=2)
        self.assertAlmostEqual(result.progress_route_2m_m, .004 * 111_195, delta=2)
        self.assertEqual(result.progress_gps_updates_2m, 1)

    def test_no_gps_and_forbidden_schedule_fact(self):
        traffic = pd.DataFrame({"packet_id": ["p"], "tr_id": ["1"],
                                "event_time": [pd.Timestamp("2026-01-01 09:59")],
                                "available_time": [pd.Timestamp("2026-01-01 09:59")],
                                "gps_valid": [False], "lon": [np.nan], "lat": [np.nan],
                                "speed": [0.]})
        result = augment(self.point, traffic, self.routes).iloc[0]
        self.assertTrue(np.isnan(result.progress_route_remaining_m))
        self.assertEqual(result.progress_gps_updates_2m, 0)
        with self.assertRaises(ValueError):
            augment(self.point.assign(time_fact_begin="2026-01-01 10:15"), traffic, self.routes)


if __name__ == "__main__":
    unittest.main()
