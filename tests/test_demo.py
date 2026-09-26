"""Check that the teaching scenario exercises the real models causally."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd

from mos_trans.backend_api import Replay
from mos_trans.demo import DAY, VEHICLE, generate
from mos_trans.inference import FORBIDDEN, Predictor


class DemoTests(unittest.TestCase):
    def test_schedule_fault_and_model_response(self):
        with tempfile.TemporaryDirectory() as folder:
            archive, features = generate(Path(folder))
            frame = pd.read_parquet(features)
            early = frame.loc[frame["T"] == DAY + timedelta(minutes=5)].iloc[0]
            stalled = frame.loc[frame["T"] == DAY + timedelta(minutes=20)].iloc[0]
            self.assertEqual(early.cur_dev_s, 0)
            self.assertGreaterEqual(stalled.cur_dev_s, 300)
            self.assertLessEqual(stalled.realtime_speed_mean_5m, 2)
            predictor = Predictor()
            forecasts = predictor.predict(frame.drop(columns=list(FORBIDDEN), errors="ignore"))
            self.assertGreater(forecasts.iloc[15].probability_delay_over_120s,
                               forecasts.iloc[0].probability_delay_over_120s)
            self.assertGreater(forecasts.iloc[15].predicted_delay_s,
                               forecasts.iloc[0].predicted_delay_s)
            replay = Replay(archive, features)
            before_receipt = replay.position(VEHICLE, DAY + timedelta(minutes=15))
            after_receipt = replay.position(VEHICLE, DAY + timedelta(minutes=15, seconds=2))
            self.assertLess(before_receipt.event_time, DAY + timedelta(minutes=15))
            self.assertEqual(after_receipt.event_time, DAY + timedelta(minutes=15))


if __name__ == "__main__":
    unittest.main()
