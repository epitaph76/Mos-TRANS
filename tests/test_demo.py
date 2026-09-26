"""Check that the teaching scenario exercises the real models causally."""

import asyncio
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd
import httpx

from mos_trans.backend_api import Replay, next_observed_stop
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
            replay = Replay(archive, features, full_route=True)
            before_receipt = replay.position(VEHICLE, DAY + timedelta(minutes=15))
            after_receipt = replay.position(VEHICLE, DAY + timedelta(minutes=15, seconds=2))
            self.assertLess(before_receipt.event_time, DAY + timedelta(minutes=15))
            self.assertEqual(after_receipt.event_time, DAY + timedelta(minutes=15))
            self.assertEqual(len(replay.stops[VEHICLE]), 16)
            stalled_packet = replay.position(VEHICLE, DAY + timedelta(minutes=20))
            previous, upcoming = next_observed_stop(replay.stops[VEHICLE], stalled_packet)
            self.assertEqual(previous["id"], "DEMO-05")
            self.assertEqual(upcoming["id"], "DEMO-06")
            no_point = asyncio.run(replay.snapshot(9 * 3600 + 4 * 60, "", None))["vehicles"][0]
            self.assertEqual(no_point["forecastAvailability"], "no_point")
            self.assertEqual(no_point["nearestForecastPointAt"], (DAY + timedelta(minutes=5)).isoformat())

            async def unavailable_snapshot():
                transport = httpx.MockTransport(lambda _: httpx.Response(503))
                async with httpx.AsyncClient(transport=transport) as client:
                    return await replay.snapshot(9 * 3600 + 5 * 60, "http://ml", client)

            unavailable = asyncio.run(unavailable_snapshot())["vehicles"][0]
            self.assertEqual(unavailable["forecastAvailability"], "ml_unavailable")


if __name__ == "__main__":
    unittest.main()
