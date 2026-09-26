"""Contract checks for the joined artifacts and causal transport intake."""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mos_trans.backend_api import Replay, explanation
from mos_trans.inference import Predictor, prediction_frame
from mos_trans.ndtp import NAV, NPH, NPL, LiveStore, crc16_modbus, parse_frame


ROOT = Path(__file__).resolve().parents[1]


def frame(payload: bytes, unit_id: int = 42) -> tuple[bytes, bytes]:
    crc = int.from_bytes(crc16_modbus(payload).to_bytes(2, "big"), "little")
    return NPL.pack(0x7E7E, len(payload), 0, crc, 2, unit_id, 0), payload


class NdtpTests(unittest.IsolatedAsyncioTestCase):
    def test_navigation_frame_and_crc(self):
        nav = NAV.pack(1767673800, 376173210, 557551234, 0xE0, 0, 25, 30, 90, 0, 150, 8, 2)
        header, payload = frame(NPH.pack(1, 101, 1, 2) + bytes([0, 0]) + nav)
        result = parse_frame(header, payload, datetime.now(timezone.utc))
        self.assertEqual(result.unit_id, 42)
        self.assertAlmostEqual(result.lon, 37.617321)
        self.assertAlmostEqual(result.lat, 55.7551234)
        self.assertEqual(result.speed, 25)
        with self.assertRaisesRegex(ValueError, "CRC"):
            parse_frame(header, payload[:-1] + b"\x00", datetime.now(timezone.utc))

    async def test_tcp_receiver_accepts_handshake_then_navigation(self):
        store = LiveStore()
        server = await asyncio.start_server(store.handle, "127.0.0.1", 0)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
            for payload in (
                NPH.pack(0, 100, 1, 1) + bytes(18),
                NPH.pack(1, 101, 1, 2) + bytes([0, 0]) + NAV.pack(
                    int(datetime.now(timezone.utc).timestamp()), 376173210, 557551234,
                    0xE0, 0, 25, 30, 90, 0, 150, 8, 2),
            ):
                header, body = frame(payload)
                writer.write(header + body)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            for _ in range(20):
                if store.frames == 2:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(store.frames, 2)
            self.assertEqual(store.snapshot()["vehicles"][0]["estimateSeconds"], None)
        finally:
            server.close()
            await server.wait_closed()


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor(
            ROOT / "artifacts/route_delay_trend_submission/catboost_residual_final.cbm",
            ROOT / "artifacts/calibrated_probability/model.joblib",
        )

    def test_horizon_and_future_columns(self):
        path = ROOT / "data/processed_route/validate_features.parquet"
        if not path.exists():
            self.skipTest("Locally prepared dataset is unavailable")
        frame = prediction_frame(path, self.predictor).head(1)
        at = frame.iloc[0]["T"].to_pydatetime()
        for seconds in (600, 901):
            altered = frame.copy()
            altered["target_time_begin"] = at + timedelta(seconds=seconds)
            with self.assertRaisesRegex(ValueError, "Target stop"):
                self.predictor.predict(altered)
        leaked = frame.copy()
        leaked["time_fact_begin"] = at
        with self.assertRaisesRegex(ValueError, "Future"):
            self.predictor.predict(leaked)

    def test_reproduces_both_released_validate_artifacts(self):
        path = ROOT / "data/processed_route/validate_features.parquet"
        if not path.exists():
            self.skipTest("Locally prepared dataset is unavailable")
        result = self.predictor.predict(prediction_frame(path, self.predictor)).sort_values("sample_id")
        reg = pd.read_csv(ROOT / "artifacts/route_delay_trend_submission/validate_predictions_diagnostics.csv").sort_values("sample_id")
        prob = pd.read_csv(ROOT / "artifacts/calibrated_probability/validate_probabilities.csv").sort_values("sample_id")
        self.assertEqual(len(result), 151)
        self.assertEqual(result.sample_id.tolist(), reg.sample_id.tolist())
        self.assertEqual(result.sample_id.tolist(), prob.sample_id.tolist())
        self.assertLess(np.max(np.abs(result.predicted_delay_s - reg.prediction)), 1e-8)
        self.assertLess(np.max(np.abs(result.probability_delay_over_120s - prob.probability_delay_over_120s)), 1e-8)


class ReplayTests(unittest.TestCase):
    def test_no_unreceived_future_position(self):
        replay = Replay.__new__(Replay)
        t0 = datetime(2026, 1, 6, 12)
        old = type("Packet", (), {"event_time": t0, "lat": 55.0, "lon": 37.0})()
        future = type("Packet", (), {"event_time": t0.replace(minute=1), "lat": 56.0, "lon": 38.0})()
        replay.gps = {"bus": ([t0, t0.replace(minute=2)], [old, future])}
        self.assertIs(replay.position("bus", t0.replace(minute=1)), old)
        self.assertIs(replay.position("bus", t0.replace(minute=2)), future)

    def test_reason_is_labeled_as_observed_pattern(self):
        reason, action = explanation(pd.Series({
            "realtime_stopped_fraction_5m": 0.3,
            "realtime_speed_mean_5m": 11,
            "context_vehicle_count_500m": 0,
            "context_speed_mean_500m": np.nan,
            "cur_dev_s": 278,
        }))
        self.assertEqual(reason, "Накопленное отставание от графика")
        self.assertIn("интервала", action)


if __name__ == "__main__":
    unittest.main()
