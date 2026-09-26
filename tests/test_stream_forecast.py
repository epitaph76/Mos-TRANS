"""Stream requests use only received packets and planned schedule fields."""

from __future__ import annotations

import asyncio
import csv
import io
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from mos_trans.backend_api import Replay
from mos_trans.stream_forecast import RouteMatcher, build_stream_features


DAY = datetime(2026, 1, 6)


def csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def at(seconds: int) -> str:
    return (DAY + timedelta(seconds=seconds)).isoformat(sep=" ")


def fixture(path: Path) -> None:
    stops = [{"tr_id": "7", "tt_action_item_id": f"stop-{i}",
              "time_begin": at(i * 720), "geom": f"POINT ({37 + i * .001} 55)",
              "building_address": f"Stop {i}"} for i in range(3)]
    packets = [
        {"packet_id": "first", "tr_id": "7", "unit_id": "70", "event_time": at(0),
         "receive_time": at(10), "location_valid": "true", "lon": 37,
         "lat": 55, "speed": 0, "heading": 90, "is_hist_data": "false"},
        {"packet_id": "second", "tr_id": "7", "unit_id": "70", "event_time": at(60),
         "receive_time": at(65), "location_valid": "true", "lon": 37.00008,
         "lat": 55, "speed": 0, "heading": 90, "is_hist_data": "false"},
        {"packet_id": "invalid", "tr_id": "7", "unit_id": "70", "event_time": at(120),
         "receive_time": at(125), "location_valid": "false", "lon": 37.0002,
         "lat": 55, "speed": 12, "heading": 90, "is_hist_data": "false"},
        {"packet_id": "far", "tr_id": "7", "unit_id": "70", "event_time": at(180),
         "receive_time": at(185), "location_valid": "true", "lon": 38,
         "lat": 56, "speed": 12, "heading": 90, "is_hist_data": "false"},
        {"packet_id": "last", "tr_id": "7", "unit_id": "70", "event_time": at(1440),
         "receive_time": at(1441), "location_valid": "true", "lon": 37.002,
         "lat": 55, "speed": 0, "heading": 90, "is_hist_data": "false"},
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("validate/schedule_plan.csv", csv_bytes(stops, list(stops[0])))
        archive.writestr("validate/traffic.csv", csv_bytes(packets, list(packets[0])))


class RouteMatcherTests(unittest.TestCase):
    def test_repeated_corridor_advances_to_second_pass(self):
        stops = pd.DataFrame({"stop_lon": [37, 37.001, 37, 37.001],
                              "stop_lat": [55] * 4,
                              "time_begin": pd.to_datetime([at(n) for n in (0, 600, 1200, 1800)])})
        matcher = RouteMatcher(stops)
        first, _ = matcher.match(37.0005, 55, DAY + timedelta(seconds=300))
        self.assertEqual(matcher.previous_index, 0)
        second, _ = matcher.match(37.0005, 55, DAY + timedelta(seconds=1500))
        self.assertEqual(matcher.previous_index, 2)
        self.assertAlmostEqual(first, 0, delta=2)
        self.assertAlmostEqual(second, 0, delta=2)


class StreamReplayTests(unittest.TestCase):
    def test_every_eligible_fix_and_replay_availability(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "sample.zip"
            output = Path(folder) / "stream"
            fixture(dataset)
            summary = build_stream_features(dataset, output)
            status = pd.read_parquet(output / "stream_status.parquet")
            features = pd.read_parquet(output / "stream_features.parquet")
            self.assertEqual(summary["forecast_requests"], 2)
            self.assertEqual(status.availability.tolist(),
                             ["ready", "ready", "bad_gps", "off_route", "no_target"])
            self.assertEqual(features.packet_id.tolist(), ["first", "second"])
            self.assertEqual(features.target_stop_id.tolist(), ["stop-1", "stop-1"])
            self.assertEqual(features.realtime_packet_count_2m.tolist(), [1, 2])
            self.assertEqual(status.stopped_duration_s.tolist()[:2], [0.0, 60.0])
            self.assertEqual(status.segment_speed_mean_kmh.tolist()[:2], [0.0, 0.0])
            replay = Replay(dataset, output / "stream_features.parquet",
                            status_path=output / "stream_status.parquet")
            self.assertTrue(replay.active_rows(DAY + timedelta(seconds=9)).empty)

            def response(request: httpx.Request) -> httpx.Response:
                rows = __import__("json").loads(request.content)["features"]
                return httpx.Response(200, json={"predictions": [
                    {"sample_id": row["sample_id"], "predicted_delay_s": 42.0,
                     "probability_delay_over_120s": 0.2} for row in rows]})

            async def snapshots() -> tuple[dict, dict, dict, dict]:
                async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
                    first = await replay.snapshot(10, "http://ml", client)
                    later = await replay.snapshot(65, "http://ml", client)
                    bad = await replay.snapshot(125, "http://ml", client)
                    repeat = await replay.snapshot(10, "http://ml", client)
                    return first, later, bad, repeat

            first, later, bad, repeat = asyncio.run(snapshots())
            self.assertEqual(first["vehicles"][0]["estimateSeconds"], 42.0)
            self.assertEqual(first["vehicles"][0]["forecastPacketId"], "first")
            self.assertEqual(later["vehicles"][0]["forecastPacketId"], "second")
            self.assertEqual(bad["vehicles"][0]["forecastAvailability"], "bad_gps")
            self.assertEqual(first["vehicles"], repeat["vehicles"])

    def test_ml_failure_is_distinct_from_missing_point(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "sample.zip"
            output = Path(folder) / "stream"
            fixture(dataset)
            build_stream_features(dataset, output)
            replay = Replay(dataset, output / "stream_features.parquet",
                            status_path=output / "stream_status.parquet")

            async def snapshot() -> dict:
                async with httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda _: httpx.Response(503))) as client:
                    return await replay.snapshot(10, "http://ml", client)

            vehicle = asyncio.run(snapshot())["vehicles"][0]
            self.assertEqual(vehicle["forecastAvailability"], "ml_unavailable")
            self.assertIsNone(vehicle["probability"])


if __name__ == "__main__":
    unittest.main()
