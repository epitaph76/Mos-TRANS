import asyncio
import unittest
from datetime import datetime, timedelta, timezone

import httpx

from mos_trans.ndtp import LiveStore, NPL, encode_frame, parse_frame
from scripts.replay_server_dataset import compare_labels, navigation, select_start_time


UTC = timezone.utc
START = datetime(2026, 9, 26, 18, 0, tzinfo=UTC)
PLAN = {"route": "test", "trips": [{"id": "T1", "direction": "A", "stops": [
    {"id": "a", "name": "A", "lat": 55.0, "lon": 37.0, "time": START.isoformat()},
    {"id": "b", "name": "B", "lat": 55.0, "lon": 37.01,
     "time": (START + timedelta(minutes=10)).isoformat()},
    {"id": "c", "name": "C", "lat": 55.0, "lon": 37.02,
     "time": (START + timedelta(minutes=20)).isoformat()},
]}]}


def _navigation(row, request_id):
    return navigation(16653764, request_id, {
        "event_time": row["time"], "lat": str(row["lat"]), "lon": str(row["lon"]),
        "speed": str(row["speed"]), "heading": str(row["heading"]),
    })


class LiveEmulationTests(unittest.IsolatedAsyncioTestCase):
    async def test_gps_only_trip_and_separate_connections(self):
        store = LiveStore()
        store.configure(10, {"route": "577", "tripId": "gps-only", "trips": []})
        store.configure(11, PLAN)
        server = await asyncio.start_server(store.handle, "127.0.0.1", 0)
        try:
            port = server.sockets[0].getsockname()[1]
            for unit in (10, 11):
                _, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(encode_frame(unit, 0, 100, 1,
                    __import__("struct").pack("<HHHIII", 6, 2, 0, unit, 65535, 0)))
                writer.write(navigation(unit, 2, {"event_time": (START + timedelta(minutes=6)).isoformat(),
                    "lat": "55.0", "lon": "37.004", "speed": "20", "heading": "90"}))
                await writer.drain()
                writer.close()
                await writer.wait_closed()
            for _ in range(100):
                if store.nav_frames == 2 and store.connections == 0:
                    break
                await asyncio.sleep(0.01)
            vehicles = {vehicle["id"]: vehicle for vehicle in store.snapshot()["vehicles"]}
            self.assertEqual(store.nav_frames, 2)
            self.assertEqual(vehicles["10"]["route"], "577")
            self.assertEqual(vehicles["10"]["forecastAvailability"], "no_schedule")
            self.assertIsNone(vehicles["10"]["estimateSeconds"])
            self.assertEqual(vehicles["11"]["tripId"], "T1")
            self.assertGreaterEqual(len(vehicles["11"]["stops"]), 2)
        finally:
            server.close()
            await server.wait_closed()

    async def test_reconnect_and_crc_rejection(self):
        store = LiveStore()
        server = await asyncio.start_server(store.handle, "127.0.0.1", 0)
        try:
            port = server.sockets[0].getsockname()[1]
            for second in (0, 15):
                row = {"event_time": (START + timedelta(seconds=second)).isoformat(),
                       "lat": "55", "lon": "37", "speed": "10", "heading": "90"}
                frame = navigation(12, second + 2, row)
                if second == 0:
                    with self.assertRaisesRegex(ValueError, "CRC"):
                        parse_frame(frame[:NPL.size], frame[NPL.size:-1] + b"\xff", START)
                _, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(frame)
                await writer.drain()
                writer.close()
                await writer.wait_closed()
            for _ in range(100):
                if store.nav_frames == 2 and store.connections == 0:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(store.nav_frames, 2)
            self.assertEqual(store.connections, 0)
        finally:
            server.close()
            await server.wait_closed()

    async def test_ml_unavailable_keeps_gps_without_fake_prediction(self):
        store = LiveStore()
        store.configure(16653764, PLAN)
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda _: httpx.Response(503))) as client:
            store.ml_client, store.ml_url = client, "http://ml"
            server = await asyncio.start_server(store.handle, "127.0.0.1", 0)
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
                writer.write(_navigation({"time": (START + timedelta(minutes=6)).isoformat(),
                    "lat": 55.0, "lon": 37.004, "speed": 20, "heading": 90}, 2))
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                for _ in range(100):
                    if store.forecast_errors:
                        break
                    await asyncio.sleep(0.01)
                vehicle = store.snapshot()["vehicles"][0]
                self.assertEqual(vehicle["forecastAvailability"], "ml_unavailable")
                self.assertNotEqual(vehicle["forecastMethod"], "ml")
                self.assertIsNone(vehicle["probability"])
            finally:
                server.close()
                await server.wait_closed()

    async def test_ndtp_stream_calls_ml_and_exposes_prediction(self):
        requests = []

        def respond(request):
            payload = __import__("json").loads(request.content)
            self.assertIs(payload["explain_all"], True)
            features = payload["features"][0]
            requests.append(features)
            return httpx.Response(200, json={"predictions": [{
                "sample_id": features["sample_id"], "predicted_delay_s": 142.0,
                "probability_delay_over_120s": 0.72, "delay_explanation": None}]})

        store = LiveStore()
        store.configure(16653764, PLAN)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            store.ml_client, store.ml_url = client, "http://ml"
            server = await asyncio.start_server(store.handle, "127.0.0.1", 0)
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
                row = {"time": (START + timedelta(minutes=6)).isoformat(),
                       "lat": 55.0, "lon": 37.004, "speed": 20, "heading": 90}
                writer.write(_navigation(row, 2))
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                for _ in range(100):
                    if store.routes[16653764].prediction is not None:
                        break
                    await asyncio.sleep(0.01)
                vehicle = store.snapshot()["vehicles"][0]
                self.assertEqual(vehicle["forecastMethod"], "ml")
                self.assertEqual(vehicle["estimateSeconds"], 142.0)
                self.assertEqual(vehicle["probability"], 0.72)
                self.assertEqual(datetime.fromisoformat(vehicle["predictedArrivalAt"]),
                                 START + timedelta(minutes=20, seconds=142))
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0]["target_stop_id"], "c")
                self.assertNotIn("target_delay_s", requests[0])
            finally:
                server.close()
                await server.wait_closed()

    async def test_ndtp_receiver_calculates_from_configured_plan(self):
        store = LiveStore()
        store.configure(16653764, PLAN)
        server = await asyncio.start_server(store.handle, "127.0.0.1", 0)
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
            for index, lon in enumerate([37.004, 37.005]):
                packet = _navigation({"time": (START + timedelta(minutes=6, seconds=index * 15)).isoformat(),
                                      "lat": 55.0, "lon": lon, "speed": 20, "heading": 90}, index + 2)
                self.assertEqual(NPL.unpack(packet[:NPL.size])[1], len(packet) - NPL.size)
                writer.write(packet)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            for _ in range(30):
                if store.frames == 2:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(store.errors, 0)
            vehicle = store.snapshot()["vehicles"][0]
            self.assertEqual(vehicle["route"], "test")
            self.assertEqual(vehicle["tripId"], "T1")
            self.assertEqual(vehicle["nextStop"]["id"], "b")
            self.assertAlmostEqual(vehicle["currentDeviationSeconds"], 75, delta=4)
            self.assertLess(vehicle["gpsAgeMin"], 1)
            self.assertEqual(len(store.routes[16653764].history), 2)
        finally:
            server.close()
            await server.wait_closed()


class ReplayReportTests(unittest.TestCase):
    def test_recorded_moscow_start_time_keeps_packet_order(self):
        points = [{"available": datetime(2026, 9, 27, 6, 59, tzinfo=UTC)},
                  {"available": datetime(2026, 9, 27, 7, 0, tzinfo=UTC)},
                  {"available": datetime(2026, 9, 27, 7, 1, tzinfo=UTC)}]
        self.assertEqual(select_start_time(points, "10:00"), points[1:])
        self.assertEqual(select_start_time(points, "10:00:30"), points[2:])
        self.assertEqual(select_start_time(points, "2026-09-27T10:00:00+03:00"), points[1:])
        self.assertEqual(select_start_time(points, None), points)
        with self.assertRaisesRegex(ValueError, "valid Moscow time"):
            select_start_time(points, "25:00")
        with self.assertRaisesRegex(ValueError, "No GPS packets"):
            select_start_time(points, "12:00")

    def test_label_matching_uses_only_prior_predictions(self):
        at = START.isoformat()
        label = {"tr_id": "trip", "target_stop_id": "stop", "T": at,
                 "target_delay_s": "100", "cur_dev_s": "20"}
        later = {"tripId": "trip", "targetStopId": "stop",
                 "at": (START + timedelta(seconds=1)).isoformat(),
                 "predictedDelaySeconds": 100}
        self.assertEqual(compare_labels([label], [later])["matched"], 0)
        earlier = {**later, "at": (START - timedelta(seconds=1)).isoformat(),
                   "predictedDelaySeconds": 90}
        report = compare_labels([label], [later, earlier])
        self.assertEqual(report["matched"], 1)
        self.assertEqual(report["maeSeconds"], 10)


if __name__ == "__main__":
    unittest.main()
