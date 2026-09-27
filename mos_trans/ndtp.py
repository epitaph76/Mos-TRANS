"""NDTP TCP receiver and live route forecasting."""

from __future__ import annotations

import asyncio
import os
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from mos_trans.live_route import LiveRoute
from mos_trans.live_features import live_feature_frame
from mos_trans.inference import json_records


NPL = struct.Struct("<HHHHBIH")
NPH = struct.Struct("<HHHI")
NAV = struct.Struct("<IIIBBHHHHHBB")


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc


def encode_frame(unit_id: int, service: int, kind: int, request_id: int, body: bytes) -> bytes:
    """Encode the unencrypted NPL/NPH subset used by recorded telemetry replay."""
    payload = NPH.pack(service, kind, 1, request_id) + body
    crc = int.from_bytes(crc16_modbus(payload).to_bytes(2, "big"), "little")
    return NPL.pack(0x7E7E, len(payload), 0, crc, 2, unit_id, 0) + payload


@dataclass(frozen=True)
class Navigation:
    unit_id: int
    event_time: datetime
    received_at: datetime
    lon: float | None
    lat: float | None
    speed: int
    heading: int


def parse_frame(header: bytes, payload: bytes, received_at: datetime) -> Navigation | None:
    if len(header) != NPL.size:
        raise ValueError("Invalid NPL header size")
    signature, size, _flags, _crc, kind, unit_id, _request_id = NPL.unpack(header)
    if signature != 0x7E7E or kind != 2 or size != len(payload) or size < NPH.size:
        raise ValueError("Invalid NDTP frame header")
    if header[6:8] != crc16_modbus(payload).to_bytes(2, "big"):
        raise ValueError("Invalid NDTP CRC")
    service, message_type, _nph_flags, _nph_request = NPH.unpack_from(payload)
    if service == 0 and message_type == 100:
        return None
    if service != 1 or message_type != 101 or len(payload) < NPH.size + 2 + NAV.size:
        raise ValueError("Unsupported NDTP message")
    if payload[NPH.size] != 0:
        raise ValueError("Realtime packet must start with G6CellNav00")
    timestamp, lon_raw, lat_raw, flags, _battery, speed, _speed_max, course, _track, _alt, _nsat, _pdop = NAV.unpack_from(payload, NPH.size + 2)
    valid = bool(flags & 0x80)
    lon = lon_raw / 1e7 * (1 if flags & 0x40 else -1) if valid else None
    lat = lat_raw / 1e7 * (1 if flags & 0x20 else -1) if valid else None
    if lon is not None and not (-180 <= lon <= 180 and -90 <= lat <= 90):
        lon = lat = None
    return Navigation(unit_id, datetime.fromtimestamp(timestamp, timezone.utc), received_at,
                      lon, lat, speed, course)


class LiveStore:
    def __init__(self) -> None:
        self.latest: dict[int, Navigation] = {}
        self.connections = 0
        self.frames = 0
        self.errors = 0
        self.routes: dict[int, LiveRoute] = {}
        self.metadata: dict[int, dict] = {}
        self.prediction_log: dict[int, list[dict]] = {}
        self.nav_frames = 0
        self.ml_client = None
        self.ml_url: str | None = None
        self.forecast_tasks: dict[int, asyncio.Task] = {}
        self.last_forecast_at: dict[int, datetime] = {}
        self.forecast_errors = 0

    def configure(self, unit_id: int, plan: dict) -> None:
        if not plan.get("route"):
            raise ValueError("Route name is required")
        trips = plan.get("trips", [])
        route = LiveRoute(plan, unit_id) if trips else None
        task = self.forecast_tasks.pop(unit_id, None)
        if task is not None:
            task.cancel()
        if route is None:
            self.routes.pop(unit_id, None)
        else:
            self.routes[unit_id] = route
        self.metadata[unit_id] = {"route": plan["route"], "tripId": plan.get("tripId"),
                                  "sourceVehicleId": plan.get("sourceVehicleId")}
        self.prediction_log[unit_id] = []
        self.latest.pop(unit_id, None)
        self.last_forecast_at.pop(unit_id, None)

    async def score(self, route: LiveRoute, packet: Navigation, features,
                    target_id: str, trip_id: str) -> None:
        try:
            response = await self.ml_client.post(f"{self.ml_url}/predict",
                json={"features": json_records(features), "explain_all": True}, timeout=5.0)
            response.raise_for_status()
            result = response.json()["predictions"][0]
            if self.routes.get(packet.unit_id) is not route:
                return
            target = next(stop for trip in route.trips if trip["id"] == trip_id
                          for stop in trip["stops"] if stop["id"] == target_id)
            predicted_arrival = target["instant"] + timedelta(seconds=result["predicted_delay_s"])
            forecast = {"sampleId": result["sample_id"],
                "estimateSeconds": result["predicted_delay_s"],
                "predictedArrivalAt": predicted_arrival.isoformat(),
                "probability": result["probability_delay_over_120s"],
                "delayExplanation": result.get("delay_explanation"),
                "forecastMethod": "ml", "forecastGeneratedAt": packet.event_time.isoformat(),
                "forecastStopId": target_id, "tripId": trip_id}
            route.prediction = forecast
            self.prediction_log.setdefault(packet.unit_id, []).append({
                "unitId": packet.unit_id, "tripId": trip_id, "targetStopId": target_id,
                "at": packet.event_time.isoformat(), "predictedDelaySeconds": result["predicted_delay_s"],
                "probability": result["probability_delay_over_120s"],
            })
            if (route.last_result and route.last_result.get("forecastStopId") == target_id
                    and route.last_result.get("tripId") == trip_id):
                route.last_result.update(forecast)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.forecast_errors += 1
            if self.routes.get(packet.unit_id) is route and route.last_result:
                route.last_result["forecastAvailability"] = "ml_unavailable"

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            while True:
                header = await reader.readexactly(NPL.size)
                size = NPL.unpack(header)[1]
                if size < NPH.size or size > 65535:
                    raise ValueError("Invalid NDTP payload size")
                payload = await reader.readexactly(size)
                packet = parse_frame(header, payload, datetime.now(timezone.utc))
                self.frames += 1
                if packet is not None:
                    self.nav_frames += 1
                    old = self.latest.get(packet.unit_id)
                    if old is None or packet.event_time >= old.event_time:
                        self.latest[packet.unit_id] = packet
                        route = self.routes.get(packet.unit_id)
                        if route is not None:
                            route.process(packet)
                            if (self.ml_client is not None and self.ml_url
                                    and route.last_result.get("forecastAvailability") == "ready"):
                                task = self.forecast_tasks.get(packet.unit_id)
                                last = self.last_forecast_at.get(packet.unit_id)
                                if ((task is None or task.done()) and
                                        (last is None or (packet.event_time - last).total_seconds() >= 30)):
                                    try:
                                        features = live_feature_frame(route, packet)
                                        if features is not None:
                                            target_id = route.last_result["forecastStopId"]
                                            trip_id = route.last_result["tripId"]
                                            self.last_forecast_at[packet.unit_id] = packet.event_time
                                            self.forecast_tasks[packet.unit_id] = asyncio.create_task(
                                                self.score(route, packet, features, target_id, trip_id))
                                    except (ValueError, KeyError, TypeError):
                                        self.forecast_errors += 1
                                        route.last_result["forecastAvailability"] = "ml_unavailable"
        except asyncio.IncompleteReadError:
            pass
        except (ValueError, OverflowError):
            self.errors += 1
        finally:
            self.connections -= 1
            writer.close()
            await writer.wait_closed()

    def snapshot(self) -> dict:
        now = datetime.now(timezone.utc)
        visible_age_s = float(os.getenv("NDTP_VISIBLE_MAX_AGE_S", "120"))
        vehicles = []
        for packet in self.latest.values():
            if (now - packet.received_at).total_seconds() > visible_age_s:
                continue
            route = self.routes.get(packet.unit_id)
            derived = route.last_result if route is not None else None
            vehicle = {
                "id": str(packet.unit_id), "position": [packet.lat, packet.lon] if packet.lat is not None else None,
                "speed": packet.speed, "heading": packet.heading,
                "gpsAgeMin": max(0, (now - packet.received_at).total_seconds()) / 60,
                "stale": (now - packet.received_at).total_seconds() > 120,
                "receivedAt": packet.received_at.isoformat(),
                "gpsEventTime": packet.event_time.isoformat(),
                "estimateSeconds": None, "probability": None, "sampleId": None,
                "reason": "Нет привязки к расписанию", "recommendation": "Подключить плановое расписание для устройства",
                "source": "NDTP",
            }
            vehicle.update(self.metadata.get(packet.unit_id, {}))
            if route is None and packet.unit_id in self.metadata:
                vehicle.update({"forecastAvailability": "no_schedule",
                    "reason": "Нет планового расписания для рейса",
                    "recommendation": "Доступно только положение по GPS"})
            if route is not None and route.trips:
                vehicle["stops"] = [
                    {key: value for key, value in stop.items() if key != "instant"}
                    for stop in route.trips[0]["stops"]]
            if derived:
                vehicle.update(derived)
            vehicles.append(vehicle)
        return {
            "source": "ndtp-emulator",
            "connections": self.connections, "frames": self.frames,
            "navFrames": self.nav_frames, "errors": self.errors,
            "forecastErrors": self.forecast_errors,
            "vehicles": vehicles,
        }

    def report(self) -> dict:
        predictions = [row for rows in self.prediction_log.values() for row in rows]
        return {"navFrames": self.nav_frames, "ndtpErrors": self.errors,
                "forecastErrors": self.forecast_errors,
                "pendingForecasts": sum(not task.done() for task in self.forecast_tasks.values()),
                "predictions": predictions}
