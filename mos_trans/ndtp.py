"""Minimal NDTP TCP receiver for the supplied emulator's handshake and Nav00."""

from __future__ import annotations

import asyncio
import os
import struct
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mos_trans.preprocessing.core import DataSource, _point


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


@dataclass(frozen=True)
class Navigation:
    unit_id: int
    event_time: datetime
    received_at: datetime
    lon: float | None
    lat: float | None
    speed: int
    heading: int


class LiveCatalog:
    """Resolve terminal IDs and planned routes from the same validate dataset."""

    def __init__(self, dataset: Path) -> None:
        self.vehicles: dict[int, str] = {}
        self.stops: dict[str, list[dict]] = defaultdict(list)
        source = DataSource(dataset)
        try:
            with source.csv_rows("validate/traffic.csv") as traffic:
                for row in traffic:
                    unit_id, tr_id = int(row["unit_id"]), str(row["tr_id"])
                    previous = self.vehicles.setdefault(unit_id, tr_id)
                    if previous != tr_id:
                        raise ValueError(f"Ambiguous unit_id {unit_id}: {previous}, {tr_id}")
            with source.csv_rows("validate/schedule_plan.csv") as schedule:
                for row in schedule:
                    lon, lat = _point(row["geom"])
                    self.stops[str(row["tr_id"])].append({
                        "id": str(row["tt_action_item_id"]),
                        "time": row["time_begin"], "lon": lon, "lat": lat,
                        "name": row["building_address"].strip() or "Остановка",
                    })
        finally:
            source.close()
        for stops in self.stops.values():
            stops.sort(key=lambda stop: stop["time"])

    def next_stop(self, tr_id: str, packet: Navigation) -> dict | None:
        stops = self.stops.get(tr_id, [])
        if packet.lon is None or packet.lat is None or len(stops) < 2:
            return None
        event_at = packet.event_time.replace(tzinfo=None)
        if event_at.date() == datetime.fromisoformat(stops[0]["time"]).date():
            return next((stop for stop in stops
                         if datetime.fromisoformat(stop["time"]) > event_at), None)
        nearest = (float("inf"), 0)
        for index, (start, end) in enumerate(zip(stops, stops[1:])):
            dx = (end["lon"] - start["lon"]) * 0.56
            dy = end["lat"] - start["lat"]
            length = dx * dx + dy * dy
            fraction = max(0.0, min(1.0, (
                (packet.lon - start["lon"]) * 0.56 * dx
                + (packet.lat - start["lat"]) * dy
            ) / length)) if length else 0.0
            distance = ((packet.lon - start["lon"]) * 0.56 - fraction * dx) ** 2 + (
                packet.lat - start["lat"] - fraction * dy
            ) ** 2
            if distance < nearest[0]:
                nearest = (distance, index + (2 if fraction >= 0.999 else 1))
        return stops[nearest[1]] if nearest[1] < len(stops) else None


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
    def __init__(self, catalog: LiveCatalog | None = None) -> None:
        self.catalog = catalog
        self.latest: dict[int, Navigation] = {}
        self.connections = 0
        self.frames = 0
        self.errors = 0

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
                    old = self.latest.get(packet.unit_id)
                    if old is None or packet.event_time >= old.event_time:
                        self.latest[packet.unit_id] = packet
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
            age = (now - packet.received_at).total_seconds()
            if age > visible_age_s:
                continue
            tr_id = self.catalog.vehicles.get(packet.unit_id) if self.catalog else None
            stops = self.catalog.stops.get(tr_id, []) if tr_id and self.catalog else []
            vehicles.append({
                "id": tr_id or str(packet.unit_id), "unitId": packet.unit_id,
                "position": [packet.lat, packet.lon] if packet.lat is not None else None,
                "speed": packet.speed, "heading": packet.heading,
                # A replayed dataset packet has an old event timestamp but a fresh receipt.
                "gpsAgeMin": max(0, age) / 60,
                "gpsEventTime": packet.event_time.isoformat(),
                "stale": age > 120, "receivedAt": packet.received_at.isoformat(),
                "stops": stops, "nextStop": self.catalog.next_stop(tr_id, packet) if tr_id else None,
                "estimateSeconds": None, "probability": None, "sampleId": None,
                "reason": "Прогноз по живому пакету пока недоступен" if tr_id else "Устройство отсутствует в датасете",
                "recommendation": "Проверить поступление следующих пакетов" if tr_id else "Проверить unit_id устройства",
                "source": "NDTP · маршрут из датасета" if tr_id else "NDTP · неизвестное устройство",
            })
        return {
            "source": "ndtp-emulator",
            "connections": self.connections, "frames": self.frames, "errors": self.errors,
            "vehicles": vehicles,
        }
