"""Minimal NDTP TCP receiver for the supplied emulator's handshake and Nav00."""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from datetime import datetime, timezone


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
        return {
            "source": "ndtp-emulator",
            "connections": self.connections, "frames": self.frames, "errors": self.errors,
            "vehicles": [{
                "id": str(packet.unit_id), "position": [packet.lat, packet.lon] if packet.lat is not None else None,
                "speed": packet.speed, "heading": packet.heading,
                "gpsAgeMin": max(0, (now - packet.event_time).total_seconds()) / 60,
                "stale": (now - packet.received_at).total_seconds() > 120,
                "receivedAt": packet.received_at.isoformat(),
                "estimateSeconds": None, "probability": None, "sampleId": None,
                "reason": "Нет привязки к расписанию", "recommendation": "Подключить плановое расписание для устройства",
                "source": "NDTP-эмулятор",
            } for packet in self.latest.values()],
        }
