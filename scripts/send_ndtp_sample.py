"""Send real validate/traffic.csv fixes in the supplied emulator's NDTP format."""

from __future__ import annotations

import argparse
import csv
import io
import socket
import struct
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


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


def frame(unit_id: int, body: bytes) -> bytes:
    payload_crc = int.from_bytes(crc16_modbus(body).to_bytes(2, "big"), "little")
    return NPL.pack(0x7E7E, len(body), 0, payload_crc, 2, unit_id, 0) + body


def rows(archive: zipfile.ZipFile, member: str):
    with io.TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def sample_packets(dataset: Path, vehicle: str, count: int) -> list[dict]:
    with zipfile.ZipFile(dataset) as archive:
        point = next((datetime.fromisoformat(row["T"]) for row in rows(archive, "validate/points.csv")
                      if row["tr_id"] == vehicle), None)
        if point is None:
            raise ValueError(f"No validate point for tr_id {vehicle}")
        candidates = [row for row in rows(archive, "validate/traffic.csv")
                      if row["tr_id"] == vehicle and row["location_valid"] == "True"
                      and row["lon"] and row["lat"] and row["speed"] and row["heading"]
                      and datetime.fromisoformat(row["event_time"]) <= point]
    candidates.sort(key=lambda row: row["event_time"])
    return candidates[-count:]


def send(host: str, port: int, packets: list[dict], interval: float) -> None:
    if not packets:
        raise ValueError("No suitable GPS packets found")
    unit_id = int(packets[0]["unit_id"])
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.sendall(frame(unit_id, NPH.pack(0, 100, 1, 1)
                                 + struct.pack("<HHHIII", 6, 2, 0, unit_id, 65535, 0)))
        time.sleep(0.2)
        for index, row in enumerate(packets, 2):
            lon, lat = float(row["lon"]), float(row["lat"])
            timestamp = int(datetime.fromisoformat(row["event_time"]).replace(tzinfo=timezone.utc).timestamp())
            flags = 0x80 | (0x40 if lon >= 0 else 0) | (0x20 if lat >= 0 else 0)
            speed = round(float(row["speed"]))
            heading = round(float(row["heading"]))
            nav = NAV.pack(timestamp, round(abs(lon) * 1e7), round(abs(lat) * 1e7),
                           flags, 0, speed, speed, heading, 0,
                           round(float(row["alt"] or 0)), 0, 0)
            connection.sendall(frame(unit_id, NPH.pack(1, 101, 1, index) + b"\x00\x00" + nav))
            print(f"sent packet_id={row['packet_id']} tr_id={row['tr_id']} unit_id={unit_id} "
                  f"event_time={row['event_time']} lon={lon} lat={lat} speed={speed}")
            time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--host", default="xn--80axeckfde.xn--e1agiq1a.xn--p1ai")
    parser.add_argument("--port", type=int, default=9201)
    parser.add_argument("--vehicle", default="131672")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    send(args.host, args.port, sample_packets(args.dataset, args.vehicle, args.count), args.interval)
