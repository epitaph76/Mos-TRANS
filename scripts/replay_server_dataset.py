"""Replay accepted server_dataset trips into the application's NDTP TCP port.

Only GPS and planned schedules are sent to the application. GPS-derived labels
are read after playback solely to evaluate forecasts already made by the ML API.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import math
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PREFIX = "server_dataset/repo_adapter/"
POINT_RE = re.compile(r"^POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)$")


def _point(value: str) -> tuple[float, float]:
    match = POINT_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Invalid schedule geometry: {value!r}")
    return float(match[1]), float(match[2])


def encode_frame(unit_id: int, service: int, kind: int, request_id: int, body: bytes) -> bytes:
    """Build the little-endian NPL/NPH frame with a Modbus CRC."""
    payload = struct.pack("<HHHI", service, kind, 1, request_id) + body
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    wire_crc = int.from_bytes(crc.to_bytes(2, "big"), "little")
    return struct.pack("<HHHHBIH", 0x7E7E, len(payload), 0, wire_crc, 2, unit_id, 0) + payload


def _rows(archive: zipfile.ZipFile, member: str, *, adapter: bool = True) -> list[dict]:
    name = (PREFIX if adapter else "server_dataset/") + member
    with archive.open(name) as binary:
        return list(csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")))


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _unit_ids(trip_ids: list[str]) -> dict[str, int]:
    taken: set[int] = set()
    result = {}
    for trip_id in sorted(trip_ids):
        candidate = (zlib.crc32(trip_id.encode("utf-8")) & 0x7FFFFFFF) or 1
        while candidate in taken:
            candidate = candidate % 0x7FFFFFFF + 1
        result[trip_id] = candidate
        taken.add(candidate)
    return result


def load_replay(archive_path: Path, tr_id: str | None = None) -> tuple[list[dict], list[dict]]:
    """Read only the source traffic and planned schedules needed for replay."""
    with zipfile.ZipFile(archive_path) as archive:
        assignments = _rows(archive, "trip_assignments.csv", adapter=False)
        accepted = {row["trip_instance_id"]: row for row in assignments
                    if row["accepted"].strip().lower() == "true"}
        if tr_id is not None:
            if tr_id not in accepted:
                raise ValueError(f"Trip {tr_id} is not an accepted trip in server_dataset.zip")
            accepted = {tr_id: accepted[tr_id]}
        traffic = [row for row in _rows(archive, "train/traffic.csv")
                   if row["tr_id"] in accepted and row["location_valid"].lower() == "true"
                   and row["lon"] and row["lat"]]
        schedule: dict[str, list[dict]] = defaultdict(list)
        for split, filename in (("train", "schedule.csv"), ("test", "schedule.csv"),
                                ("validate", "schedule_plan.csv")):
            for row in _rows(archive, f"{split}/{filename}"):
                if row["tr_id"] in accepted:
                    schedule[row["tr_id"]].append(row)

    by_trip: dict[str, list[dict]] = defaultdict(list)
    for row in traffic:
        by_trip[row["tr_id"]].append(row)
    unit_ids = _unit_ids(list(accepted))
    configs: list[dict] = []
    packets: list[dict] = []
    for trip_id in sorted(accepted):
        fixes = by_trip[trip_id]
        if not fixes:
            raise ValueError(f"Accepted trip {trip_id} has no valid GPS packets")
        assignment = accepted[trip_id]
        unit_id = unit_ids[trip_id]
        stops = []
        for row in sorted(schedule[trip_id], key=lambda item: item["time_begin"]):
            lon, lat = _point(row["geom"])
            stops.append({"id": row["tt_action_item_id"],
                          "name": row["building_address"] or "Остановка",
                          "lon": lon, "lat": lat,
                          "time": _instant(row["time_begin"]).isoformat()})
        configs.append({
            "unitId": unit_id, "route": fixes[0]["route_number"],
            "tripId": trip_id, "sourceVehicleId": assignment["source_vehicle_id"],
            "trips": [{"id": trip_id, "direction": assignment["direction"],
                       "stops": stops}] if len(stops) >= 2 else [],
        })
        for row in fixes:
            event = _instant(row["event_time"])
            received = _instant(row["receive_time"]) if row["receive_time"] else event
            packets.append({"unitId": unit_id, "tripId": trip_id, "row": row,
                            "available": max(event, received)})
    packets.sort(key=lambda item: (item["available"], item["unitId"], item["row"]["packet_id"]))
    return configs, packets


MOSCOW = timezone(timedelta(hours=3))


def select_start_time(packets: list[dict], start_at: str | None) -> list[dict]:
    """Seek by recorded availability time; clock-only input is Moscow time."""
    if start_at is None:
        return packets
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", start_at):
        hour, minute, *seconds = map(int, start_at.split(":"))
        second = seconds[0] if seconds else 0
        if hour > 23 or minute > 59 or second > 59:
            raise ValueError("--start-at must be a valid Moscow time, e.g. 10:00")
        day = packets[0]["available"].astimezone(MOSCOW).date()
        requested = datetime(day.year, day.month, day.day, hour, minute, second,
                             tzinfo=MOSCOW)
    else:
        try:
            requested = datetime.fromisoformat(start_at)
        except ValueError as exc:
            raise ValueError("--start-at must be HH:MM, HH:MM:SS or an ISO timestamp") from exc
        if requested.tzinfo is None:
            requested = requested.replace(tzinfo=MOSCOW)
    selected = [packet for packet in packets if packet["available"] >= requested]
    if not selected:
        last = packets[-1]["available"].astimezone(MOSCOW).isoformat()
        raise ValueError(f"No GPS packets at or after --start-at; last packet is {last}")
    return selected


def load_labels(archive_path: Path, trip_ids: set[str]) -> list[dict]:
    """Open GPS-derived target labels only after replay has completed."""
    with zipfile.ZipFile(archive_path) as archive:
        return [row for row in _rows(archive, "samples.csv", adapter=False)
                if row["tr_id"] in trip_ids]


def navigation(unit_id: int, request_id: int, row: dict) -> bytes:
    lat, lon = float(row["lat"]), float(row["lon"])
    speed_value = float(row["speed"]) if row["speed"] else 0.0
    heading_value = float(row["heading"]) if row["heading"] else 0.0
    speed = max(0, min(65535, round(speed_value))) if math.isfinite(speed_value) else 0
    heading = max(0, min(360, round(heading_value))) if math.isfinite(heading_value) else 0
    flags = 0x80 | (0x20 if lat >= 0 else 0) | (0x40 if lon >= 0 else 0)
    # NDTP Nav00 has second resolution. Floor the available source event time.
    timestamp = int(_instant(row["event_time"]).timestamp())
    nav = struct.pack("<IIIBBHHHHHBB", timestamp, round(abs(lon) * 10_000_000),
                      round(abs(lat) * 10_000_000), flags, 0, speed, speed,
                      heading, 0, 0, 0, 0)
    return encode_frame(unit_id, 1, 101, request_id, b"\x00\x00" + nav)


def _json_request(url: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url}: HTTP {exc.code}: {detail}") from exc


async def _connect(host: str, port: int, unit_id: int) -> asyncio.StreamWriter:
    _, writer = await asyncio.open_connection(host, port)
    writer.write(encode_frame(unit_id, 0, 100, 1,
                             struct.pack("<HHHIII", 6, 2, 0, unit_id, 65535, 0)))
    await writer.drain()
    return writer


async def _send(host: str, port: int, writers: dict[int, asyncio.StreamWriter],
                unit_id: int, payload: bytes) -> bool:
    for attempt in range(3):
        try:
            writer = writers.get(unit_id)
            if writer is None or writer.is_closing():
                writer = await _connect(host, port, unit_id)
                writers[unit_id] = writer
            writer.write(payload)
            await writer.drain()
            return True
        except (OSError, ConnectionError):
            old = writers.pop(unit_id, None)
            if old is not None:
                old.close()
            await asyncio.sleep(0.2 * (attempt + 1))
    return False


def compare_labels(labels: list[dict], predictions: list[dict]) -> dict:
    by_target: dict[tuple[str, str], list[tuple[datetime, float]]] = defaultdict(list)
    for row in predictions:
        by_target[(row["tripId"], row["targetStopId"])].append(
            (_instant(row["at"]), float(row["predictedDelaySeconds"])))
    for rows in by_target.values():
        rows.sort(key=lambda item: item[0])
    errors, baseline = [], []
    for label in labels:
        if not label["target_delay_s"]:
            continue
        at = _instant(label["T"])
        candidates = by_target.get((label["tr_id"], label["target_stop_id"]), [])
        prior = next((prediction for prediction in reversed(candidates) if prediction[0] <= at), None)
        if prior is None:
            continue
        actual = float(label["target_delay_s"])
        errors.append(abs(prior[1] - actual))
        baseline.append(abs(float(label["cur_dev_s"]) - actual))
    return {"labels": len(labels), "matched": len(errors),
            "coverage": len(errors) / len(labels) if labels else None,
            "maeSeconds": sum(errors) / len(errors) if errors else None,
            "currentDeviationBaselineMaeSeconds": sum(baseline) / len(baseline) if baseline else None}


async def replay(archive_path: Path, tr_id: str | None, api: str, host: str, port: int,
                 speed: float, max_packets: int = 0, report_path: Path | None = None,
                 start_at: str | None = None) -> dict:
    if speed <= 0 or max_packets < 0:
        raise ValueError("Speed must be positive and max-packets nonnegative")
    configs, packets = load_replay(archive_path, tr_id)
    total_points = len(packets)
    packets = select_start_time(packets, start_at)
    selected_units = {packet["unitId"] for packet in packets}
    configs = [config for config in configs if config["unitId"] in selected_units]
    if start_at is not None:
        first_at = packets[0]["available"].astimezone(MOSCOW).isoformat()
        print(f"NDTP: --start-at {start_at}; first GPS {first_at}; "
              f"{len(packets)}/{total_points} packets remain", file=sys.stderr)
    if max_packets:
        packets = packets[:max_packets]
        selected = {packet["unitId"] for packet in packets}
        configs = [config for config in configs if config["unitId"] in selected]
    api = api.rstrip("/")
    baseline = await asyncio.to_thread(_json_request, f"{api}/api/live/report")
    for config in configs:
        await asyncio.to_thread(_json_request, f"{api}/api/live/configure", config)
    writers: dict[int, asyncio.StreamWriter] = {}
    request_ids = defaultdict(lambda: 2)
    sent, failed = 0, []
    first = packets[0]["available"].timestamp()
    try:
        # A single scheduler sends across all sockets in source availability order.
        start = time.monotonic()
        for packet in packets:
            due = start + (packet["available"].timestamp() - first) / speed
            await asyncio.sleep(max(0, due - time.monotonic()))
            unit_id = packet["unitId"]
            payload = navigation(unit_id, request_ids[unit_id], packet["row"])
            request_ids[unit_id] = request_ids[unit_id] % 0xFFFFFFFF + 1
            if await _send(host, port, writers, unit_id, payload):
                sent += 1
            else:
                failed.append(packet["row"]["packet_id"])
            if sent and sent % 500 == 0:
                print(f"NDTP: {sent}/{len(packets)} packets", file=sys.stderr)
    finally:
        for writer in writers.values():
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for writer in writers.values()),
                             return_exceptions=True)
    # Let the TCP receiver and pending ML calls finish before taking the report.
    deadline = time.monotonic() + 30
    while True:
        live = await asyncio.to_thread(_json_request, f"{api}/api/live/report")
        accepted = live["navFrames"] - baseline["navFrames"]
        if (accepted >= sent and live["pendingForecasts"] == 0) or time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.2)
    selected_ids = {config["tripId"] for config in configs}
    predictions = [row for row in live["predictions"] if row["tripId"] in selected_ids]
    first_event_times: dict[str, datetime] = {}
    for packet in packets:
        first_event_times.setdefault(packet["tripId"], _instant(packet["row"]["event_time"]))
    labels = [row for row in load_labels(archive_path, selected_ids)
              if _instant(row["T"]) >= first_event_times[row["tr_id"]]]
    evaluation = compare_labels(labels, predictions)
    summary = {"trips": len(configs), "plannedTrips": sum(bool(c["trips"]) for c in configs),
               "gpsOnlyTrips": sum(not c["trips"] for c in configs), "packets": len(packets),
               "sent": sent, "received": accepted, "failedPacketIds": failed,
               "ndtpErrors": live["ndtpErrors"] - baseline["ndtpErrors"],
               "forecastErrors": live["forecastErrors"] - baseline["forecastErrors"],
               "pendingForecasts": live["pendingForecasts"],
               "predictions": len(predictions), "evaluation": evaluation}
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({**summary, "predictionLog": predictions},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--tr-id", help="Replay one accepted trip; default is all 88")
    parser.add_argument("--start-at",
                        help="Start at recorded Moscow time HH:MM[:SS] or ISO timestamp")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9201)
    parser.add_argument("--speed", type=float, default=60)
    parser.add_argument("--max-packets", type=int, default=0, help="Global smoke-test cap")
    parser.add_argument("--report", type=Path, help="Write full JSON report and prediction log")
    args = parser.parse_args()
    try:
        result = asyncio.run(replay(args.archive, args.tr_id, args.api, args.host, args.port,
                                    args.speed, args.max_packets, args.report, args.start_at))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["sent"] != result["received"] or result["failedPacketIds"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
