"""Build a compact historical dispatcher demo from the supplied nested ZIP."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from map_matching import Fix, clean_fixes, match_trace, simplify_route
from route_network import build_network


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Предиктор задержек транспорта.zip"
OUTPUT = ROOT / "src" / "data" / "transport.json"
TIMELINE_OUTPUT = ROOT / "src" / "data" / "timeline.json"
TIMES = ["04:00", "07:40", "11:40", "14:40", "18:20", "22:20"]
DAY = "2026-01-06"


def timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def records(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with archive.open(name) as data:
        return list(csv.DictReader(io.TextIOWrapper(data, encoding="utf-8-sig", newline="")))


def geom(value: str) -> tuple[float, float] | None:
    match = re.fullmatch(r"POINT \(([-\d.]+) ([-\d.]+)\)", value)
    return (float(match[1]), float(match[2])) if match else None


def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read(outer.infolist()[0]))) as archive:
            traffic = records(archive, "validate/traffic.csv")
            schedule = records(archive, "validate/schedule_plan.csv")
            points = records(archive, "validate/points.csv")

    traces: dict[str, list[Fix]] = defaultdict(list)
    for row in traffic:
        if row["location_valid"] != "True" or not row["lon"] or not row["lat"]:
            continue
        traces[row["tr_id"]].append(Fix(
            lon=float(row["lon"]), lat=float(row["lat"]), time_s=timestamp(row["event_time"]),
            speed_kmh=float(row["speed"] or 0),
            heading=float(row["heading"]) if row["heading"] else None,
        ))
    traces = {key: clean_fixes(value) for key, value in traces.items()}

    day_start = timestamp(f"{DAY} 00:00:00")

    stops: dict[str, list[dict]] = defaultdict(list)
    for row in schedule:
        coordinates = geom(row["geom"])
        if coordinates:
            stops[row["tr_id"]].append({
                "id": row["tt_action_item_id"], "time": row["time_begin"],
                "lon": coordinates[0], "lat": coordinates[1],
                "name": row["building_address"].strip() or "Остановка",
            })
    for value in stops.values():
        value.sort(key=lambda stop: stop["time"])

    forecasts: dict[str, list[dict]] = defaultdict(list)
    for row in points:
        forecasts[row["tr_id"]].append(row)
    for value in forecasts.values():
        value.sort(key=lambda point: point["T"])

    network = build_network(traces)
    print(f"Shared network: {len(network['nodes'])} nodes, {len(network['edges'])} edges")
    timeline = {
        "date": DAY,
        "network": network,
        "vehicles": [{
            "id": vehicle_id,
            "track": [[round(fix.time_s - day_start, 1), round(fix.lat, 6),
                       round(fix.lon, 6), round(fix.speed_kmh),
                       round(fix.heading) if fix.heading is not None else None]
                      for fix in track],
            "stops": stops.get(vehicle_id, []),
            "forecasts": [{
                "time": round(timestamp(point["T"]) - day_start),
                "estimateSeconds": float(point["cur_dev_s"]),
                "forecastTime": point["target_time_begin"],
                "forecastStopId": point["target_stop_id"],
                "sampleId": point["sample_id"],
            } for point in forecasts.get(vehicle_id, [])],
        } for vehicle_id, track in sorted(traces.items()) if track],
    }
    timeline["start"] = min(vehicle["track"][0][0] for vehicle in timeline["vehicles"])
    timeline["end"] = max(vehicle["track"][-1][0] for vehicle in timeline["vehicles"])

    snapshots = []
    for label in TIMES:
        instant = timestamp(f"{DAY} {label}:00")
        vehicles = []
        for vehicle_id, track in traces.items():
            prior = [fix for fix in track if fix.time_s <= instant]
            if not prior or instant - prior[-1].time_s > 25 * 60:
                continue
            latest = prior[-1]
            section = [fix for fix in track if abs(fix.time_s - instant) <= 35 * 60]
            reference = [fix for fix in section if abs(fix.time_s - latest.time_s) > 60]
            route = simplify_route(reference if len(reference) >= 2 else section, 27)
            if len(route) > 180:
                route = route[:: max(1, len(route) // 180)]
            recent = [fix for fix in track if latest.time_s - 5 * 60 <= fix.time_s <= latest.time_s]
            matched = match_trace(recent, route)
            near_stops = [stop for stop in stops.get(vehicle_id, []) if abs(timestamp(stop["time"]) - instant) <= 45 * 60]
            upcoming = [stop for stop in stops.get(vehicle_id, []) if timestamp(stop["time"]) >= instant][:3]
            estimate = next((point for point in forecasts.get(vehicle_id, []) if timestamp(point["T"]) == instant), None)
            vehicles.append({
                "id": vehicle_id,
                "position": [round(matched.lat if matched and not matched.off_route else latest.lat, 6),
                             round(matched.lon if matched and not matched.off_route else latest.lon, 6)],
                "rawPosition": [round(latest.lat, 6), round(latest.lon, 6)],
                "speed": round(latest.speed_kmh),
                "heading": latest.heading,
                "gpsAgeMin": round((instant - latest.time_s) / 60, 1),
                "matchDistanceM": matched.distance_m if matched else None,
                "matchConfidence": matched.confidence if matched else 0,
                "offRoute": matched.off_route if matched else True,
                "route": [[round(lat, 6), round(lon, 6)] for lon, lat in route],
                "stops": near_stops[:10],
                "nextStop": upcoming[0] if upcoming else None,
                "estimateSeconds": float(estimate["cur_dev_s"]) if estimate else None,
                "forecastTime": estimate["target_time_begin"] if estimate else None,
                "forecastStopId": estimate["target_stop_id"] if estimate else None,
                "sampleId": estimate["sample_id"] if estimate else None,
            })
        vehicles.sort(key=lambda item: (item["estimateSeconds"] is None, -abs(item["estimateSeconds"] or 0), item["id"]))
        snapshots.append({"time": label, "vehicles": vehicles})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"date": DAY, "snapshots": snapshots}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    TIMELINE_OUTPUT.write_text(json.dumps(timeline, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPUT}: {OUTPUT.stat().st_size:,} bytes")
    print(f"Wrote {TIMELINE_OUTPUT}: {TIMELINE_OUTPUT.stat().st_size:,} bytes")
    for item in snapshots:
        print(item["time"], len(item["vehicles"]), "vehicles", sum(v["estimateSeconds"] is not None for v in item["vehicles"]), "estimates")


if __name__ == "__main__":
    main()
