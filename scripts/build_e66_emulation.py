"""Build the small, reproducible е66 NDTP demo fixture from the supplied archive."""

import argparse
import csv
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from map_matching import Fix, clean_fixes, simplify_route
from build_road_routes import request_legs


VEHICLE = "c35ec13ea8bfa789"
STOP_ROWS = {
    "A": {"5367": 1, "7172": 2, "21256": 3, "21236": 4, "3432": 5,
          "8629": 6, "18073": 7, "3982": 8, "8633": 9, "8440": 10,
          "4680": 11, "5369": 12, "5371": 14},
    "B": {"5371": 1, "5369": 3, "4703": 4, "4704": 5, "4707": 6,
          "3893": 7, "3897": 8, "14131": 9, "6405": 10, "14015": 11,
          "3833": 12, "5375": 13, "3592": 14},
}
RUNS = [("B1", "B", "18:21"), ("A1", "A", "20:08"),
        ("B2", "B", "20:55"), ("A2", "A", "21:38"),
        ("B3", "B", "22:16"), ("A3", "A", "23:34")]
MSK = timezone(timedelta(hours=3))


def minute(value: str) -> int:
    return int(value[:2]) * 60 + int(value[3:5])


def build(archive_path: Path, schedule_path: Path, output: Path) -> None:
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(archive_path) as archive:
        catalog = {str(stop["stop_id"]): stop for stop in json.loads(archive.read("bootstrap/stops_catalog.json"))}
        with archive.open("data/traffic.csv") as source:
            points = [{"time": row["gps_time"] or row["event_time"],
                       "lat": float(row["lat"]), "lon": float(row["lon"]),
                       "speed": float(row["speed"]) if row["speed"] else None,
                       "heading": float(row["heading"]) if row["heading"] else None}
                      for row in csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig", newline=""))
                      if row["tr_id"] == VEHICLE and row["route_number"].casefold() == "е66"
                      and row["location_valid"].lower() == "true" and row["lat"] and row["lon"]]
    points.sort(key=lambda row: row["time"])
    trips = []
    base = datetime(2026, 9, 26, tzinfo=MSK)
    for run_id, direction, origin in RUNS:
        timetable = schedule["directions"][direction]["stops"]
        split = [([minute(t) for t in row["times"] if minute(t) >= 300],
                  [minute(t) for t in row["times"] if minute(t) < 300]) for row in timetable]
        trip_index = split[0][0].index(minute(origin))
        by_row = {number: stop_id for stop_id, number in STOP_ROWS[direction].items()}
        stops = []
        for number in sorted(by_row):
            day, early = split[number - 1]
            # The B timetable crosses midnight after its third stop.
            index = trip_index + (1 if direction == "B" and number >= 4 else 0)
            planned_minute = day[index] if index < len(day) else early[index - len(day)] + 1440
            stop_id = by_row[number]
            catalog_stop = catalog[stop_id]
            stops.append({"id": stop_id, "name": catalog_stop["stop_name"],
                          "lat": catalog_stop["lat"], "lon": catalog_stop["lon"],
                          "time": (base + timedelta(minutes=planned_minute)).isoformat()})
        trips.append({"id": run_id, "direction": direction, "stops": stops})
    paths = {}
    for direction, run_id in (("A", "A1"), ("B", "B2")):
        trip = next(item for item in trips if item["id"] == run_id)
        start = datetime.fromisoformat(trip["stops"][0]["time"]).timestamp() - 120
        end = datetime.fromisoformat(trip["stops"][-1]["time"]).timestamp() + 120
        fixes = [Fix(lon=point["lon"], lat=point["lat"],
                     time_s=datetime.fromisoformat(point["time"]).timestamp(),
                     speed_kmh=point["speed"] or 0, heading=point["heading"])
                 for point in points if start <= datetime.fromisoformat(point["time"]).timestamp() <= end]
        path = simplify_route(clean_fixes(fixes), spacing_m=12)
        paths[direction] = [[round(lat, 7), round(lon, 7)] for lon, lat in path]
    road_segments = []
    road_directions = {}
    for direction in ("A", "B"):
        stops = next(item["stops"] for item in trips if item["direction"] == direction)
        road_directions[direction] = []
        for leg in request_legs(stops):
            road_directions[direction].append(len(road_segments))
            road_segments.append(leg)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"route": "е66", "vehicle": VEHICLE,
                                  "source": schedule["source"],
                                  "schedule_retrieved_at": schedule["retrieved_at_utc"],
                                  "schedule_note": "Current Saturday timetable; historical revision unverified",
                                  "trips": trips, "paths": paths,
                                  "roadRoutes": {"segments": road_segments, "directions": road_directions},
                                  "points": points}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(points)} points, {len(trips)} trips, {len(road_segments)} road legs -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("mos_trans/replay_data/e66.json"))
    args = parser.parse_args()
    build(args.archive, args.schedule, args.output)
