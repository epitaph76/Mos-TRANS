"""Bake road geometry for the historical replay from its public stop coordinates.

Run after build_plan.py. The browser reads the resulting file locally; it never
needs to call a routing server while playing the recording.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROUTER = "https://router.project-osrm.org/route/v1/driving/"
MAX_WAYPOINTS = 35


def key(first: dict, second: dict) -> tuple[float, float, float, float]:
    return (round(first["lon"], 6), round(first["lat"], 6),
            round(second["lon"], 6), round(second["lat"], 6))


def request_legs(stops: list[dict]) -> list[list[list[float]]]:
    coordinates = ";".join(f'{stop["lon"]:.6f},{stop["lat"]:.6f}' for stop in stops)
    url = ROUTER + coordinates + "?" + urlencode({
        "steps": "true", "overview": "false", "geometries": "geojson",
        "continue_straight": "true",
    })
    request = Request(url, headers={"User-Agent": "mos-trans-road-route-builder/1.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=35) as response:
                payload = json.load(response)
            if payload.get("code") != "Ok":
                raise RuntimeError(f'OSRM returned {payload.get("code")}')
            legs = payload["routes"][0]["legs"]
            if len(legs) != len(stops) - 1:
                raise RuntimeError("OSRM returned an unexpected leg count")
            result = []
            for leg in legs:
                points = []
                for step in leg["steps"]:
                    for lon, lat in step["geometry"]["coordinates"]:
                        point = [round(lat, 6), round(lon, 6)]
                        if not points or points[-1] != point:
                            points.append(point)
                result.append(points)
            return result
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            if attempt == 2:
                raise RuntimeError(f"Routing request failed: {error}") from error
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def broken_trip(first: dict, second: dict) -> bool:
    return (datetime.fromisoformat(second["time"]) -
            datetime.fromisoformat(first["time"])).total_seconds() > 1800


def build(plan_path: Path, output_path: Path) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cache: dict[tuple[float, float, float, float], list[list[float]]] = {}
    for vehicle in plan["vehicles"]:
        stops = vehicle["stops"]
        index = 0
        while index < len(stops) - 1:
            pair = key(stops[index], stops[index + 1])
            if pair in cache or broken_trip(stops[index], stops[index + 1]):
                index += 1
                continue
            end = index + 1
            while (end < len(stops) - 1 and end - index < MAX_WAYPOINTS - 1
                   and key(stops[end], stops[end + 1]) not in cache
                   and not broken_trip(stops[end], stops[end + 1])):
                end += 1
            try:
                legs = request_legs(stops[index:end + 1])
            except RuntimeError as error:
                if end > index + 1:
                    end = index + 1
                    legs = request_legs(stops[index:end + 1])
                else:
                    raise RuntimeError(f'{vehicle["id"]} stop {index}: {error}') from error
            for offset, points in enumerate(legs):
                cache[key(stops[index + offset], stops[index + offset + 1])] = points
            print(f'{vehicle["id"]}: routed stops {index}-{end}', flush=True)
            index = end
            time.sleep(0.25)

    segments = []
    ids = {}
    vehicles = {}
    for vehicle in plan["vehicles"]:
        sequence = []
        for first, second in zip(vehicle["stops"], vehicle["stops"][1:]):
            pair = key(first, second)
            if broken_trip(first, second) or pair not in cache or len(cache[pair]) < 2:
                sequence.append(-1)
                continue
            if pair not in ids:
                ids[pair] = len(segments)
                segments.append(cache[pair])
            sequence.append(ids[pair])
        vehicles[vehicle["id"]] = sequence
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"segments": segments, "vehicles": vehicles},
                                      ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(segments)} road segments to {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("src/data/plan.json"))
    parser.add_argument("--output", type=Path, default=Path("src/data/road_routes.json"))
    args = parser.parse_args()
    build(args.plan, args.output)
