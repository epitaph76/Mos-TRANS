"""Deterministic one-vehicle teaching run through the production feature pipeline."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from mos_trans.features.delay_context import add_delay_trend_features
from mos_trans.features.route_sections import (
    add_route_signatures, attach_planned_sections, build_route_segments, load_schedule,
)
from mos_trans.preprocessing.core import (
    DataSource, TrafficIndex, _features, _sample, _schedule, clean_traffic_row,
)

DAY = datetime(2026, 1, 6, 9)
VEHICLE = "130389"
STOP_NAMES = (
    "Улица Академика Янгеля", "Россошанская улица", "Чертановская улица",
    "Кировоградская улица", "Пражская", "Красного Маяка",
    "Чертаново Центральное", "Чертаново Северное", "Каховская улица",
    "Севастопольский проспект", "Нахимовский проспект", "Нагорная улица",
    "Варшавское шоссе", "Даниловский рынок", "Тульская", "Серпуховская",
)


def _at(seconds: float) -> str:
    return (DAY + timedelta(seconds=seconds)).isoformat(sep=" ", timespec="seconds")


def _coordinates(index: float) -> tuple[float, float]:
    return 37.597 + 0.0023 * index, 55.612 + 0.0061 * index


def _progress(seconds: float) -> tuple[float, float, float]:
    """Return travelled stop index, observed speed and current schedule deviation."""
    if seconds <= 900:
        progress = seconds / 180
        speed = 15.0 if seconds % 180 >= 30 else 0.0
    elif seconds < 1320:
        progress, speed = 5.0, 0.0
    else:
        progress = min(15.0, 5 + (seconds - 1320) / 240)
        speed = 8.0 if seconds % 240 >= 35 else 0.0
    return progress, speed, max(0.0, seconds - progress * 180)


def _csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def generate(output_dir: Path, plan_path: Path | None = None) -> tuple[Path, Path]:
    """Write CSV archive and calculate features with the shared preprocessing code."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "scenario.zip"
    feature_path = output_dir / "features.parquet"

    schedule = []
    for index, name in enumerate(STOP_NAMES):
        lon, lat = _coordinates(index)
        schedule.append({
            "tr_id": VEHICLE, "tt_action_item_id": f"DEMO-{index:02d}",
            "time_begin": _at(index * 180), "geom": f"POINT ({lon:.6f} {lat:.6f})",
            "building_address": name,
        })

    traffic = []
    for seconds in range(0, 3001, 20):
        progress, speed, _ = _progress(seconds)
        lower = min(14, int(progress))
        fraction = min(1.0, progress - lower)
        lon_a, lat_a = _coordinates(lower)
        lon_b, lat_b = _coordinates(lower + 1)
        traffic.append({
            "packet_id": f"demo-{seconds:04d}", "tr_id": VEHICLE,
            "unit_id": "demo-unit-01", "event_time": _at(seconds),
            "receive_time": _at(seconds + 2), "location_valid": "true",
            "lon": round(lon_a + (lon_b - lon_a) * fraction, 6),
            "lat": round(lat_a + (lat_b - lat_a) * fraction, 6),
            "speed": speed, "heading": 14, "is_hist_data": "false",
        })

    points = []
    for seconds in range(300, 2161, 60):
        _, _, deviation = _progress(seconds)
        target_index = next((i for i in range(1, len(schedule))
                             if 600 < i * 180 - seconds <= 900), None)
        if target_index is None:
            continue
        points.append({
            "sample_id": f"demo_{seconds:04d}",
            "tr_id": VEHICLE, "T": _at(seconds),
            "target_stop_id": schedule[target_index]["tt_action_item_id"],
            "target_time_begin": schedule[target_index]["time_begin"],
            "cur_dev_s": round(deviation, 1),
        })

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("validate/schedule_plan.csv", _csv_bytes(schedule,
                         ["tr_id", "tt_action_item_id", "time_begin", "geom", "building_address"]))
        archive.writestr("validate/traffic.csv", _csv_bytes(traffic,
                         ["packet_id", "tr_id", "unit_id", "event_time", "receive_time",
                          "location_valid", "lon", "lat", "speed", "heading", "is_hist_data"]))
        archive.writestr("validate/points.csv", _csv_bytes(points,
                         ["sample_id", "tr_id", "T", "target_stop_id", "target_time_begin", "cur_dev_s"]))

    source = DataSource(archive_path)
    try:
        stops, by_vehicle = _schedule(source, "validate/schedule_plan.csv")
        with source.csv_rows("validate/traffic.csv") as rows:
            index = TrafficIndex([clean_traffic_row(row) for row in rows])
        with source.csv_rows("validate/points.csv") as rows:
            samples = [_sample(row, stops) for row in rows]
        features = pd.DataFrame([
            _features(sample, stops[sample["target_stop_id"]], by_vehicle[sample["tr_id"]], index)
            for sample in samples
        ])
    finally:
        source.close()
    route = add_route_signatures(load_schedule(archive_path, "validate"))
    features["T"] = features["T"].astype("datetime64[us]")
    features["target_time_begin"] = features["target_time_begin"].astype("datetime64[us]")
    features = attach_planned_sections(features, build_route_segments(route))
    features = add_delay_trend_features(features)
    features.to_parquet(feature_path, index=False)

    if plan_path:
        nodes = [[_coordinates(i)[1], _coordinates(i)[0]] for i in range(len(schedule))]
        plan = {
            "date": DAY.date().isoformat(), "start": 9 * 3600 + 300,
            "end": 9 * 3600 + 3000,
            "vehicles": [{"id": VEHICLE, "stops": [
                {"id": row["tt_action_item_id"], "time": row["time_begin"],
                 "lon": _coordinates(i)[0], "lat": _coordinates(i)[1],
                 "name": row["building_address"]} for i, row in enumerate(schedule)
            ]}],
            "network": {"mergeMeters": 0, "nodes": nodes,
                        "edges": [[i, i + 1, [VEHICLE]] for i in range(len(nodes) - 1)]},
        }
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return archive_path, feature_path


if __name__ == "__main__":
    generate(Path("data/demo"), Path("src/data/demo_plan.json"))
