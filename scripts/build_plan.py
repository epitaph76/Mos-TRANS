"""Export only the public planned schedule for the browser map."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from mos_trans.preprocessing.core import DataSource, _point


def build(dataset: Path, output: Path) -> None:
    stops = defaultdict(list)
    forecast_times = []
    source = DataSource(dataset)
    try:
        with source.csv_rows("validate/schedule_plan.csv") as rows:
            for row in rows:
                lon, lat = _point(row["geom"])
                stops[str(row["tr_id"])].append({
                    "id": str(row["tt_action_item_id"]), "time": row["time_begin"],
                    "lon": lon, "lat": lat,
                    "name": row["building_address"].strip() or "Остановка",
                })
        with source.csv_rows("validate/points.csv") as rows:
            forecast_times = [datetime.fromisoformat(row["T"]) for row in rows]
    finally:
        source.close()
    nodes, edges = [], []
    for vehicle, sequence in sorted(stops.items()):
        sequence.sort(key=lambda stop: stop["time"])
        prior = None
        for stop in sequence:
            index = len(nodes)
            nodes.append([stop["lat"], stop["lon"]])
            if prior is not None and nodes[prior] != nodes[index]:
                edges.append([prior, index, [vehicle]])
            prior = index
    day = min(datetime.fromisoformat(s["time"]).date() for group in stops.values() for s in group)
    midnight = datetime.combine(day, datetime.min.time())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "date": day.isoformat(),
        "start": int((min(forecast_times) - midnight).total_seconds()),
        "end": int((max(forecast_times) - midnight).total_seconds()) + 59,
        "vehicles": [{"id": vehicle, "stops": group} for vehicle, group in sorted(stops.items())],
        "network": {"mergeMeters": 0, "nodes": nodes, "edges": edges},
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--output", type=Path, default=Path("src/data/plan.json"))
    args = parser.parse_args()
    build(args.dataset, args.output)
