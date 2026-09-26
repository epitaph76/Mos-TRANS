"""Causal forecast requests made from the historical telemetry stream.

The published validate points are deliberately absent from this pipeline.  Each
received GPS fix can become a request if a planned stop is 10--15 minutes away.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from mos_trans.features.delay_context import add_delay_trend_features
from mos_trans.features.route_sections import (
    add_route_signatures, attach_planned_sections, build_route_segments, load_schedule,
)
from mos_trans.preprocessing.core import (
    DataSource, GPS_MAX_AGE_S, TrafficIndex, _features, _schedule,
    clean_traffic_row,
)


STREAM_VERSION = 4
MAX_MATCH_DISTANCE_M = 150.0
MAX_DEVIATION_S = 600.0
EARTH_METERS_PER_DEGREE = 111_320.0


@dataclass
class RouteMatcher:
    """Match fixes to a planned stop sequence, including repeated corridors."""

    stops: pd.DataFrame

    def __post_init__(self) -> None:
        ordered = self.stops.sort_values("time_begin", kind="stable")
        latitude = float(ordered.stop_lat.mean())
        scale = np.array([EARTH_METERS_PER_DEGREE * math.cos(math.radians(latitude)),
                          EARTH_METERS_PER_DEGREE])
        points = ordered[["stop_lon", "stop_lat"]].to_numpy(float) * scale
        self.start = points[:-1]
        self.vector = np.diff(points, axis=0)
        self.length2 = np.einsum("ij,ij->i", self.vector, self.vector)
        self.length = np.sqrt(self.length2)
        self.progress = np.r_[0.0, np.cumsum(self.length)]
        self.times = ordered.time_begin.to_numpy(dtype="datetime64[s]").astype("int64")
        self.scale = scale
        self.previous_progress: float | None = None
        self.previous_index: int | None = None
        self.previous_time: datetime | None = None

    def match(self, lon: float, lat: float, event_time: datetime,
              speed: float | None = None) -> tuple[float | None, float]:
        if not len(self.start):
            return None, math.inf
        point = np.array([lon, lat]) * self.scale
        fraction = np.clip(np.einsum("ij,ij->i", point - self.start, self.vector)
                           / np.maximum(self.length2, 1), 0, 1)
        snapped = self.start + fraction[:, None] * self.vector
        distance = np.linalg.norm(snapped - point, axis=1)
        progress = self.progress[:-1] + fraction * self.length
        planned = self.times[:-1] + fraction * np.diff(self.times)
        actual = np.datetime64(event_time, "s").astype("int64")
        score = distance + np.minimum(120.0, np.abs(planned - actual) * 0.025)
        if self.previous_progress is not None and self.previous_time is not None:
            elapsed = max(0.0, (event_time - self.previous_time).total_seconds())
            backward = np.maximum(0.0, self.previous_progress - progress - 30)
            excessive_forward = np.maximum(0.0, progress - self.previous_progress
                                           - (elapsed * 40 + 300))
            score += np.minimum(1000.0, backward * 2 + excessive_forward)
        score[self.length2 < 1] = math.inf
        score[np.diff(self.times) <= 0] = math.inf
        index = int(np.argmin(score))
        if not np.isfinite(score[index]):
            return None, math.inf
        metres = float(distance[index])
        deviation = actual - float(planned[index])
        if metres > MAX_MATCH_DISTANCE_M or abs(deviation) > MAX_DEVIATION_S:
            return None, metres
        self.previous_progress = float(progress[index])
        self.previous_index = index
        self.previous_time = event_time
        return deviation, metres


def build_stream_features(dataset: Path, output_dir: Path,
                          split: str = "validate") -> dict[str, int]:
    """Build requests and per-packet availability using only causal inputs."""
    if split not in {"train", "test", "validate"}:
        raise ValueError(f"Unknown split: {split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = DataSource(dataset)
    try:
        schedule_file = "validate/schedule_plan.csv" if split == "validate" else f"{split}/schedule.csv"
        stops, planned_times = _schedule(source, schedule_file)
        with source.csv_rows(f"{split}/traffic.csv") as reader:
            packets = [clean_traffic_row(row) for row in reader]
    finally:
        source.close()
    packets = [p for p in packets if p.available_time is not None]
    packets.sort(key=lambda p: (p.available_time, p.event_time, p.packet_id))
    unique_packets = []
    seen_packets: set[str] = set()
    for packet in packets:
        if packet.packet_id not in seen_packets:
            unique_packets.append(packet)
            seen_packets.add(packet.packet_id)
    packets = unique_packets
    index = TrafficIndex(packets)
    schedule = add_route_signatures(load_schedule(dataset, split))
    segments = build_route_segments(schedule)
    matchers = {str(vehicle): RouteMatcher(group) for vehicle, group in schedule.groupby("tr_id")}
    stop_ids = defaultdict(list)
    for vehicle, group in schedule.groupby("tr_id"):
        ordered = group.sort_values("time_begin", kind="stable")
        stop_ids[str(vehicle)] = ordered.tt_action_item_id.astype(str).tolist()
    requests: list[dict] = []
    statuses: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    latest_event: dict[str, datetime] = {}
    stop_state: dict[str, tuple[datetime, float]] = {}
    segment_speed_state: dict[str, tuple[int, float, int]] = {}
    for packet in packets:
        vehicle = str(packet.tr_id)
        if vehicle not in matchers:
            continue
        at = packet.available_time
        status = {"packet_id": packet.packet_id, "tr_id": vehicle,
                  "available_time": at, "event_time": packet.event_time,
                  "sample_id": None, "availability": "no_target",
                  "gps_match_distance_m": None, "estimated_cur_dev_s": None,
                  "mean_speed_5m": None, "segment_speed_mean_kmh": None,
                  "stopped_duration_s": 0.0}
        speeds = [item.speed for item in index.window(vehicle, at, 300)
                  if item.speed is not None]
        status["mean_speed_5m"] = float(np.mean(speeds)) if speeds else None
        times = planned_times[vehicle]
        target_index = bisect_right(times, at + timedelta(seconds=600))
        target_exists = (target_index < len(times)
                         and times[target_index] <= at + timedelta(seconds=900))
        late = vehicle in latest_event and packet.event_time < latest_event[vehicle]
        if not late:
            latest_event[vehicle] = packet.event_time
        if late:
            status["availability"] = "late_packet"
        elif not packet.gps_valid or (at - packet.event_time).total_seconds() > GPS_MAX_AGE_S:
            status["availability"] = "bad_gps"
        else:
            previous_stop = stop_state.get(vehicle)
            if packet.speed is not None and packet.speed <= 2:
                if previous_stop is not None:
                    gap = (packet.event_time - previous_stop[0]).total_seconds()
                    stopped = previous_stop[1] + gap if 0 <= gap <= 120 else 0.0
                else:
                    stopped = 0.0
                stop_state[vehicle] = (packet.event_time, stopped)
                status["stopped_duration_s"] = stopped
            else:
                stop_state.pop(vehicle, None)
            deviation, distance = matchers[vehicle].match(
                packet.lon, packet.lat, packet.event_time, packet.speed)
            status["gps_match_distance_m"] = distance if math.isfinite(distance) else None
            if deviation is None:
                status["availability"] = "off_route"
            else:
                status["estimated_cur_dev_s"] = float(deviation)
                segment = matchers[vehicle].previous_index
                if segment is not None and packet.speed is not None:
                    previous_speed = segment_speed_state.get(vehicle)
                    if previous_speed is not None and previous_speed[0] == segment:
                        total, count = previous_speed[1] + packet.speed, previous_speed[2] + 1
                    else:
                        total, count = packet.speed, 1
                    segment_speed_state[vehicle] = (segment, total, count)
                    status["segment_speed_mean_kmh"] = total / count
            if deviation is not None and target_exists:
                target_id = stop_ids[vehicle][target_index]
                sample_id = f"stream:{packet.packet_id}:{target_id}"
                sample = {"sample_id": sample_id, "tr_id": vehicle, "T": at,
                          "target_stop_id": target_id,
                          "target_time_begin": times[target_index],
                          "cur_dev_s": float(deviation), "target_delay_s": None,
                          "target_delta_s": None}
                row = _features(sample, stops[target_id], times, index)
                row["packet_id"] = packet.packet_id
                row["available_time"] = at
                row["gps_match_distance_m"] = distance
                requests.append(row)
                status["sample_id"] = sample_id
                status["availability"] = "ready"
        counts[status["availability"]] += 1
        statuses.append(status)
    if not requests:
        raise ValueError("No stream forecast requests could be built")
    features = pd.DataFrame(requests)
    features["T"] = pd.to_datetime(features["T"]).astype("datetime64[us]")
    features["target_time_begin"] = pd.to_datetime(
        features["target_time_begin"]).astype("datetime64[us]")
    features = attach_planned_sections(features, segments)
    features = add_delay_trend_features(features)
    features.to_parquet(output_dir / "stream_features.parquet", index=False)
    pd.DataFrame(statuses).to_parquet(output_dir / "stream_status.parquet", index=False)
    source_stat = dataset.stat()
    report = {"version": STREAM_VERSION, "split": split,
              "source_size": source_stat.st_size,
              "source_mtime_ns": source_stat.st_mtime_ns, "packets": len(statuses),
              "forecast_requests": len(features), "availability": dict(counts)}
    (output_dir / "stream_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
