"""Build point-in-time safe, model-independent transport features.

Every feature for a sample at T is derived from telemetry whose event and
receive timestamps are both no later than T. Schedule facts are never read.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import zipfile
from bisect import bisect_left, bisect_right
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq


SPLITS = {
    "train": ("labels/labels_train.csv", "train/schedule.csv"),
    "test": ("labels/labels_test.csv", "test/schedule.csv"),
    "validate": ("validate/points.csv", "validate/schedule_plan.csv"),
}
TRAFFIC_FILES = ("train/traffic.csv", "test/traffic.csv", "validate/traffic.csv")
WINDOWS_MIN = (2, 5, 10)
GPS_MAX_AGE_S = 120
SPEED_MAX_AGE_S = 120
CONTEXT_MAX_AGE_S = 60
STOPPED_SPEED_KMH = 2.0
CONTEXT_RADIUS_M = 500.0
POINT_RE = re.compile(r"^POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)$")


def _time(value: str) -> datetime:
    """Parse source timestamps without changing their common time scale."""
    return datetime.fromisoformat(value)


def _number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _point(value: str) -> tuple[float, float]:
    match = POINT_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid schedule geometry: {value!r}")
    lon, lat = map(float, match.groups())
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError(f"Invalid schedule coordinates: {value!r}")
    return lon, lat


@dataclass(frozen=True, slots=True)
class Telemetry:
    packet_id: str
    tr_id: str
    unit_id: str
    event_time: datetime
    receive_time: datetime | None
    location_valid: bool
    gps_valid: bool
    lon: float | None
    lat: float | None
    speed: float | None
    speed_outlier: bool
    heading: float | None
    is_hist_data: bool

    @property
    def available_time(self) -> datetime | None:
        if self.receive_time is None:
            return None
        return max(self.event_time, self.receive_time)


def clean_traffic_row(row: dict[str, str]) -> Telemetry:
    """Apply the same GPS and speed rules for all model branches."""
    speed = _number(row["speed"])
    speed_outlier = speed is not None and not (0 <= speed <= 100)
    if speed_outlier:
        speed = None
    lon, lat = _number(row["lon"]), _number(row["lat"])
    location_valid = row["location_valid"].strip().lower() == "true"
    gps_valid = bool(
        location_valid
        and lon is not None
        and lat is not None
        and -180 <= lon <= 180
        and -90 <= lat <= 90
    )
    if not gps_valid:
        lon = lat = None
    heading = _number(row["heading"])
    if heading is not None and not 0 <= heading <= 360:
        heading = None
    return Telemetry(
        packet_id=row["packet_id"],
        tr_id=row["tr_id"],
        unit_id=row["unit_id"],
        event_time=_time(row["event_time"]),
        receive_time=_time(row["receive_time"]) if row["receive_time"] else None,
        location_valid=location_valid,
        gps_valid=gps_valid,
        lon=lon,
        lat=lat,
        speed=speed,
        speed_outlier=speed_outlier,
        heading=heading,
        is_hist_data=row["is_hist_data"].strip().lower() == "true",
    )


class DataSource:
    """Read CSV members directly from dataset.zip or an extracted directory."""

    def __init__(self, path: Path):
        self.path = path
        self.prefix = ""
        if path.is_file() and path.suffix.lower() == ".zip":
            self.archive = zipfile.ZipFile(path)
            names = set(self.archive.namelist())
            anchor = "train/traffic.csv"
            if anchor not in names:
                candidates = sorted(name[: -len(anchor)] for name in names if name.endswith(anchor))
                if len(candidates) != 1:
                    self.archive.close()
                    raise ValueError(
                        "Could not locate a unique train/traffic.csv root inside "
                        f"archive: {path}"
                    )
                self.prefix = candidates[0]
        elif path.is_dir():
            self.archive = None
            if not (self.path / "train" / "traffic.csv").is_file():
                candidates = sorted(self.path.glob("*/train/traffic.csv"))
                if len(candidates) != 1:
                    raise ValueError(
                        "Could not locate a unique train/traffic.csv root inside "
                        f"directory: {path}"
                    )
                self.path = candidates[0].parents[1]
        else:
            raise ValueError(f"Expected dataset.zip or extracted directory: {path}")

    @contextmanager
    def csv_rows(self, member: str) -> Iterator[csv.DictReader]:
        if self.archive:
            with self.archive.open(f"{self.prefix}{member}") as binary:
                with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as stream:
                    yield csv.DictReader(stream)
        else:
            with (self.path / member).open(encoding="utf-8-sig", newline="") as stream:
                yield csv.DictReader(stream)

    def close(self) -> None:
        if self.archive:
            self.archive.close()


class TrafficIndex:
    """Per-vehicle event-time index with receive-time availability checks."""

    def __init__(self, packets: list[Telemetry]):
        grouped: dict[str, list[Telemetry]] = defaultdict(list)
        for packet in packets:
            grouped[packet.tr_id].append(packet)
        self.packets: dict[str, list[Telemetry]] = {}
        self.times: dict[str, list[datetime]] = {}
        for tr_id, group in grouped.items():
            group.sort(key=lambda p: (p.event_time, p.receive_time or datetime.max, p.packet_id))
            self.packets[tr_id] = group
            self.times[tr_id] = [p.event_time for p in group]

    def window(self, tr_id: str, now: datetime, seconds: int) -> list[Telemetry]:
        times = self.times.get(tr_id)
        if times is None:
            return []
        start = bisect_left(times, now - timedelta(seconds=seconds))
        stop = bisect_right(times, now)
        return [p for p in self.packets[tr_id][start:stop] if p.receive_time is not None and p.receive_time <= now]

    def latest(self, tr_id: str, now: datetime, *, max_age_s: int | None = None, gps: bool = False, speed: bool = False) -> Telemetry | None:
        times = self.times.get(tr_id)
        if times is None:
            return None
        stop = bisect_right(times, now)
        cutoff = now - timedelta(seconds=max_age_s) if max_age_s is not None else None
        for position in range(stop - 1, -1, -1):
            packet = self.packets[tr_id][position]
            if cutoff is not None and packet.event_time < cutoff:
                break
            if packet.receive_time is None or packet.receive_time > now:
                continue
            if gps and not packet.gps_valid:
                continue
            if speed and packet.speed is None:
                continue
            return packet
        return None


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = lat2 - lat1, math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000 * math.asin(min(1.0, math.sqrt(a)))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _context(index: TrafficIndex, tr_id: str, now: datetime, gps: Telemetry | None) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = {
        "context_vehicle_count_500m": 0,
        "context_speed_count_500m": 0,
        "context_speed_mean_500m": None,
        "context_speed_median_500m": None,
        "context_stopped_fraction_500m": None,
    }
    if gps is None:
        return result
    speeds: list[float] = []
    count = 0
    for other_id in index.packets:
        if other_id == tr_id:
            continue
        other = index.latest(other_id, now, max_age_s=CONTEXT_MAX_AGE_S, gps=True)
        if other is None:
            continue
        if _haversine_m(gps.lon, gps.lat, other.lon, other.lat) > CONTEXT_RADIUS_M:
            continue
        count += 1
        if other.speed is not None:
            speeds.append(other.speed)
    result["context_vehicle_count_500m"] = count
    result["context_speed_count_500m"] = len(speeds)
    result["context_speed_mean_500m"] = _mean(speeds)
    result["context_speed_median_500m"] = _median(speeds)
    result["context_stopped_fraction_500m"] = _mean([float(s <= STOPPED_SPEED_KMH) for s in speeds])
    return result


def _schedule(source: DataSource, member: str) -> tuple[dict[str, dict], dict[str, list[datetime]]]:
    stops: dict[str, dict] = {}
    by_vehicle: dict[str, list[datetime]] = defaultdict(list)
    with source.csv_rows(member) as reader:
        for row in reader:
            stop_id = row["tt_action_item_id"]
            if stop_id in stops:
                raise ValueError(f"Duplicate schedule stop: {stop_id}")
            lon, lat = _point(row["geom"])
            planned = _time(row["time_begin"])
            stops[stop_id] = {"tr_id": row["tr_id"], "time": planned, "lon": lon, "lat": lat, "address": row["building_address"]}
            by_vehicle[row["tr_id"]].append(planned)
    for times in by_vehicle.values():
        times.sort()
    return stops, by_vehicle


def _sample(row: dict[str, str], stops: dict[str, dict]) -> dict:
    stop_id = row["target_stop_id"]
    if stop_id not in stops:
        raise ValueError(f"Target stop {stop_id} is missing from its split schedule")
    stop = stops[stop_id]
    now, target_time = _time(row["T"]), _time(row["target_time_begin"])
    if stop["tr_id"] != row["tr_id"] or stop["time"] != target_time:
        raise ValueError(f"Target stop/time mismatch for {row['sample_id']}")
    horizon = (target_time - now).total_seconds()
    if not 600 < horizon <= 900:
        raise ValueError(f"Forecast horizon outside (600, 900] s for {row['sample_id']}: {horizon}")
    sample = {
        "sample_id": row["sample_id"],
        "tr_id": row["tr_id"],
        "T": now,
        "target_stop_id": stop_id,
        "target_time_begin": target_time,
        "cur_dev_s": float(row["cur_dev_s"]),
        "target_delay_s": None,
        "target_delta_s": None,
    }
    if "target_delay_s" in row:
        sample["target_delay_s"] = float(row["target_delay_s"])
        sample["target_delta_s"] = sample["target_delay_s"] - sample["cur_dev_s"]
    return sample


def _features(sample: dict, stop: dict, planned_times: list[datetime], index: TrafficIndex) -> dict:
    tr_id, now = sample["tr_id"], sample["T"]
    result = dict(sample)
    last_packet = index.latest(tr_id, now)
    last_speed = index.latest(tr_id, now, max_age_s=SPEED_MAX_AGE_S, speed=True)
    last_gps = index.latest(tr_id, now, max_age_s=GPS_MAX_AGE_S, gps=True)
    result.update({
        "realtime_data_age_s": (now - last_packet.event_time).total_seconds() if last_packet else None,
        "realtime_speed_last": last_speed.speed if last_speed else None,
        "realtime_speed_age_s": (now - last_speed.event_time).total_seconds() if last_speed else None,
        "realtime_gps_age_s": (now - last_gps.event_time).total_seconds() if last_gps else None,
        "realtime_distance_to_target_m": _haversine_m(last_gps.lon, last_gps.lat, stop["lon"], stop["lat"]) if last_gps else None,
    })
    for minutes in WINDOWS_MIN:
        packets = index.window(tr_id, now, minutes * 60)
        speeds = [p.speed for p in packets if p.speed is not None]
        suffix = f"{minutes}m"
        result[f"realtime_packet_count_{suffix}"] = len(packets)
        result[f"realtime_speed_count_{suffix}"] = len(speeds)
        result[f"realtime_speed_mean_{suffix}"] = _mean(speeds)
        result[f"realtime_speed_std_{suffix}"] = _std(speeds)
        result[f"realtime_stopped_fraction_{suffix}"] = _mean([float(s <= STOPPED_SPEED_KMH) for s in speeds])
    result.update(_context(index, tr_id, now, last_gps))
    target_time = sample["target_time_begin"]
    result.update({
        "schedule_horizon_s": (target_time - now).total_seconds(),
        "schedule_target_lon": stop["lon"],
        "schedule_target_lat": stop["lat"],
        "schedule_target_address": stop["address"],
        "schedule_stops_remaining": bisect_right(planned_times, target_time) - bisect_right(planned_times, now),
        "historical_hour": now.hour,
        "historical_day_of_week": now.weekday(),
        "historical_time_sin": math.sin(2 * math.pi * (now.hour * 3600 + now.minute * 60 + now.second) / 86400),
        "historical_time_cos": math.cos(2 * math.pi * (now.hour * 3600 + now.minute * 60 + now.second) / 86400),
    })
    return result


SAMPLE_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("tr_id", pa.string()), ("T", pa.timestamp("us")),
    ("target_stop_id", pa.string()), ("target_time_begin", pa.timestamp("us")),
    ("cur_dev_s", pa.float64()), ("target_delay_s", pa.float64()), ("target_delta_s", pa.float64()),
])
TRAFFIC_SCHEMA = pa.schema([
    ("packet_id", pa.string()), ("tr_id", pa.string()), ("unit_id", pa.string()),
    ("event_time", pa.timestamp("us")), ("receive_time", pa.timestamp("us")),
    ("available_time", pa.timestamp("us")), ("location_valid", pa.bool_()),
    ("gps_valid", pa.bool_()), ("lon", pa.float64()), ("lat", pa.float64()),
    ("speed", pa.float64()), ("speed_outlier", pa.bool_()), ("heading", pa.float64()),
    ("is_hist_data", pa.bool_()),
])
feature_fields = list(SAMPLE_SCHEMA)
feature_fields += [
    pa.field("realtime_data_age_s", pa.float64()),
    pa.field("realtime_speed_last", pa.float64()),
    pa.field("realtime_speed_age_s", pa.float64()),
    pa.field("realtime_gps_age_s", pa.float64()),
    pa.field("realtime_distance_to_target_m", pa.float64()),
]
for _minutes in WINDOWS_MIN:
    _suffix = f"{_minutes}m"
    feature_fields.extend([
        pa.field(f"realtime_packet_count_{_suffix}", pa.int64()),
        pa.field(f"realtime_speed_count_{_suffix}", pa.int64()),
        pa.field(f"realtime_speed_mean_{_suffix}", pa.float64()),
        pa.field(f"realtime_speed_std_{_suffix}", pa.float64()),
        pa.field(f"realtime_stopped_fraction_{_suffix}", pa.float64()),
    ])
feature_fields.extend([
    pa.field("context_vehicle_count_500m", pa.int64()),
    pa.field("context_speed_count_500m", pa.int64()),
    pa.field("context_speed_mean_500m", pa.float64()),
    pa.field("context_speed_median_500m", pa.float64()),
    pa.field("context_stopped_fraction_500m", pa.float64()),
    pa.field("schedule_horizon_s", pa.float64()),
    pa.field("schedule_target_lon", pa.float64()),
    pa.field("schedule_target_lat", pa.float64()),
    pa.field("schedule_target_address", pa.string()),
    pa.field("schedule_stops_remaining", pa.int64()),
    pa.field("historical_hour", pa.int64()),
    pa.field("historical_day_of_week", pa.int64()),
    pa.field("historical_time_sin", pa.float64()),
    pa.field("historical_time_cos", pa.float64()),
])
FEATURE_SCHEMA = pa.schema(feature_fields)


def _write(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def build_dataset(input_path: str | Path, output_path: str | Path) -> dict[str, int]:
    """Create one cleaned traffic pool and aligned samples/features per split."""
    source = DataSource(Path(input_path))
    output = Path(output_path)
    try:
        unique: dict[str, Telemetry] = {}
        for member in TRAFFIC_FILES:
            with source.csv_rows(member) as reader:
                for row in reader:
                    packet = clean_traffic_row(row)
                    previous = unique.setdefault(packet.packet_id, packet)
                    if previous != packet:
                        raise ValueError(f"Conflicting packet_id across traffic files: {packet.packet_id}")
        packets = sorted(unique.values(), key=lambda p: (p.tr_id, p.event_time, p.receive_time or datetime.max, p.packet_id))
        index = TrafficIndex(packets)
        _write(output / "traffic_clean.parquet", [
            {
                "packet_id": p.packet_id, "tr_id": p.tr_id, "unit_id": p.unit_id,
                "event_time": p.event_time, "receive_time": p.receive_time,
                "available_time": p.available_time, "location_valid": p.location_valid,
                "gps_valid": p.gps_valid, "lon": p.lon, "lat": p.lat,
                "speed": p.speed, "speed_outlier": p.speed_outlier,
                "heading": p.heading, "is_hist_data": p.is_hist_data,
            }
            for p in packets
        ], TRAFFIC_SCHEMA)
        counts = {"traffic": len(packets)}
        for split, (points_file, schedule_file) in SPLITS.items():
            stops, by_vehicle = _schedule(source, schedule_file)
            with source.csv_rows(points_file) as reader:
                samples = [_sample(row, stops) for row in reader]
            samples.sort(key=lambda s: s["sample_id"])
            if len(samples) != len({s["sample_id"] for s in samples}):
                raise ValueError(f"Duplicate sample_id in {split}")
            features = [_features(s, stops[s["target_stop_id"]], by_vehicle[s["tr_id"]], index) for s in samples]
            _write(output / f"{split}_samples.parquet", samples, SAMPLE_SCHEMA)
            _write(output / f"{split}_features.parquet", features, FEATURE_SCHEMA)
            counts[split] = len(samples)
        return counts
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shared Mos-TRANS preprocessing artifacts")
    parser.add_argument("--input", required=True, help="Path to dataset.zip or extracted dataset directory")
    parser.add_argument("--output", required=True, help="Directory for ignored Parquet output")
    args = parser.parse_args()
    counts = build_dataset(args.input, args.output)
    print("Prepared " + ", ".join(f"{name}={count}" for name, count in counts.items()))
