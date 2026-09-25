"""Point-in-time movement and planned-stop-route features for ETA experiments."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mos_trans.preprocessing.core import DataSource, SPLITS, _point


WINDOWS = (120, 300, 600)
MAX_GPS_AGE_S = 120
MAX_SPEED_AGE_S = 120
EARTH_M_PER_DEGREE = 111_195.0
FEATURES = (
    "progress_route_remaining_m", "progress_route_snap_error_m",
    "progress_next_stop_distance_m", "progress_stop_streak_s",
    *(f"progress_route_{s // 60}m_m" for s in WINDOWS),
    *(f"progress_direct_{s // 60}m_m" for s in WINDOWS),
    *(f"progress_stopped_{s // 60}m_s" for s in WINDOWS),
    *(f"progress_gps_updates_{s // 60}m" for s in WINDOWS),
)


def _ns(value) -> int:
    return int(pd.Timestamp(value).value)


def _xy(lon: float, lat: float, latitude_ref: float) -> np.ndarray:
    return np.array([lon * EARTH_M_PER_DEGREE * math.cos(math.radians(latitude_ref)),
                     lat * EARTH_M_PER_DEGREE], dtype=float)


def _distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    latitude_ref = (lat1 + lat2) / 2
    return float(np.linalg.norm(_xy(lon1, lat1, latitude_ref) - _xy(lon2, lat2, latitude_ref)))


@dataclass
class Route:
    ids: list[str]
    times: np.ndarray
    points: np.ndarray
    cumulative: np.ndarray
    positions: dict[str, int]
    latitude_ref: float

    def remaining(self, lon: float, lat: float, target: int) -> tuple[float, float]:
        """Project on stop-to-stop chords; this approximates, not maps, the road route."""
        point = _xy(lon, lat, self.latitude_ref)
        first = max(0, target - 30)
        route_xy = self.points[first:target + 1]
        if target == first:
            distance = float(np.linalg.norm(point - route_xy[0]))
            return distance, distance
        start = route_xy[:-1]
        vectors = route_xy[1:] - start
        lengths_sq = (vectors * vectors).sum(axis=1)
        fraction = np.divide(((point - start) * vectors).sum(axis=1), lengths_sq,
                             out=np.zeros_like(lengths_sq), where=lengths_sq > 0)
        fraction = np.clip(fraction, 0, 1)
        projected = start + fraction[:, None] * vectors
        errors = np.linalg.norm(projected - point, axis=1)
        local_segment = int(np.argmin(errors))
        segment = first + local_segment
        remaining = float(self.cumulative[target] - self.cumulative[segment]
                          - fraction[local_segment] * math.sqrt(lengths_sq[local_segment]))
        return remaining, float(errors[local_segment])


def load_plans(dataset_zip: str | Path) -> dict[str, dict[str, Route]]:
    """Read only planned stop IDs, times and geometry, never factual arrival times."""
    source = DataSource(Path(dataset_zip))
    result = {}
    try:
        for split, (_, schedule_file) in SPLITS.items():
            grouped: dict[str, list[tuple[str, int, float, float]]] = {}
            with source.csv_rows(schedule_file) as rows:
                for row in rows:
                    lon, lat = _point(row["geom"])
                    grouped.setdefault(str(row["tr_id"]), []).append(
                        (str(row["tt_action_item_id"]), _ns(row["time_begin"]), lon, lat))
            routes = {}
            for vehicle, stops in grouped.items():
                stops.sort(key=lambda value: (value[1], value[0]))
                ids = [stop[0] for stop in stops]
                if len(ids) != len(set(ids)):
                    raise ValueError("Duplicate planned stop ID")
                longitude_latitude = np.asarray([(stop[2], stop[3]) for stop in stops], dtype=float)
                ref = float(longitude_latitude[:, 1].mean())
                points = np.asarray([_xy(lon, lat, ref) for lon, lat in longitude_latitude])
                cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
                routes[vehicle] = Route(ids, np.asarray([stop[1] for stop in stops], dtype=np.int64),
                                        points, cumulative, {key: i for i, key in enumerate(ids)}, ref)
            result[split] = routes
    finally:
        source.close()
    return result


class VehiclePackets:
    def __init__(self, frame: pd.DataFrame):
        ordered = frame.sort_values(["event_time", "packet_id"])
        self.event = ordered.event_time.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.available = ordered.available_time.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.has_receipt = self.available != np.iinfo(np.int64).min
        self.gps = ordered.gps_valid.fillna(False).to_numpy(bool)
        self.lon = ordered.lon.to_numpy(float)
        self.lat = ordered.lat.to_numpy(float)
        self.speed = ordered.speed.to_numpy(float)

    def latest_gps(self, cutoff_ns: int) -> int | None:
        stop = int(np.searchsorted(self.event, cutoff_ns, side="right"))
        oldest = cutoff_ns - MAX_GPS_AGE_S * 1_000_000_000
        for i in range(stop - 1, -1, -1):
            if self.event[i] < oldest:
                break
            if self.has_receipt[i] and self.available[i] <= cutoff_ns and self.gps[i]:
                return i
        return None

    def window(self, now_ns: int, seconds: int) -> np.ndarray:
        start_ns = now_ns - seconds * 1_000_000_000
        start = int(np.searchsorted(self.event, start_ns, side="left"))
        end = int(np.searchsorted(self.event, now_ns, side="right"))
        return np.arange(start, end)[self.has_receipt[start:end] & (self.available[start:end] <= now_ns)]

    def stopped_seconds(self, now_ns: int, seconds: int) -> float:
        """Last known speed persists at most 120 s; late packets count only once received."""
        start_ns = now_ns - seconds * 1_000_000_000
        indices = self.window(now_ns, seconds)
        before = int(np.searchsorted(self.event, start_ns, side="right")) - 1
        while before >= 0:
            if self.has_receipt[before] and self.available[before] <= now_ns and np.isfinite(self.speed[before]):
                if self.event[before] >= start_ns - MAX_SPEED_AGE_S * 1_000_000_000:
                    indices = np.r_[before, indices]
                break
            before -= 1
        indices = indices[np.isfinite(self.speed[indices])]
        total = 0.0
        for position, i in enumerate(indices):
            end_ns = min(now_ns, self.event[i] + MAX_SPEED_AGE_S * 1_000_000_000)
            if position + 1 < len(indices):
                end_ns = min(end_ns, self.event[indices[position + 1]])
            begin_ns = max(start_ns, self.event[i])
            if self.speed[i] <= 2 and end_ns > begin_ns:
                total += (end_ns - begin_ns) / 1e9
        return min(float(seconds), total)

    def stop_streak(self, now_ns: int) -> float:
        stop = int(np.searchsorted(self.event, now_ns, side="right"))
        previous = now_ns
        beginning = now_ns
        for i in range(stop - 1, -1, -1):
            if not self.has_receipt[i] or self.available[i] > now_ns or not np.isfinite(self.speed[i]):
                continue
            if (previous - self.event[i]) > MAX_SPEED_AGE_S * 1_000_000_000 or self.speed[i] > 2:
                break
            beginning = self.event[i]
            previous = self.event[i]
        return min(600.0, float((now_ns - beginning) / 1e9))


def augment(frame: pd.DataFrame, traffic: pd.DataFrame, routes: dict[str, Route]) -> pd.DataFrame:
    """Add causal movement features to an aligned common-preprocessor table."""
    required = {"sample_id", "tr_id", "T", "target_stop_id", "schedule_target_lon", "schedule_target_lat"}
    if not required.issubset(frame) or "time_fact_begin" in frame:
        raise ValueError("Missing point fields or forbidden factual schedule time")
    packets = {str(vehicle): VehiclePackets(group) for vehicle, group in traffic.groupby("tr_id", sort=False)}
    rows = []
    for sample in frame.itertuples(index=False):
        vehicle = str(sample.tr_id)
        if vehicle not in routes or str(sample.target_stop_id) not in routes[vehicle].positions:
            raise ValueError(f"Missing planned route/target for {sample.sample_id}")
        route = routes[vehicle]
        target = route.positions[str(sample.target_stop_id)]
        now = _ns(sample.T)
        packet = packets.get(vehicle)
        result = {name: np.nan for name in FEATURES}
        for seconds in WINDOWS:
            suffix = f"{seconds // 60}m"
            result[f"progress_stopped_{suffix}_s"] = packet.stopped_seconds(now, seconds) if packet else 0.0
            result[f"progress_gps_updates_{suffix}"] = int(packet.gps[packet.window(now, seconds)].sum()) if packet else 0
        result["progress_stop_streak_s"] = packet.stop_streak(now) if packet else 0.0
        current = packet.latest_gps(now) if packet else None
        if current is not None:
            lon, lat = packet.lon[current], packet.lat[current]
            remaining, error = route.remaining(lon, lat, target)
            if error <= 300:
                result["progress_route_remaining_m"] = remaining
            result["progress_route_snap_error_m"] = error
            next_index = min(int(np.searchsorted(route.times, now, side="right")), target)
            next_lon = route.points[next_index, 0] / (EARTH_M_PER_DEGREE * math.cos(math.radians(route.latitude_ref)))
            next_lat = route.points[next_index, 1] / EARTH_M_PER_DEGREE
            result["progress_next_stop_distance_m"] = _distance_m(lon, lat, next_lon, next_lat)
            current_direct = _distance_m(lon, lat, sample.schedule_target_lon, sample.schedule_target_lat)
            for seconds in WINDOWS:
                past = packet.latest_gps(now - seconds * 1_000_000_000)
                if past is None:
                    continue
                past_lon, past_lat = packet.lon[past], packet.lat[past]
                past_remaining, past_error = route.remaining(past_lon, past_lat, target)
                suffix = f"{seconds // 60}m"
                route_change = past_remaining - remaining
                if error <= 300 and past_error <= 300 and abs(route_change) <= seconds * (100 / 3.6) + 300:
                    result[f"progress_route_{suffix}_m"] = route_change
                direct_change = (
                    _distance_m(past_lon, past_lat, sample.schedule_target_lon, sample.schedule_target_lat)
                    - current_direct)
                if abs(direct_change) <= seconds * (100 / 3.6) + 300:
                    result[f"progress_direct_{suffix}_m"] = direct_change
        rows.append(result)
    extra = pd.DataFrame(rows, index=frame.index)
    return pd.concat([frame, extra], axis=1)


def prepare(dataset_zip: str | Path, processed_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Cache augmented splits under a dataset-and-code hash."""
    dataset_zip, processed_dir, output_dir = map(Path, (dataset_zip, processed_dir, output_dir))
    digest = hashlib.sha256()
    for path in (dataset_zip, Path(__file__), processed_dir / "traffic_clean.parquet"):
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    root = output_dir / digest.hexdigest()[:20]
    paths = {split: root / f"{split}_features.parquet" for split in SPLITS}
    if all(path.is_file() for path in paths.values()):
        return paths
    root.mkdir(parents=True, exist_ok=True)
    traffic = pd.read_parquet(processed_dir / "traffic_clean.parquet")
    routes = load_plans(dataset_zip)
    for split, path in paths.items():
        frame = pd.read_parquet(processed_dir / f"{split}_features.parquet")
        augmented = augment(frame, traffic, routes[split])
        augmented.to_parquet(path, index=False)
    return paths
