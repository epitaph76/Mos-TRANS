from pathlib import Path
import hashlib
import re
import zipfile

import numpy as np
import pandas as pd

SCHEDULE_FILES = {
    "train": "train/schedule.csv",
    "test": "test/schedule.csv",
    "validate": "validate/schedule_plan.csv",
}

POINT_PATTERN =  re.compile(
    r"POINT\s*\("
    r"([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)"
    r"\)"
)

MAX_SECTION_DURATION_S = 1800

def find_zip_member(archive:zipfile.ZipFile, suffix: str) -> str:
    candidates = [name for name in archive.namelist()
                  if name.endswith(suffix)]

    if len(candidates) != 1:
        raise ValueError(f"Не удалось найти {suffix}: "f"{candidates}")
    return candidates[0]


def parse_point(value: str) -> tuple[float, float]:
    match = POINT_PATTERN.fullmatch(str(value).strip())

    if match is None:
        raise ValueError(f"Неверная геометрия: {value}")

    lon = float(match.group(1))
    lat = float(match.group(2))

    return lon, lat


def make_route_signature(vehicle_schedule: pd.DataFrame) -> str:
    ordered = vehicle_schedule.sort_values("time_begin")

    points = [
        (round(float(lon), 5), round(float(lat), 5))
        for lon, lat in zip(ordered["stop_lon"], ordered["stop_lat"])
    ]

    raw = "|".join(
        f"{lon:.5f},{lat:.5f}"
        for lon, lat in points
    )

    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def add_route_signatures(schedule: pd.DataFrame) -> pd.DataFrame:
    schedule = schedule.copy()
    signatures = {}

    for tr_id, vehicle_schedule in schedule.groupby("tr_id"):
        signatures[tr_id] = (make_route_signature(vehicle_schedule))

    schedule["route_signature"] = (schedule["tr_id"].map(signatures))

    return schedule


def make_section_id(row: pd.Series) -> str:
    return (
        f"{row['route_signature']}:"
        f"{row['stop_lon']:.5f},"
        f"{row['stop_lat']:.5f}:"
        f"{row['next_lon']:.5f},"
        f"{row['next_lat']:.5f}"
    )


def build_route_segments(schedule: pd.DataFrame) -> pd.DataFrame:
    schedule = (schedule.sort_values(["tr_id", "time_begin"]).copy())
    grouped = schedule.groupby("tr_id", sort=False)

    schedule["next_time"] = grouped["time_begin"].shift(-1)
    schedule["next_lon"] = grouped["stop_lon"].shift(-1)
    schedule["next_lat"] = grouped["stop_lat"].shift(-1)
    schedule["next_address"] = grouped["building_address"].shift(-1)

    segments = schedule.dropna(
        subset=[
            "next_time",
            "next_lon",
            "next_lat",
        ]
    ).copy()

    segments["section_duration_s"] = (segments["next_time"] - segments["time_begin"]).dt.total_seconds()
    segments = segments[(segments["section_duration_s"] > 0)
                        & (segments["section_duration_s"] <= MAX_SECTION_DURATION_S)].copy()
    segments["section_id"] = segments.apply(make_section_id, axis=1)

    return segments


def load_schedule(dataset_path: Path, split: str) -> pd.DataFrame:
    member_suffix = SCHEDULE_FILES[split]

    with zipfile.ZipFile(dataset_path) as archive:
        member = find_zip_member(archive, member_suffix)

        schedule = pd.read_csv(archive.open(member), dtype={
            "tr_id": str,
            "tt_action_item_id": str
            }
        )

    coordinates = schedule["geom"].map(parse_point)
    schedule["stop_lon"] = [point[0] for point in coordinates]
    schedule["stop_lat"] = [point[1] for point in coordinates]
    schedule["time_begin"] = (pd.to_datetime(schedule["time_begin"]).astype("datetime64[us]"))

    return schedule

def attach_planned_sections(frame: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["_original_order"] = np.arange(len(frame))

    route_by_vehicle = (segments[["tr_id", "route_signature"]]
                        .drop_duplicates("tr_id")
                        .set_index("tr_id")["route_signature"])
    
    frame["route_signature"] = frame["tr_id"].map(route_by_vehicle)
    
    parts = []

    for tr_id, rows in frame.groupby("tr_id", sort=False):
        vehicle_segments = (segments[segments["tr_id"] == tr_id]
                            .sort_values("time_begin")
                            .copy())
        rows = rows.sort_values("T").copy()

        if vehicle_segments.empty:
            rows["section_from_time"] = pd.NaT
            rows["section_to_time"] = pd.NaT
            rows["section_id"] = pd.NA
            rows["section_duration_s"] = np.nan
            rows["stop_lon"] = np.nan
            rows["stop_lat"] = np.nan
            rows["next_lon"] = np.nan
            rows["next_lat"] = np.nan
            rows["section_planned_progress"] = np.nan
            rows["planned_section_available"] = 0

            parts.append(rows)
            continue

        enriched = pd.merge_asof(
            rows,
            vehicle_segments[
                [
                    "time_begin",
                    "next_time",
                    "section_id",
                    "section_duration_s",
                    "stop_lon",
                    "stop_lat",
                    "next_lon",
                    "next_lat",
                ]
            ].sort_values("time_begin"),
            left_on="T",
            right_on="time_begin",
            direction="backward",
        )

        enriched = enriched.rename(columns={"time_begin": ("section_from_time"),
                                            "next_time": ("section_to_time")})

        valid_section = (enriched["section_from_time"].notna()
                         & enriched["section_to_time"].notna()
                         & (enriched["T"] >= enriched["section_from_time"])
                         & (enriched["T"] < enriched["section_to_time"]))
        
        segment_columns = [
            "section_id",
            "section_duration_s",
            "stop_lon",
            "stop_lat",
            "next_lon",
            "next_lat",
        ]

        enriched.loc[~valid_section, segment_columns] = np.nan
        enriched["section_planned_progress"] = np.nan
        enriched.loc[valid_section, "section_planned_progress"] = ((enriched.loc[valid_section, "T"]
                                                                    - enriched.loc[valid_section,"section_from_time"])
                                                                    .dt.total_seconds()
                                                                    / enriched.loc[valid_section, "section_duration_s"])

        enriched["section_planned_progress"] = (enriched["section_planned_progress"].clip(0.0, 1.0))
        enriched["planned_section_available"] = valid_section.astype(int)

        parts.append(enriched)

    result = pd.concat(parts, ignore_index=True)

    result = (result.sort_values("_original_order")
              .drop(columns="_original_order")
              .reset_index(drop=True)
    )

    return result