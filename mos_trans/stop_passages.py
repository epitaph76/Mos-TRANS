"""Estimate a stop passage from the interval spent inside a stop radius."""

from __future__ import annotations

import math

METRES_PER_DEGREE = 111_320


def near_interval(a: dict, b: dict, stop: dict, radius_m: float = 35.0):
    """Return entry, exit and minimum distance for a linear GPS segment.

    Times are interpolated at the radius boundary. A segment wholly inside the
    radius contributes its full time span; a tangent contributes one instant.
    """
    scale_x = METRES_PER_DEGREE * math.cos(math.radians(stop["lat"]))
    ax = (a["lon"] - stop["lon"]) * scale_x
    ay = (a["lat"] - stop["lat"]) * METRES_PER_DEGREE
    bx = (b["lon"] - stop["lon"]) * scale_x
    by = (b["lat"] - stop["lat"]) * METRES_PER_DEGREE
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    closest = max(0.0, min(1.0, -(ax * dx + ay * dy) / length_sq)) if length_sq else 0.0
    minimum = math.hypot(ax + closest * dx, ay + closest * dy)
    if minimum > radius_m:
        return None
    if length_sq == 0:
        lo, hi = 0.0, 1.0
    else:
        projection = -(ax * dx + ay * dy) / length_sq
        perpendicular_sq = ax * ax + ay * ay - projection * projection * length_sq
        half_width = math.sqrt(max(0.0, (radius_m * radius_m - perpendicular_sq) / length_sq))
        lo, hi = max(0.0, projection - half_width), min(1.0, projection + half_width)
        if hi < lo:
            return None
    duration = b["time"] - a["time"]
    return a["time"] + duration * lo, a["time"] + duration * hi, minimum


def merge_intervals(hits: list[dict], max_gap_s: float = 45.0) -> list[dict]:
    """Group consecutive near-stop segments into separate visits."""
    visits: list[dict] = []
    for hit in sorted(hits, key=lambda item: item["entry_time"]):
        if visits and (hit["entry_time"] - visits[-1]["exit_time"]).total_seconds() <= max_gap_s:
            visit = visits[-1]
            visit["exit_time"] = max(visit["exit_time"], hit["exit_time"])
            if hit["distance"] < visit["distance"]:
                visit["distance"] = hit["distance"]
                visit["direction"] = hit["direction"]
        else:
            visits.append(dict(hit))
    for visit in visits:
        visit["time"] = visit["entry_time"] + (visit["exit_time"] - visit["entry_time"]) / 2
        visit["near_duration_s"] = (visit["exit_time"] - visit["entry_time"]).total_seconds()
    return visits


def strongest_visits(visits: list[dict], repeat_window_s: float = 100.0) -> list[dict]:
    """Keep one candidate when a vehicle briefly re-enters the same stop radius."""
    if not visits:
        return []
    clusters: list[list[dict]] = [[visits[0]]]
    for visit in visits[1:]:
        if (visit["entry_time"] - clusters[-1][-1]["exit_time"]).total_seconds() <= repeat_window_s:
            clusters[-1].append(visit)
        else:
            clusters.append([visit])
    return [min(group, key=lambda visit: (visit["distance"], -visit["near_duration_s"]))
            for group in clusters]
