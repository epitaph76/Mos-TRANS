"""GPS-to-route matching for decoded NDTP navigation fixes.

Coordinates use WGS84. Distances and projections use a local equirectangular
plane, which is accurate enough for short urban route segments in Moscow.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians


@dataclass(frozen=True)
class Fix:
    lon: float
    lat: float
    time_s: float
    speed_kmh: float = 0.0
    heading: float | None = None
    valid: bool = True


@dataclass(frozen=True)
class Match:
    lon: float
    lat: float
    distance_m: float
    progress_m: float
    confidence: float
    off_route: bool


def _xy(lon: float, lat: float, origin_lat: float) -> tuple[float, float]:
    return lon * 111_320 * cos(radians(origin_lat)), lat * 111_320


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    x1, y1 = _xy(a[0], a[1], (a[1] + b[1]) / 2)
    x2, y2 = _xy(b[0], b[1], (a[1] + b[1]) / 2)
    return hypot(x2 - x1, y2 - y1)


def clean_fixes(fixes: list[Fix]) -> list[Fix]:
    """Discard invalid NDTP coordinates, duplicates and impossible GPS jumps."""
    result: list[Fix] = []
    for fix in sorted(fixes, key=lambda item: item.time_s):
        if not fix.valid or not (36.5 < fix.lon < 38.5 and 55.0 < fix.lat < 56.5):
            continue
        if result:
            dt = fix.time_s - result[-1].time_s
            if dt <= 0:
                continue
            jump = distance_m((result[-1].lon, result[-1].lat), (fix.lon, fix.lat))
            if jump > max(100, dt * 36):  # 130 km/h with a GPS margin
                continue
            if jump < 8 and dt < 120:
                continue
        result.append(fix)
    return result


def simplify_route(fixes: list[Fix], spacing_m: float = 35) -> list[tuple[float, float]]:
    """Keep meaningful bends while limiting the payload sent to the map."""
    if not fixes:
        return []
    points = [(fix.lon, fix.lat) for fix in fixes]
    if len(points) < 3:
        return points

    def recurse(start: int, end: int, keep: set[int]) -> None:
        if end - start < 2:
            return
        a, b = points[start], points[end]
        farthest, max_dist = start, 0.0
        for index in range(start + 1, end):
            projected = _project(points[index], a, b)
            dist = distance_m(points[index], projected)
            if dist > max_dist:
                farthest, max_dist = index, dist
        if max_dist > spacing_m:
            keep.add(farthest)
            recurse(start, farthest, keep)
            recurse(farthest, end, keep)

    keep = {0, len(points) - 1}
    recurse(0, len(points) - 1, keep)
    return [points[index] for index in sorted(keep)]


def _project(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    lat0 = (a[1] + b[1]) / 2
    px, py = _xy(*point, lat0)
    ax, ay = _xy(*a, lat0)
    bx, by = _xy(*b, lat0)
    dx, dy = bx - ax, by - ay
    fraction = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy))) if dx or dy else 0.0
    return a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction


def match_fix(
    fix: Fix,
    route: list[tuple[float, float]],
    previous_progress_m: float | None = None,
) -> Match | None:
    """Score candidates by GPS distance, heading and forward route continuity."""
    if not fix.valid or len(route) < 2:
        return None
    progress = 0.0
    best: tuple[float, Match] | None = None
    for a, b in zip(route, route[1:]):
        length = distance_m(a, b)
        if length < 1:
            continue
        snapped = _project((fix.lon, fix.lat), a, b)
        dist = distance_m((fix.lon, fix.lat), snapped)
        along = progress + distance_m(a, snapped)
        score = dist
        if fix.heading is not None and fix.speed_kmh >= 6:
            bearing = (degrees(atan2((b[0] - a[0]) * cos(radians(fix.lat)), b[1] - a[1])) + 360) % 360
            delta = abs((fix.heading - bearing + 180) % 360 - 180)
            score += 35 * (delta / 180)
        if previous_progress_m is not None:
            score += min(300, max(0, previous_progress_m - along - 30) * 1.2)
        confidence = max(0.0, round(1 - dist / 150, 2))
        candidate = Match(snapped[0], snapped[1], round(dist, 1), round(along, 1), confidence, dist > 150)
        if best is None or score < best[0]:
            best = score, candidate
        progress += length
    return best[1] if best else None


def match_trace(fixes: list[Fix], route: list[tuple[float, float]]) -> Match | None:
    """Map-match consecutive NDTP fixes, carrying route progress forward."""
    previous: Match | None = None
    for fix in clean_fixes(fixes):
        previous = match_fix(fix, route, previous.progress_m if previous else None)
    return previous
