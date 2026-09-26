"""Conflate repeated GPS passes into shared, undirected display corridors.

This is a display network inferred from telemetry, not official road geometry.
Time gaps are never bridged, and differently oriented nearby streets are not
merged except at a shared junction. Vehicle ownership is retained on edges.
"""

from collections import defaultdict
from heapq import heappop, heappush
from math import ceil, cos, floor, hypot, radians

from map_matching import Fix

LAT_SCALE = 111_320
LON_SCALE = LAT_SCALE * cos(radians(55.75))
MERGE_METERS = 35
STEP_METERS = 12


def build_network(traces: dict[str, list[Fix]]) -> dict:
    anchors: list[tuple[float, float, float, float]] = []
    observations: list[list[float]] = []
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    adjacency: dict[int, set[int]] = defaultdict(set)
    edges: dict[tuple[int, int], set[str]] = defaultdict(set)

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def length(a, b):
        return hypot(anchors[a][0] - anchors[b][0], anchors[a][1] - anchors[b][1])

    def node(x, y, dx, dy):
        cx, cy = floor(x / MERGE_METERS), floor(y / MERGE_METERS)
        best, best_distance = None, MERGE_METERS
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for index in cells.get((cx + ox, cy + oy), ()):
                    nx, ny, ndx, ndy = anchors[index]
                    distance = hypot(x - nx, y - ny)
                    # Opposite directions share a corridor, perpendicular roads do not.
                    if distance < best_distance and (abs(dx * ndx + dy * ndy) >= 0.7 or distance < 7):
                        best, best_distance = index, distance
        if best is None:
            best = len(anchors)
            anchors.append((x, y, dx, dy))
            observations.append([0.0, 0.0, 0])
            cells[cx, cy].append(best)
        observation = observations[best]
        observation[0] += x
        observation[1] += y
        observation[2] += 1
        return best

    def existing_path(a, b, limit):
        queue = [(0.0, a)]
        distances, parents = {a: 0.0}, {}
        while queue:
            distance, current = heappop(queue)
            if distance > distances[current]:
                continue
            if current == b:
                path = []
                while current != a:
                    previous = parents[current]
                    path.append(key(previous, current))
                    current = previous
                return path
            for neighbor in adjacency[current]:
                candidate = distance + length(current, neighbor)
                if candidate <= limit and candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    parents[neighbor] = current
                    heappush(queue, (candidate, neighbor))
        return None

    def connect(a, b, vehicle_id):
        if a == b:
            return
        edge = key(a, b)
        if edge in edges:
            edges[edge].add(vehicle_id)
            return
        # Reuse the existing corridor instead of adding a chord on every pass.
        path = existing_path(a, b, length(a, b) * 1.65 + 10)
        if path:
            for step in path:
                edges[step].add(vehicle_id)
        else:
            edges[edge].add(vehicle_id)
            adjacency[a].add(b)
            adjacency[b].add(a)

    for vehicle_id, fixes in sorted(traces.items()):
        previous = None
        for first, second in zip(fixes, fixes[1:]):
            if second.time_s - first.time_s > 180:
                previous = None
                continue
            ax, ay = first.lon * LON_SCALE, first.lat * LAT_SCALE
            bx, by = second.lon * LON_SCALE, second.lat * LAT_SCALE
            distance = hypot(bx - ax, by - ay)
            if distance < 2:
                continue
            dx, dy = (bx - ax) / distance, (by - ay) / distance
            steps = max(1, ceil(distance / STEP_METERS))
            for i in range(steps + 1):
                fraction = i / steps
                current = node(ax + (bx - ax) * fraction, ay + (by - ay) * fraction, dx, dy)
                if previous is not None:
                    connect(previous, current, vehicle_id)
                previous = current

    coordinates = [[round(y / count / LAT_SCALE, 6), round(x / count / LON_SCALE, 6)]
                   for x, y, count in observations]
    return {"mergeMeters": MERGE_METERS, "nodes": coordinates,
            "edges": [[a, b, sorted(owners)] for (a, b), owners in sorted(edges.items())]}
