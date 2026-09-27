"""Causal schedule matching for a configured NDTP vehicle."""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta

from mos_trans.stop_passages import near_interval


def _projection(lat: float, lon: float, start: dict, end: dict) -> tuple[float, float]:
    cos_lat = math.cos(math.radians(lat))
    ax = (start["lon"] - lon) * 111_320 * cos_lat
    ay = (start["lat"] - lat) * 111_320
    dx = (end["lon"] - start["lon"]) * 111_320 * cos_lat
    dy = (end["lat"] - start["lat"]) * 111_320
    length2 = dx * dx + dy * dy
    fraction = max(0.0, min(1.0, -(ax * dx + ay * dy) / length2)) if length2 else 0.0
    return math.hypot(ax + fraction * dx, ay + fraction * dy), fraction


class LiveRoute:
    def __init__(self, plan: dict, unit_id: int):
        self.plan = plan
        self.unit_id = unit_id
        self.trips = []
        if not plan.get("route") or not plan.get("trips"):
            raise ValueError("Route name and at least one scheduled trip are required")
        for trip in plan["trips"]:
            stops = [{**stop, "instant": datetime.fromisoformat(stop["time"])}
                     for stop in trip["stops"]]
            if len(stops) < 2 or any(b["instant"] < a["instant"] for a, b in zip(stops, stops[1:])):
                raise ValueError("Trip stops must have nondecreasing planned times")
            self.trips.append({**trip, "stops": stops})
        self.history = deque(maxlen=2000)
        self.stop_events: list[dict] = []
        self.open_visits: dict[tuple[str, str], dict] = {}
        self.observed_stops: set[tuple[str, str]] = set()
        self.deviation_history: list[tuple[datetime, float]] = []
        self.prediction: dict | None = None
        self.stopped_since: datetime | None = None
        self.last_result: dict | None = None

    def process(self, packet) -> dict:
        if packet.lat is None or packet.lon is None:
            self.last_result = {"forecastAvailability": "bad_gps"}
            return self.last_result
        event_time = packet.event_time
        candidates = []
        for trip in self.trips:
            stops = trip["stops"]
            # Ignore layovers before departure and after the final stop.
            if not stops[0]["instant"] - timedelta(minutes=2) <= event_time <= stops[-1]["instant"] + timedelta(minutes=3):
                continue
            for index, (start, end) in enumerate(zip(stops, stops[1:])):
                distance, fraction = _projection(packet.lat, packet.lon, start, end)
                planned = start["instant"] + (end["instant"] - start["instant"]) * fraction
                time_gap = abs((event_time - planned).total_seconds())
                # Use time as a soft tie-breaker at overlapping termini.
                score = distance + min(time_gap, 600) * 0.12
                candidates.append((score, distance, time_gap, trip, index, fraction, planned))
        self.history.append(packet)
        window = [item for item in self.history
                  if 0 <= (event_time - item.event_time).total_seconds() <= 300]
        mean_speed = sum(item.speed for item in window) / len(window) if window else None
        if packet.speed <= 2:
            self.stopped_since = self.stopped_since or event_time
        else:
            self.stopped_since = None
        stopped = max(0, (event_time - self.stopped_since).total_seconds()) if self.stopped_since else 0
        common = {"meanSpeed5m": mean_speed, "stoppedDurationSeconds": stopped,
                  "tripId": None, "route": self.plan["route"]}
        if not candidates:
            self.last_result = {**common, "forecastAvailability": "no_target"}
            return self.last_result
        _, distance, time_gap, trip, index, fraction, planned = min(candidates, key=lambda row: row[0])
        if distance > 300 or time_gap > 8 * 60:
            self.last_result = {**common, "forecastAvailability": "off_route"}
            return self.last_result
        stops = trip["stops"]
        projected_delay = (event_time - planned).total_seconds()
        next_stop = stops[index + 1]
        previous = self.history[-2] if len(self.history) > 1 else None
        if previous is not None and previous.lat is not None and previous.lon is not None:
            gap = (event_time - previous.event_time).total_seconds()
            jump = math.hypot((packet.lon - previous.lon) * 111_320 * math.cos(math.radians(packet.lat)),
                              (packet.lat - previous.lat) * 111_320)
            if 1 <= gap <= 45 and jump <= 280:
                a = {"time": previous.event_time, "lon": previous.lon, "lat": previous.lat}
                b = {"time": event_time, "lon": packet.lon, "lat": packet.lat}
                for stop in stops:
                    key = (trip["id"], stop["id"])
                    interval = near_interval(a, b, stop)
                    visit = self.open_visits.get(key)
                    if interval is not None:
                        entry, exit_, minimum = interval
                        if visit is None:
                            self.open_visits[key] = {"entry": entry, "exit": exit_, "distance": minimum}
                        else:
                            visit["exit"] = exit_
                            visit["distance"] = min(visit["distance"], minimum)
                    elif visit is not None:
                        self.open_visits.pop(key)
                        if key not in self.observed_stops:
                            midpoint = visit["entry"] + (visit["exit"] - visit["entry"]) / 2
                            self.stop_events.append({"key": key, "trip": trip["id"], "stopId": stop["id"],
                                "name": stop["name"], "time": midpoint.isoformat(), "planned": stop["time"],
                                "entryTime": visit["entry"].isoformat(), "exitTime": visit["exit"].isoformat(),
                                "delaySeconds": (midpoint - stop["instant"]).total_seconds()})
                            self.observed_stops.add(key)
        observed = next((event for event in reversed(self.stop_events) if event["trip"] == trip["id"]
                         and datetime.fromisoformat(event["time"]) <= event_time), None)
        current_delay = observed["delaySeconds"] if observed is not None else projected_delay
        target = next((stop for stop in stops[index + 1:]
                       if 600 < (stop["instant"] - event_time).total_seconds() <= 900), None)
        self.deviation_history.append((event_time, current_delay))
        self.deviation_history = self.deviation_history[-120:]
        self.last_result = {
            **common, "tripId": trip["id"], "direction": trip["direction"],
            "stops": [{key: value for key, value in stop.items() if key != "instant"} for stop in stops],
            "nextStop": {key: value for key, value in next_stop.items() if key != "instant"},
            "currentDeviationSeconds": current_delay,
            "estimateSeconds": current_delay if target else None,
            "forecastTime": target["time"] if target else None,
            "forecastStop": ({key: value for key, value in target.items() if key != "instant"}
                             if target else None),
            "forecastStopId": target["id"] if target else None,
            "forecastAvailability": "ready" if target else "no_target",
            "forecastMethod": "schedule_projection" if target else None,
            "distanceToRouteMeters": round(distance, 1),
            "stopEvents": self.stop_events[-20:],
            "reason": ("Отклонение по пройденной остановке" if observed is not None
                       else "Оценка по положению между остановками"),
            "recommendation": "Сравнить с фактическим прибытием на остановку",
        }
        if (target is not None and self.prediction is not None
                and self.prediction["forecastStopId"] == target["id"]
                and self.prediction["tripId"] == trip["id"]):
            self.last_result.update(self.prediction)
        return self.last_result
