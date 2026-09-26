"""Dispatcher API: causal archive replay and live NDTP intake."""

from __future__ import annotations

import asyncio
import os
from bisect import bisect_right
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from mos_trans.features.build import build_feature_set
from mos_trans.demo import generate as generate_demo
from mos_trans.inference import FORBIDDEN, json_records
from mos_trans.ndtp import LiveStore
from mos_trans.preprocessing.core import DataSource, _point, _time, build_dataset, clean_traffic_row


def prepare_data(dataset: Path, processed: Path, routed: Path) -> None:
    if not (processed / "validate_features.parquet").exists():
        build_dataset(dataset, processed)
    if not (routed / "validate_features.parquet").exists():
        build_feature_set(processed, routed, dataset)


def explanation(row: pd.Series) -> tuple[str, str]:
    stopped = row.get("realtime_stopped_fraction_5m")
    speed = row.get("realtime_speed_mean_5m")
    neighbors = row.get("context_vehicle_count_500m")
    context_speed = row.get("context_speed_mean_500m")
    if pd.notna(stopped) and stopped >= 0.7 and pd.notna(speed) and speed <= 2:
        return "Длительная остановка", "Проверить посадку и связь с водителем"
    if pd.notna(neighbors) and neighbors >= 2 and pd.notna(context_speed) and context_speed < 10:
        return "Медленный соседний поток", "Оценить объезд и приоритет движения"
    if pd.notna(speed) and speed < 10:
        return "Снижение скорости", "Проверить участок и резервное ТС"
    if pd.notna(row.get("cur_dev_s")) and row["cur_dev_s"] > 120:
        return "Накопленное отставание от графика", "Оценить регулирование интервала и резервное ТС"
    return "Причина не определена", "Наблюдать за следующими пакетами"


def stop_label(stop: dict) -> str:
    return stop["name"] if stop["name"] != "Остановка" else f"ост. {stop['id']}"


def next_observed_stop(stops: list[dict], packet) -> tuple[dict | None, dict | None]:
    """Find the next stop from the received GPS fix on the demo corridor."""
    if len(stops) < 2:
        return None, None
    best = (float("inf"), 0, 0.0)
    scale = 0.56
    for index, (start, end) in enumerate(zip(stops, stops[1:])):
        dx = (end["lon"] - start["lon"]) * scale
        dy = end["lat"] - start["lat"]
        length = dx * dx + dy * dy
        fraction = max(0.0, min(1.0, (((packet.lon - start["lon"]) * scale * dx)
                                            + (packet.lat - start["lat"]) * dy) / length)) if length else 0.0
        distance = ((packet.lon - start["lon"]) * scale - fraction * dx) ** 2 + (
            packet.lat - start["lat"] - fraction * dy) ** 2
        if distance < best[0]:
            best = (distance, index, fraction)
    _, index, fraction = best
    next_index = index + (2 if fraction >= 0.999 else 1)
    return (stops[next_index - 1] if next_index > 0 else None,
            stops[next_index] if next_index < len(stops) else None)


class Replay:
    def __init__(self, dataset: Path, features_path: Path,
                 source_name: str = "Исторический NDTP", full_route: bool = False):
        self.source_name = source_name
        self.full_route = full_route
        self.rows = pd.read_parquet(features_path).drop(columns=list(FORBIDDEN), errors="ignore")
        self.rows["T"] = pd.to_datetime(self.rows["T"])
        self.rows["target_time_begin"] = pd.to_datetime(self.rows["target_time_begin"])
        self.rows_by_id = {str(row.sample_id): row for _, row in self.rows.iterrows()}
        self.ordered = self.rows.sort_values("T")
        self.row_times = {}
        self.vehicle_rows = {}
        for vehicle, group in self.ordered.groupby("tr_id"):
            self.row_times[str(vehicle)] = group["T"].tolist()
            self.vehicle_rows[str(vehicle)] = list(group.itertuples(index=False))
        self.predictions: dict[str, dict] = {}
        self.stops: dict[str, list[dict]] = defaultdict(list)
        self.gps: dict[str, tuple[list[datetime], list]] = {}
        source = DataSource(dataset)
        try:
            with source.csv_rows("validate/schedule_plan.csv") as schedule:
                for row in schedule:
                    lon, lat = _point(row["geom"])
                    self.stops[str(row["tr_id"])].append({
                        "id": str(row["tt_action_item_id"]), "time": row["time_begin"],
                        "lon": lon, "lat": lat, "name": row["building_address"].strip() or "Остановка",
                    })
            packets = defaultdict(list)
            with source.csv_rows("validate/traffic.csv") as traffic:
                for row in traffic:
                    packet = clean_traffic_row(row)
                    if packet.available_time is not None:
                        packets[packet.tr_id].append(packet)
        finally:
            source.close()
        for stops in self.stops.values():
            stops.sort(key=lambda stop: stop["time"])
        self.stop_times = {vehicle: [_time(stop["time"]) for stop in stops]
                           for vehicle, stops in self.stops.items()}
        for vehicle, group in packets.items():
            group.sort(key=lambda p: (p.available_time, p.event_time, p.packet_id))
            times, best_at = [], []
            best = None
            for packet in group:
                times.append(packet.available_time)
                if packet.gps_valid and (best is None or packet.event_time >= best.event_time):
                    best = packet
                best_at.append(best)
            self.gps[vehicle] = times, best_at
        self.day = self.ordered.iloc[0]["T"].date()

    def active_rows(self, at: datetime) -> pd.DataFrame:
        mask = (self.ordered["T"] <= at) & (self.ordered["T"] > at - timedelta(seconds=60))
        active = self.ordered.loc[mask]
        horizon = (active["target_time_begin"] - active["T"]).dt.total_seconds()
        return active.loc[(horizon > 600) & (horizon <= 900)]

    def position(self, vehicle: str, at: datetime):
        index = self.gps.get(vehicle)
        if not index:
            return None
        times, best_at = index
        i = bisect_right(times, at) - 1
        return best_at[i] if i >= 0 else None

    async def ensure_predictions(self, rows: pd.DataFrame, ml_url: str, client: httpx.AsyncClient) -> str:
        missing = rows.loc[~rows.sample_id.astype(str).isin(self.predictions)]
        if missing.empty:
            return "ready"
        payload = {"features": json_records(missing)}
        try:
            response = await client.post(f"{ml_url}/predict", json=payload, timeout=3.0)
            response.raise_for_status()
            for item in response.json()["predictions"]:
                self.predictions[str(item["sample_id"])] = item
            return "ready"
        except (httpx.HTTPError, KeyError, ValueError):
            return "unavailable"

    async def snapshot(self, at_seconds: float, ml_url: str, client: httpx.AsyncClient) -> dict:
        at = datetime.combine(self.day, datetime.min.time()) + timedelta(seconds=at_seconds)
        active = self.active_rows(at)
        model_status = await self.ensure_predictions(active, ml_url, client)
        active_by_vehicle = {str(row.tr_id): row for _, row in active.iterrows()}
        vehicles = []
        for vehicle, stops in self.stops.items():
            packet = self.position(vehicle, at)
            if packet is None:
                continue
            gps_age = max(0, (at - packet.event_time).total_seconds())
            stop_index = bisect_right(self.stop_times[vehicle], at)
            nearby = (stops if self.full_route else
                      stops[max(0, stop_index - 5):stop_index + 11])
            if self.full_route:
                previous_stop, next_stop = next_observed_stop(stops, packet)
            else:
                next_stop = stops[stop_index] if stop_index < len(stops) else None
                previous_stop = stops[stop_index - 1] if stop_index else None
            section = (f"{stop_label(previous_stop)} → {stop_label(next_stop)}"
                       if previous_stop is not None and next_stop is not None else None)
            row = active_by_vehicle.get(vehicle)
            prior_times = self.row_times.get(vehicle, [])
            prior_index = bisect_right(prior_times, at) - 1
            prior_row = (self.vehicle_rows[vehicle][prior_index]
                         if prior_index >= 0 and (at - prior_times[prior_index]).total_seconds() < 60
                         else None)
            prediction = self.predictions.get(str(row.sample_id)) if row is not None else None
            next_forecast_index = bisect_right(prior_times, at)
            nearest_point = (prior_times[next_forecast_index] if next_forecast_index < len(prior_times)
                             else prior_times[-1] if prior_times else None)
            point_direction = ("next" if next_forecast_index < len(prior_times)
                               else "previous" if prior_times else None)
            forecast_availability = ("ready" if prediction else
                                     "ml_unavailable" if row is not None and model_status == "unavailable" else
                                     "pending" if row is not None else "no_point")
            cause, action = explanation(row) if row is not None and prediction else (None, None)
            target_stop = next((stop for stop in stops if row is not None and stop["id"] == str(row.target_stop_id)), None)
            vehicles.append({
                "id": vehicle, "position": [packet.lat, packet.lon],
                "speed": packet.speed if packet.speed is not None else 0,
                "heading": packet.heading, "gpsAgeMin": gps_age / 60,
                "stale": gps_age > 120,
                "stops": nearby, "nextStop": next_stop,
                "estimateSeconds": prediction["predicted_delay_s"] if prediction else None,
                "currentDeviationSeconds": float(prior_row.cur_dev_s) if prior_row is not None else None,
                "probability": prediction["probability_delay_over_120s"] if prediction else None,
                "forecastTime": row.target_time_begin.isoformat() if row is not None else None,
                "forecastStopId": str(row.target_stop_id) if row is not None else None,
                "forecastStop": target_stop,
                "sampleId": str(row.sample_id) if row is not None else None,
                "forecastAvailability": forecast_availability,
                "nearestForecastPointAt": nearest_point.isoformat() if nearest_point is not None else None,
                "nearestForecastPointDirection": point_direction,
                "reason": cause, "recommendation": action,
                "section": section if row is not None else None,
                "source": self.source_name,
            })
        vehicles.sort(key=lambda item: (item["probability"] is None, -(item["probability"] or 0), item["id"]))
        return {"source": "historical-replay", "date": self.day.isoformat(),
                "at": at.isoformat(), "modelStatus": model_status,
                "activePoints": len(active), "vehicles": vehicles}


@asynccontextmanager
async def lifespan(app: FastAPI):
    dataset = Path(os.getenv("DATASET_PATH", "data/dataset.zip"))
    processed = Path(os.getenv("PROCESSED_PATH", "data/processed"))
    routed = Path(os.getenv("ROUTED_PATH", "data/processed_route"))
    prepare_data(dataset, processed, routed)
    app.state.replay = Replay(dataset, routed / "validate_features.parquet")
    demo_archive, demo_features = generate_demo(Path(os.getenv("DEMO_PATH", "data/demo")))
    app.state.demo = Replay(demo_archive, demo_features, "Синтетический учебный рейс", full_route=True)
    app.state.live = LiveStore()
    app.state.client = httpx.AsyncClient()
    app.state.ml_url = os.getenv("ML_URL", "http://127.0.0.1:8001").rstrip("/")
    server = await asyncio.start_server(app.state.live.handle, host="0.0.0.0",
                                        port=int(os.getenv("NDTP_PORT", "9201")))
    app.state.ndtp_server = server
    try:
        yield
    finally:
        server.close()
        await server.wait_closed()
        await app.state.client.aclose()


app = FastAPI(title="Mos-TRANS Dispatcher", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ready", "ndtpConnections": app.state.live.connections,
            "ndtpFrames": app.state.live.frames, "ndtpErrors": app.state.live.errors}


@app.get("/api/replay/snapshot")
async def replay_snapshot(at: float = Query(ge=0, le=172800)) -> dict:
    try:
        return await app.state.replay.snapshot(at, app.state.ml_url, app.state.client)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/live/snapshot")
def live_snapshot() -> dict:
    return app.state.live.snapshot()


@app.get("/api/demo/snapshot")
async def demo_snapshot(at: float = Query(ge=0, le=172800)) -> dict:
    return await app.state.demo.snapshot(at, app.state.ml_url, app.state.client)
