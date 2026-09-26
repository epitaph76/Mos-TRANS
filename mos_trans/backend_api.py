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


class Replay:
    def __init__(self, dataset: Path, features_path: Path):
        self.rows = pd.read_parquet(features_path).drop(columns=list(FORBIDDEN), errors="ignore")
        self.rows["T"] = pd.to_datetime(self.rows["T"])
        self.rows["target_time_begin"] = pd.to_datetime(self.rows["target_time_begin"])
        self.rows_by_id = {str(row.sample_id): row for _, row in self.rows.iterrows()}
        self.ordered = self.rows.sort_values("T")
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
            nearby = [stop for stop in stops if abs((_time(stop["time"]) - at).total_seconds()) <= 2700][:10]
            next_stop = next((stop for stop in stops if _time(stop["time"]) >= at), None)
            previous_stop = next((stop for stop in reversed(stops) if _time(stop["time"]) <= at), None)
            section = (f"{stop_label(previous_stop)} → {stop_label(next_stop)}"
                       if previous_stop is not None and next_stop is not None else None)
            row = active_by_vehicle.get(vehicle)
            prediction = self.predictions.get(str(row.sample_id)) if row is not None else None
            cause, action = explanation(row) if row is not None and prediction else (None, None)
            target_stop = next((stop for stop in stops if row is not None and stop["id"] == str(row.target_stop_id)), None)
            vehicles.append({
                "id": vehicle, "position": [packet.lat, packet.lon],
                "speed": packet.speed if packet.speed is not None else 0,
                "heading": packet.heading, "gpsAgeMin": gps_age / 60,
                "stale": gps_age > 120,
                "stops": nearby, "nextStop": next_stop,
                "estimateSeconds": prediction["predicted_delay_s"] if prediction else None,
                "probability": prediction["probability_delay_over_120s"] if prediction else None,
                "forecastTime": row.target_time_begin.isoformat() if row is not None else None,
                "forecastStopId": str(row.target_stop_id) if row is not None else None,
                "forecastStop": target_stop,
                "sampleId": str(row.sample_id) if row is not None else None,
                "reason": cause, "recommendation": action,
                "section": section if row is not None else None,
                "source": "Исторический NDTP",
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
