"""Build the released model's feature contract from already received NDTP fixes."""

from __future__ import annotations

import pandas as pd

from mos_trans.features.delay_context import add_delay_trend_features
from mos_trans.features.route_sections import add_route_signatures, attach_planned_sections, build_route_segments
from mos_trans.preprocessing.core import Telemetry, TrafficIndex, _features


def live_feature_frame(route, packet) -> pd.DataFrame | None:
    result = route.last_result
    if not result or result.get("forecastAvailability") != "ready":
        return None
    trip = next((trip for trip in route.trips if trip["id"] == result["tripId"]), None)
    target = result["forecastStop"]
    if trip is None or target is None:
        return None
    vehicle = str(route.unit_id)
    now = packet.event_time
    target_time = pd.Timestamp(target["time"]).to_pydatetime()
    sample = {"sample_id": f"live:{vehicle}:{int(now.timestamp())}:{trip['id']}:{target['id']}",
              "tr_id": vehicle, "T": now, "target_stop_id": str(target["id"]),
              "target_time_begin": target_time, "cur_dev_s": result["currentDeviationSeconds"]}
    telemetry = [Telemetry(packet_id=f"{vehicle}:{index}", tr_id=vehicle, unit_id=vehicle,
                 event_time=fix.event_time, receive_time=fix.event_time,
                 location_valid=fix.lon is not None and fix.lat is not None,
                 gps_valid=fix.lon is not None and fix.lat is not None,
                 lon=fix.lon, lat=fix.lat, speed=float(fix.speed) if 0 <= fix.speed <= 100 else None,
                 speed_outlier=fix.speed > 100, heading=float(fix.heading), is_hist_data=False)
                 for index, fix in enumerate(route.history) if fix.event_time <= now]
    stop = {"lon": target["lon"], "lat": target["lat"], "address": target["name"]}
    planned_times = [item["instant"] for item in trip["stops"]]
    row = _features(sample, stop, planned_times, TrafficIndex(telemetry))
    schedule = pd.DataFrame([{"tr_id": vehicle, "tt_action_item_id": str(item["id"]),
        "time_begin": pd.Timestamp(item["instant"]).tz_convert("UTC"),
        "stop_lon": item["lon"], "stop_lat": item["lat"],
        "building_address": item["name"]} for item in trip["stops"]])
    schedule = add_route_signatures(schedule)
    features = attach_planned_sections(pd.DataFrame([row]), build_route_segments(schedule))
    history = pd.DataFrame([{"tr_id": vehicle, "T": instant, "cur_dev_s": deviation}
                            for instant, deviation in route.deviation_history[-3:]])
    trend = add_delay_trend_features(history).iloc[-1]
    for name in ("previous_cur_dev_s", "previous_delay_age_s", "previous_delay_available",
                 "delay_change_since_previous_s", "delay_change_per_min", "cur_dev_lag_2",
                 "lag_2_age_s", "lag_2_available", "delay_change_lag_2_s", "delay_change_lag_2_per_min"):
        features[name] = trend[name]
    return features
