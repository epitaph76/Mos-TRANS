"""Evaluate the existing train-only regressor on causally built test packets."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from mos_trans.modeling.catboost import prepare_categorical_columns
from mos_trans.stream_forecast import build_stream_features


DEFAULT_MODEL = Path("artifacts/route_delay_trend_submission/catboost_residual.cbm")


def run(dataset: Path, output: Path, model_path: Path = DEFAULT_MODEL) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    stream = build_stream_features(dataset, output, split="test")
    features = pd.read_parquet(output / "stream_features.parquet")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    matrix = prepare_categorical_columns(
        features, ["tr_id", "route_signature", "section_id", "schedule_target_address"])
    prediction = features.cur_dev_s.to_numpy(float) + model.predict(
        matrix[model.feature_names_])
    with zipfile.ZipFile(dataset) as archive:
        schedule = pd.read_csv(archive.open("test/schedule.csv"),
                               dtype={"tt_action_item_id": str})
    schedule["time_begin"] = pd.to_datetime(schedule.time_begin)
    schedule["time_fact_begin"] = pd.to_datetime(schedule.time_fact_begin)
    schedule["actual_delay_s"] = (schedule.time_fact_begin - schedule.time_begin).dt.total_seconds()
    scored = features[["sample_id", "packet_id", "tr_id", "T", "target_stop_id",
                       "cur_dev_s"]].copy()
    scored["predicted_delay_s"] = prediction
    scored = scored.merge(schedule[["tt_action_item_id", "actual_delay_s"]],
                          left_on="target_stop_id", right_on="tt_action_item_id",
                          validate="many_to_one")
    scored = scored.dropna(subset=["actual_delay_s"])
    scored["absolute_error_s"] = (scored.predicted_delay_s - scored.actual_delay_s).abs()
    last_per_stop = scored.sort_values("T").drop_duplicates("target_stop_id", keep="last")

    with zipfile.ZipFile(dataset) as archive:
        points = pd.read_csv(archive.open("labels/labels_test.csv"), dtype={"sample_id": str,
                                                                    "tr_id": str})
    points["T"] = pd.to_datetime(points["T"]).astype("datetime64[ns]")
    estimated = features[["tr_id", "T", "cur_dev_s"]].rename(
        columns={"T": "packet_T", "cur_dev_s": "estimated_cur_dev_s"})
    estimated["packet_T"] = estimated.packet_T.astype("datetime64[ns]")
    joined = pd.merge_asof(points.sort_values("T"), estimated.sort_values("packet_T"),
                           by="tr_id", left_on="T", right_on="packet_T",
                           direction="backward", tolerance=pd.Timedelta(seconds=120))
    known = joined.dropna(subset=["estimated_cur_dev_s"])
    deviation_error = (known.estimated_cur_dev_s - known.cur_dev_s).abs()
    report = {
        "model": str(model_path), "split": "test", "stream": stream,
        "packet_predictions_with_fact": int(len(scored)),
        "packet_mae_s": float(scored.absolute_error_s.mean()),
        "target_stops_with_fact": int(len(last_per_stop)),
        "last_prediction_per_target_mae_s": float(last_per_stop.absolute_error_s.mean()),
        "published_points_with_estimated_deviation": int(len(known)),
        "published_points_total": int(len(points)),
        "estimated_deviation_mae_s": float(deviation_error.mean()),
        "estimated_deviation_median_absolute_error_s": float(deviation_error.median()),
        "note": "The regressor was trained without test rows. Facts are used only for this report. "
                "Repeated packet forecasts for one stop are correlated, and the released "
                "model was trained with supplied rather than GPS-estimated cur_dev_s.",
    }
    (output / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/stream-test"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output, args.model),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
