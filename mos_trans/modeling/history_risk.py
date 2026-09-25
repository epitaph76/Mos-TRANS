"""Evaluate causal history of the currently reported schedule deviation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from mos_trans.modeling.classifier import CATEGORICAL, FEATURES, family_map, labels
from mos_trans.modeling.compare_risk import _lightgbm_frames


HISTORY_FEATURES = ("previous_dev_s", "dev_change_s", "previous_age_s", "dev_change_10m_s")
VARIANTS = {"baseline": (), "all_history": HISTORY_FEATURES,
            "change_5m": ("dev_change_s",), "change_10m": ("dev_change_10m_s",)}


def augment(frame: pd.DataFrame) -> pd.DataFrame:
    """Use only earlier observations of cur_dev_s from the same vehicle."""
    if "time_fact_begin" in frame or not {"tr_id", "T", "cur_dev_s"}.issubset(frame):
        raise ValueError("Missing point fields or forbidden factual schedule time")
    output = frame.reset_index(drop=True).copy()
    values = np.full((len(output), len(HISTORY_FEATURES)), np.nan)
    for _, group in output.groupby("tr_id", sort=False):
        ordered = group.sort_values("T", kind="stable")
        positions = ordered.index.to_numpy()
        times = pd.to_datetime(ordered["T"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        deviations = ordered.cur_dev_s.to_numpy(float)
        for index, now in enumerate(times):
            previous = int(np.searchsorted(times, now, side="left")) - 1
            if previous >= 0 and now - times[previous] <= 600 * 1_000_000_000:
                values[positions[index], :3] = (deviations[previous],
                                                 deviations[index] - deviations[previous],
                                                 (now - times[previous]) / 1e9)
            previous_10m = int(np.searchsorted(times, now - 600 * 1_000_000_000, side="right")) - 1
            if previous_10m >= 0 and now - times[previous_10m] <= 1200 * 1_000_000_000:
                values[positions[index], 3] = deviations[index] - deviations[previous_10m]
    output.loc[:, list(HISTORY_FEATURES)] = values
    return output


def run(dataset_zip: str | Path, processed_dir: str | Path, output_dir: str | Path) -> dict:
    dataset_zip, processed_dir, output_dir = map(Path, (dataset_zip, processed_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    data = augment(pd.read_parquet(processed_dir / "train_features.parquet"))
    real = ~data.tr_id.astype(str).str.startswith("900")
    family = data.tr_id.astype(str).map(family_map(dataset_zip, set(data.loc[real, "tr_id"].astype(str))))
    predictions = pd.DataFrame({"sample_id": data.loc[real, "sample_id"].astype(str).to_numpy(),
                                "tr_id": data.loc[real, "tr_id"].astype(str).to_numpy(),
                                "actual": labels(data.loc[real, "target_delay_s"])})
    scores = {}
    for name, extra in VARIANTS.items():
        probability = np.full(len(data), np.nan)
        columns = [column for column in FEATURES if column != "tr_id"] + list(extra)
        for held in sorted(set(family[real])):
            fit_mask = family != held
            valid_mask = (family == held) & real
            fit, valid = data.loc[fit_mask], data.loc[valid_mask]
            fit_x, valid_x = _lightgbm_frames(fit, valid, columns)
            model = lgb.LGBMClassifier(objective="binary", n_estimators=200, learning_rate=.03,
                                       num_leaves=15, max_depth=5, min_child_samples=40,
                                       reg_lambda=8, random_state=42, n_jobs=4, verbosity=-1)
            model.fit(fit_x, labels(fit.target_delay_s), categorical_feature=list(CATEGORICAL))
            probability[valid_mask] = model.predict_proba(valid_x)[:, 1]
        prediction = probability[real]
        actual = predictions.actual.to_numpy(int)
        thresholds = np.round(np.arange(.05, .951, .025), 3)
        threshold = float(max(thresholds, key=lambda value: f1_score(actual, prediction >= value)))
        predictions[name] = prediction
        scores[name] = {"ap": float(average_precision_score(actual, prediction)),
                        "f1": float(f1_score(actual, prediction >= threshold)),
                        "oof_selected_threshold": threshold}
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    report = {"n_real": int(real.sum()), "scores": scores,
              "note": "Lags are computed from prior sample points within train. Production use needs a stream of prior cur_dev_s observations."}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/risk-history"))
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.processed, args.output), indent=2))


if __name__ == "__main__":
    main()
