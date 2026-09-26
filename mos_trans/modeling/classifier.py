"""Predict whether the target-stop delay exceeds 150 seconds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score,
    roc_auc_score,
)


DELAY_THRESHOLD_S = 150.0
CLASS_NAMES = ("not_over_150s", "over_150s")
SEEDS = (42, 7, 123)
EXCLUDED = frozenset({"sample_id", "T", "target_time_begin", "target_delay_s", "target_delta_s", "target_class", "time_fact_begin"})
CATEGORICAL = ("target_stop_id", "schedule_target_address")
REQUIRED = frozenset({"sample_id", "tr_id", "cur_dev_s", *CATEGORICAL})
FEATURES = (
    "tr_id", "target_stop_id", "cur_dev_s", "realtime_data_age_s",
    "realtime_speed_last", "realtime_speed_age_s", "realtime_gps_age_s",
    "realtime_distance_to_target_m", "realtime_packet_count_2m",
    "realtime_speed_count_2m", "realtime_speed_mean_2m", "realtime_speed_std_2m",
    "realtime_stopped_fraction_2m", "realtime_packet_count_5m",
    "realtime_speed_count_5m", "realtime_speed_mean_5m", "realtime_speed_std_5m",
    "realtime_stopped_fraction_5m", "realtime_packet_count_10m",
    "realtime_speed_count_10m", "realtime_speed_mean_10m", "realtime_speed_std_10m",
    "realtime_stopped_fraction_10m", "context_vehicle_count_500m",
    "context_speed_count_500m", "context_speed_mean_500m",
    "context_speed_median_500m", "context_stopped_fraction_500m",
    "schedule_horizon_s", "schedule_target_lon", "schedule_target_lat",
    "schedule_target_address", "schedule_stops_remaining", "historical_hour",
    "historical_day_of_week", "historical_time_sin", "historical_time_cos",
)


@dataclass(frozen=True)
class Config:
    iterations: int = 1500
    learning_rate: float = 0.03
    depth: int = 6
    l2_leaf_reg: float = 8.0
    patience: int = 100
    thread_count: int = 4


def labels(delay_s: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(delay_s, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Classification labels require known finite delays")
    return (values > DELAY_THRESHOLD_S).astype(np.int64)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    if not REQUIRED.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(REQUIRED - set(frame.columns))}")
    if "time_fact_begin" in frame or "target_class" in frame:
        raise ValueError("Future or target class column in feature table")
    if set(FEATURES) - set(frame):
        raise ValueError(f"Missing features: {sorted(set(FEATURES) - set(frame))}")
    columns = [name for name in frame if name in FEATURES]
    if any(name not in (*CATEGORICAL, "tr_id") and not pd.api.types.is_numeric_dtype(frame[name]) for name in columns):
        raise ValueError("Unexpected non-numeric feature")
    return columns


def model_input(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not REQUIRED.issubset(frame.columns) or set(columns) != set(FEATURES) or set(columns) - set(frame):
        raise ValueError("Missing or forbidden model input")
    if "time_fact_begin" in frame or "target_class" in frame:
        raise ValueError("Future or target class column in feature table")
    result = frame.loc[:, columns].copy()
    for name in CATEGORICAL:
        if result[name].isna().any():
            raise ValueError(f"Missing categorical feature: {name}")
        result[name] = result[name].astype(str)
    result["tr_id"] = pd.to_numeric(result["tr_id"], errors="raise")
    return result


def family_map(dataset_zip: str | Path, real_ids: set[str]) -> dict[str, str]:
    """Find synthetic parents by the ordered planned stop geometry, never by labels."""
    with zipfile.ZipFile(dataset_zip) as archive:
        path = next((name for name in archive.namelist() if name.endswith("train/schedule.csv")), None)
        if path is None:
            raise ValueError("train/schedule.csv not found")
        with archive.open(path) as binary:
            rows = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig"))
            planned: dict[str, list[tuple[str, str]]] = {}
            for row in rows:
                planned.setdefault(str(row["tr_id"]), []).append((row["time_begin"], row["geom"]))
    signatures: dict[tuple[str, ...], str] = {}
    for vehicle in real_ids:
        if vehicle not in planned:
            raise ValueError(f"Real vehicle absent from schedule: {vehicle}")
        signature = tuple(geom for _, geom in sorted(planned[vehicle]))
        if signature in signatures:
            raise ValueError("Ambiguous real route signature")
        signatures[signature] = vehicle
    result = {vehicle: vehicle for vehicle in real_ids}
    for vehicle, stops in planned.items():
        if vehicle.startswith("900"):
            signature = tuple(geom for _, geom in sorted(stops))
            if signature not in signatures:
                raise ValueError(f"Synthetic vehicle lacks a parent: {vehicle}")
            result[vehicle] = signatures[signature]
    return result


def family_split(frame: pd.DataFrame, families: dict[str, str], seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vehicles = frame.tr_id.astype(str).to_numpy()
    real = sorted({vehicle for vehicle in vehicles if not vehicle.startswith("900")})
    if len(real) < 4:
        raise ValueError("At least four real vehicle families required")
    held = sorted(np.random.default_rng(seed).choice(real, size=3, replace=False).tolist())
    parents = np.asarray([families[vehicle] for vehicle in vehicles])
    training = np.flatnonzero(~np.isin(parents, held))
    validation = np.flatnonzero(np.isin(parents, held) & ~np.char.startswith(vehicles.astype(str), "900"))
    if not len(training) or not len(validation):
        raise ValueError("Empty train or validation split")
    return training, validation, held


def _model(config: Config, iterations: int, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss",
                              iterations=iterations, learning_rate=config.learning_rate,
                              depth=config.depth, l2_leaf_reg=config.l2_leaf_reg,
                              random_seed=seed, thread_count=config.thread_count,
                              allow_writing_files=False, verbose=False)


def _metrics(actual: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    positive = probabilities[:, 1]
    predicted = (positive >= threshold).astype(np.int64)
    return {
        "n": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, positive)) if len(np.unique(actual)) == 2 else None,
        "average_precision": float(average_precision_score(actual, positive)) if actual.any() else None,
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "class_counts": np.bincount(actual, minlength=2).tolist(),
    }


def predict(model_path: str | Path, metadata_path: str | Path, frame: pd.DataFrame) -> pd.DataFrame:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    probabilities = model.predict_proba(model_input(frame, metadata["features"]))
    positive = probabilities[:, 1]
    predicted = (positive >= float(metadata.get("decision_threshold", 0.5))).astype(np.int64)
    return pd.DataFrame({"sample_id": frame.sample_id.astype(str).to_numpy(),
                         "predicted_class": predicted,
                         "probability_delay_over_150s": positive})


def run(input_dir: str | Path, dataset_zip: str | Path, output_dir: str | Path,
        config: Config = Config()) -> dict:
    input_dir, dataset_zip, output_dir = Path(input_dir), Path(dataset_zip), Path(output_dir)
    train = pd.read_parquet(input_dir / "train_features.parquet")
    test = pd.read_parquet(input_dir / "test_features.parquet")
    validate = pd.read_parquet(input_dir / "validate_features.parquet")
    if validate.target_delay_s.notna().any():
        raise ValueError("Validate must remain unlabeled")
    columns = feature_columns(train)
    x_train = model_input(train, columns)
    x_test = model_input(test, columns)
    model_input(validate, columns)  # Check the exact inference schema before training.
    y_train, y_test = labels(train.target_delay_s), labels(test.target_delay_s)
    families = family_map(dataset_zip, {str(x) for x in train.tr_id if not str(x).startswith("900")})
    folds = []
    oof = []
    fold_probabilities = []
    for seed in SEEDS:
        fit_idx, val_idx, held = family_split(train, families, seed)
        model = _model(config, config.iterations, seed)
        model.fit(x_train.iloc[fit_idx], y_train[fit_idx], cat_features=list(CATEGORICAL),
                  eval_set=(x_train.iloc[val_idx], y_train[val_idx]),
                  early_stopping_rounds=config.patience, use_best_model=True)
        probability = model.predict_proba(x_train.iloc[val_idx])
        fold_probabilities.append((y_train[val_idx], probability))
        fold_metrics = _metrics(y_train[val_idx], probability)
        fold_metrics.update({"seed": seed, "held_out_vehicles": held,
                             "trees": int(model.tree_count_), "n_train": int(len(fit_idx))})
        folds.append(fold_metrics)
        fold_rows = pd.DataFrame({"sample_id": train.sample_id.iloc[val_idx].astype(str).to_numpy(),
                                  "tr_id": train.tr_id.iloc[val_idx].astype(str).to_numpy(),
                                  "seed": seed, "actual_class": y_train[val_idx]})
        fold_rows["probability_delay_over_150s"] = probability[:, 1]
        oof.append(fold_rows)
        print(f"seed={seed}: holdout AP={fold_metrics['average_precision']:.3f}, "
              f"trees={model.tree_count_}", flush=True)
    thresholds = np.round(np.arange(0.10, 0.901, 0.05), 2)
    threshold_scores = [np.mean([f1_score(actual, probability[:, 1] >= threshold, zero_division=0)
                                 for actual, probability in fold_probabilities]) for threshold in thresholds]
    decision_threshold = float(thresholds[int(np.argmax(threshold_scores))])
    for fold, (actual, probability), rows in zip(folds, fold_probabilities, oof):
        fold.update(_metrics(actual, probability, decision_threshold))
        rows["predicted_class"] = (rows["probability_delay_over_150s"] >= decision_threshold).astype(int)
    print(f"Decision threshold={decision_threshold:.2f}, mean holdout F1={max(threshold_scores):.3f}", flush=True)
    chosen_trees = int(np.median([fold["trees"] for fold in folds]))
    local_test_model = _model(config, chosen_trees, SEEDS[0])
    local_test_model.fit(x_train, y_train, cat_features=list(CATEGORICAL))
    output_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = output_dir / "catboost_classifier_local_test.cbm"
    local_test_model.save_model(str(local_model_path))
    test_probabilities = local_test_model.predict_proba(x_test)
    final = _model(config, chosen_trees, SEEDS[0])
    final.fit(pd.concat([x_train, x_test], ignore_index=True), np.r_[y_train, y_test],
              cat_features=list(CATEGORICAL))
    model_path = output_dir / "catboost_classifier.cbm"
    final.save_model(str(model_path))
    (output_dir / "holdout_predictions.csv").write_bytes(
        pd.concat(oof).to_csv(index=False, lineterminator="\n").encode("utf-8"))
    digest = hashlib.sha256(dataset_zip.read_bytes()).hexdigest()
    report = {
        "dataset_sha256": digest, "config": asdict(config), "delay_threshold_s": DELAY_THRESHOLD_S,
        "positive_definition": "target_delay_s > 150",
        "decision_threshold": decision_threshold,
        "class_names": list(CLASS_NAMES), "features": columns, "categorical": list(CATEGORICAL),
        "holdout": folds, "holdout_f1_mean": float(np.mean([fold["f1"] for fold in folds])),
        "holdout_average_precision_mean": float(np.mean([fold["average_precision"] for fold in folds])),
        "selected_trees": chosen_trees, "test": _metrics(y_test, test_probabilities, decision_threshold),
        "selection": "Trees and decision threshold selected from real-vehicle family holdouts; test scored using the train-only model, then included in final training.",
        "local_test_model_training_splits": ["train"],
        "final_model_training_splits": ["train", "test"],
        "final_model_training_rows": int(len(train) + len(test)),
        "limits": "Published test shares vehicles and day with train; classifier probabilities are not seconds of delay.",
    }
    metadata_path = output_dir / "metrics.json"
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    test_predictions = predict(local_model_path, metadata_path, test)
    test_predictions.insert(1, "actual_class", y_test)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False, lineterminator="\n")
    predict(model_path, metadata_path, validate).to_csv(
        output_dir / "validate_predictions.csv", index=False, lineterminator="\n")
    pd.DataFrame({"feature": columns, "importance": final.get_feature_importance()}).sort_values(
        "importance", ascending=False).to_csv(output_dir / "feature_importance.csv", index=False, lineterminator="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/catboost_classifier"))
    args = parser.parse_args()
    report = run(args.input, args.dataset, args.output)
    print(json.dumps({"holdout_f1_mean": report["holdout_f1_mean"],
                      "test": report["test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
