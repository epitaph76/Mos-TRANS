"""Small grouped Optuna search for LightGBM risk classification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold

from mos_trans.modeling.classifier import CATEGORICAL, FEATURES, family_map, labels
from mos_trans.modeling.compare_risk import _lightgbm_frames


COLUMNS = [column for column in FEATURES if column != "tr_id"]


def _fit_predict(fit: pd.DataFrame, other: pd.DataFrame, params: dict) -> np.ndarray:
    fit_x, other_x = _lightgbm_frames(fit, other, COLUMNS)
    model = lgb.LGBMClassifier(objective="binary", random_state=42, n_jobs=4,
                               verbosity=-1, **params)
    model.fit(fit_x, labels(fit.target_delay_s), categorical_feature=list(CATEGORICAL))
    return model.predict_proba(other_x)[:, 1]


def _metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict:
    actual = labels(frame.target_delay_s)
    return {"n": int(len(actual)), "positives": int(actual.sum()),
            "ap": float(average_precision_score(actual, probability)),
            "f1_at_0_30": float(f1_score(actual, probability >= .30)),
            "inner_selected_threshold": threshold,
            "f1_at_inner_threshold": float(f1_score(actual, probability >= threshold))}


def _threshold(oof: pd.DataFrame) -> float:
    actual = oof.actual.to_numpy(int)
    probability = oof.probability.to_numpy(float)
    grid = np.round(np.arange(.05, .951, .025), 3)
    return float(max(grid, key=lambda value: f1_score(actual, probability >= value)))


def run(dataset_zip: str | Path, processed_dir: str | Path, output_dir: str | Path,
        trials: int = 25, seed: int = 42) -> dict:
    dataset_zip, processed_dir, output_dir = map(Path, (dataset_zip, processed_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(processed_dir / "train_features.parquet")
    real = ~data.tr_id.astype(str).str.startswith("900")
    real_ids = set(data.loc[real, "tr_id"].astype(str))
    family = data.tr_id.astype(str).map(family_map(dataset_zip, real_ids))
    outer_vehicles = sorted(np.random.default_rng(seed).choice(sorted(real_ids), size=3, replace=False).tolist())
    inner_data = data.loc[~family.isin(outer_vehicles)].copy()
    inner_family = family.loc[inner_data.index]
    outer_data = data.loc[family.isin(outer_vehicles) & real].copy()
    folds = []
    for fit_indices, valid_indices in GroupKFold(n_splits=5).split(inner_data, groups=inner_family):
        fit = inner_data.iloc[fit_indices]
        valid = inner_data.iloc[valid_indices]
        valid = valid.loc[~valid.tr_id.astype(str).str.startswith("900")]
        folds.append((fit, valid))

    def cross_validate(params: dict) -> tuple[float, pd.DataFrame]:
        predictions = []
        for fit, valid in folds:
            probabilities = _fit_predict(fit, valid, params)
            predictions.append(pd.DataFrame({"actual": labels(valid.target_delay_s),
                                             "probability": probabilities}))
        oof = pd.concat(predictions, ignore_index=True)
        return float(average_precision_score(oof.actual, oof.probability)), oof

    baseline = dict(n_estimators=200, learning_rate=.03, num_leaves=15, max_depth=5,
                    min_child_samples=40, reg_lambda=8)
    baseline_inner_ap, baseline_inner_oof = cross_validate(baseline)

    def objective(trial: optuna.Trial) -> float:
        params = dict(n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
                      learning_rate=trial.suggest_float("learning_rate", .02, .06),
                      num_leaves=trial.suggest_int("num_leaves", 7, 31, step=4),
                      max_depth=5,
                      min_child_samples=trial.suggest_int("min_child_samples", 20, 100, step=20),
                      reg_lambda=trial.suggest_float("reg_lambda", 2, 20, log=True),
                      colsample_bytree=trial.suggest_float("colsample_bytree", .7, 1, step=.1),
                      subsample=trial.suggest_float("subsample", .7, 1, step=.1),
                      subsample_freq=1)
        score, _ = cross_validate(params)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    tuned = {**study.best_params, "max_depth": 5, "subsample_freq": 1}
    _, tuned_inner_oof = cross_validate(tuned)
    baseline_threshold = _threshold(baseline_inner_oof)
    tuned_threshold = _threshold(tuned_inner_oof)
    baseline_outer = _fit_predict(inner_data, outer_data, baseline)
    tuned_outer = _fit_predict(inner_data, outer_data, tuned)
    predictions = pd.DataFrame({"sample_id": outer_data.sample_id.astype(str),
                                "tr_id": outer_data.tr_id.astype(str),
                                "actual": labels(outer_data.target_delay_s),
                                "baseline_probability": baseline_outer,
                                "tuned_probability": tuned_outer})
    predictions.to_csv(output_dir / "outer_predictions.csv", index=False)
    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)
    with dataset_zip.open("rb") as stream:
        dataset_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"dataset_sha256": dataset_hash, "seed": seed, "outer_vehicles": outer_vehicles,
              "inner_folds": 5, "trials": trials,
              "baseline_params": baseline, "best_params": tuned,
              "inner_baseline_ap": baseline_inner_ap, "inner_tuned_ap": float(study.best_value),
              "outer_baseline": _metrics(outer_data, baseline_outer, baseline_threshold),
              "outer_tuned": _metrics(outer_data, tuned_outer, tuned_threshold),
              "note": "Outer vehicles are held out of this search. The dataset and group scheme were explored in earlier experiments, so this is not a pristine independent estimate."}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/risk-tuning"))
    parser.add_argument("--trials", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.processed, args.output, args.trials), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
