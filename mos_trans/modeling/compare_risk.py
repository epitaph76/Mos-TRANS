"""Compare causal movement features and probabilistic ETA on 13 vehicle families."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score

from mos_trans.modeling.classifier import CATEGORICAL, FEATURES as BASE_FEATURES, family_map, labels, model_input
from mos_trans.modeling.progress import FEATURES as PROGRESS_FEATURES, prepare
from mos_trans.preprocessing import build_dataset


QUANTILES = (0.1, 0.5, 0.9)
CANDIDATES = ("catboost_base", "catboost_progress", "lightgbm_base", "lightgbm_progress", "lightgbm_quantiles")


@dataclass(frozen=True)
class CompareConfig:
    catboost_trees: int = 93
    lightgbm_trees: int = 200
    depth: int = 6
    lightgbm_leaves: int = 15
    lightgbm_min_child: int = 40
    learning_rate: float = 0.03
    seed: int = 42
    threads: int = 4


def _catboost(config: CompareConfig) -> CatBoostClassifier:
    return CatBoostClassifier(loss_function="Logloss", iterations=config.catboost_trees,
                              learning_rate=config.learning_rate, depth=config.depth,
                              l2_leaf_reg=8, random_seed=config.seed,
                              thread_count=config.threads, allow_writing_files=False, verbose=False)


def _lightgbm(config: CompareConfig, *, quantile: float | None = None):
    arguments = dict(n_estimators=config.lightgbm_trees, learning_rate=config.learning_rate,
                     num_leaves=config.lightgbm_leaves, max_depth=5,
                     min_child_samples=config.lightgbm_min_child,
                     reg_lambda=8.0, random_state=config.seed,
                     n_jobs=config.threads, verbosity=-1)
    if quantile is None:
        return lgb.LGBMClassifier(objective="binary", **arguments)
    return lgb.LGBMRegressor(objective="quantile", alpha=quantile, **arguments)


def _lightgbm_frames(fit: pd.DataFrame, other: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn categorical vocabularies on fit only; unseen categories become missing."""
    fit_x, other_x = fit.loc[:, columns].copy(), other.loc[:, columns].copy()
    for name in ("tr_id", *CATEGORICAL):
        categories = pd.Index(fit_x[name].astype(str).unique())
        fit_x[name] = pd.Categorical(fit_x[name].astype(str), categories=categories)
        other_values = other_x[name].astype(str)
        other_x[name] = pd.Categorical(other_values.where(other_values.isin(categories)), categories=categories)
    for name in set(columns) - {"tr_id", *CATEGORICAL}:
        fit_x[name] = pd.to_numeric(fit_x[name], errors="raise").astype(np.float32)
        other_x[name] = pd.to_numeric(other_x[name], errors="raise").astype(np.float32)
    return fit_x, other_x


def quantile_risk(quantiles: np.ndarray, current_delay: np.ndarray, threshold_s: float = 150) -> np.ndarray:
    """Approximate exceedance probability by interpolating the estimated delay CDF."""
    if quantiles.ndim != 2 or quantiles.shape[1] != 3 or len(quantiles) != len(current_delay):
        raise ValueError("Expected [n,3] quantiles and aligned current delays")
    q10, median, q90 = np.maximum.accumulate(quantiles + current_delay[:, None], axis=1).T
    lower_span = np.maximum(median - q10, 1.0)
    upper_span = np.maximum(q90 - median, 1.0)
    cdf = np.where(threshold_s <= median,
                   0.5 + 0.4 * (threshold_s - median) / lower_span,
                   0.5 + 0.4 * (threshold_s - median) / upper_span)
    return np.clip(1 - cdf, 0.001, 0.999)


def _score(actual: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = probability >= threshold
    return {"n": int(len(actual)), "positives": int(actual.sum()),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "ap": float(average_precision_score(actual, probability)) if actual.any() else None,
            "auc": float(roc_auc_score(actual, probability)) if len(np.unique(actual)) == 2 else None,
            "logloss": float(log_loss(actual, probability, labels=[0, 1]))}


def _threshold(oof: pd.DataFrame, column: str) -> tuple[float, float]:
    thresholds = np.round(np.arange(0.05, 0.951, 0.025), 3)
    actual = oof.actual_class.to_numpy(int)
    probability = oof[column].to_numpy(float)
    scores = [f1_score(actual, probability >= value, zero_division=0) for value in thresholds]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def _fit_predict(name: str, fit: pd.DataFrame, other: pd.DataFrame,
                 config: CompareConfig) -> np.ndarray:
    return _predict_fitted(_fit_model(name, fit, config), other)


def _fit_model(name: str, fit: pd.DataFrame, config: CompareConfig) -> dict:
    if name == "catboost_lightgbm_blend":
        return {"name": name, "members": [_fit_model("catboost_progress", fit, config),
                                           _fit_model("lightgbm_progress", fit, config)]}
    columns = list(BASE_FEATURES) + (list(PROGRESS_FEATURES) if name.endswith("progress") or name == "lightgbm_quantiles" else [])
    if name.startswith("catboost"):
        # Same numeric vehicle-ID treatment and feature order as the current classifier.
        fit_x = model_input(fit, list(BASE_FEATURES))
        if name.endswith("progress"):
            fit_x = pd.concat([fit_x, fit.loc[:, list(PROGRESS_FEATURES)]], axis=1)
        model = _catboost(config)
        model.fit(fit_x, labels(fit.target_delay_s), cat_features=list(CATEGORICAL))
        return {"name": name, "model": model}
    fit_x, _ = _lightgbm_frames(fit, fit, columns)
    categories = {column: list(fit_x[column].cat.categories) for column in ("tr_id", *CATEGORICAL)}
    if name == "lightgbm_quantiles":
        residual = (fit.target_delay_s - fit.cur_dev_s).to_numpy(float)
        models = []
        for alpha in QUANTILES:
            model = _lightgbm(config, quantile=alpha)
            model.fit(fit_x, residual, categorical_feature=["tr_id", *CATEGORICAL])
            models.append(model)
        return {"name": name, "models": models, "categories": categories}
    model = _lightgbm(config)
    model.fit(fit_x, labels(fit.target_delay_s), categorical_feature=["tr_id", *CATEGORICAL])
    return {"name": name, "model": model, "categories": categories}


def _predict_fitted(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    if "time_fact_begin" in frame:
        raise ValueError("Factual schedule arrival time is forbidden in prediction features")
    name = artifact["name"]
    if name == "catboost_lightgbm_blend":
        return np.mean([_predict_fitted(member, frame) for member in artifact["members"]], axis=0)
    if name.startswith("catboost"):
        matrix = model_input(frame, list(BASE_FEATURES))
        if name.endswith("progress"):
            matrix = pd.concat([matrix, frame.loc[:, list(PROGRESS_FEATURES)]], axis=1)
        return artifact["model"].predict_proba(matrix)[:, 1]
    columns = list(BASE_FEATURES) + (list(PROGRESS_FEATURES) if name.endswith("progress") or name == "lightgbm_quantiles" else [])
    matrix = frame.loc[:, columns].copy()
    for column, categories in artifact["categories"].items():
        values = matrix[column].astype(str)
        matrix[column] = pd.Categorical(values.where(values.isin(categories)), categories=categories)
    for column in set(columns) - set(artifact["categories"]):
        matrix[column] = pd.to_numeric(matrix[column], errors="raise").astype(np.float32)
    if name == "lightgbm_quantiles":
        estimates = np.column_stack([model.predict(matrix) for model in artifact["models"]])
        return quantile_risk(estimates, frame.cur_dev_s.to_numpy(float))
    return artifact["model"].predict_proba(matrix)[:, 1]


def predict(model_path: str | Path, features: pd.DataFrame) -> np.ndarray:
    """Return P(target_delay_s > 150) from an augmented, point-in-time feature frame."""
    return _predict_fitted(joblib.load(model_path), features)


def run(dataset_zip: str | Path, processed_dir: str | Path, cache_dir: str | Path,
        output_dir: str | Path, config: CompareConfig = CompareConfig(),
        evaluate_test: bool = True) -> dict:
    dataset_zip, processed_dir, cache_dir, output_dir = map(Path, (dataset_zip, processed_dir, cache_dir, output_dir))
    if not (processed_dir / "train_features.parquet").is_file():
        build_dataset(dataset_zip, processed_dir)
    paths = prepare(dataset_zip, processed_dir, cache_dir)
    train = pd.read_parquet(paths["train"])
    test = pd.read_parquet(paths["test"])
    validate = pd.read_parquet(paths["validate"])
    families = family_map(dataset_zip, {str(value) for value in train.tr_id if not str(value).startswith("900")})
    family_id = train.tr_id.astype(str).map(families)
    real_ids = sorted(set(family_id[~train.tr_id.astype(str).str.startswith("900")]))
    if len(real_ids) != 13:
        raise ValueError(f"Expected 13 original vehicle families, got {len(real_ids)}")
    oof_rows = []
    for held in real_ids:
        fit = train.loc[family_id != held]
        held_frame = train.loc[(family_id == held) & (train.tr_id.astype(str) == held)]
        row = pd.DataFrame({"sample_id": held_frame.sample_id.astype(str).to_numpy(),
                            "tr_id": held, "actual_class": labels(held_frame.target_delay_s)})
        for name in CANDIDATES:
            row[name] = _fit_predict(name, fit, held_frame, config)
        oof_rows.append(row)
        print(f"Completed family {held} ({len(held_frame)} real points)", flush=True)
    oof = pd.concat(oof_rows, ignore_index=True)
    oof["catboost_lightgbm_blend"] = (oof.catboost_progress + oof.lightgbm_progress) / 2
    names = list(CANDIDATES) + ["catboost_lightgbm_blend"]
    comparison = {}
    for name in names:
        threshold, selected_f1 = _threshold(oof, name)
        comparison[name] = {"oof_threshold": threshold, "oof_selected_f1": selected_f1,
                            **_score(oof.actual_class.to_numpy(int), oof[name].to_numpy(float), threshold)}
    # AP is threshold-free; test is inspected only after the candidate is selected.
    chosen = max(names, key=lambda name: comparison[name]["ap"])
    chosen_threshold = comparison[chosen]["oof_threshold"]
    by_family = {str(vehicle): _score(group.actual_class.to_numpy(int),
                                      group[chosen].to_numpy(float), chosen_threshold)
                 for vehicle, group in oof.groupby("tr_id", sort=True)}
    output_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False, lineterminator="\n")
    test_report = None
    if evaluate_test:
        artifact = _fit_model(chosen, train, config)
        joblib.dump(artifact, output_dir / "selected_model.joblib")
        test_probability = _predict_fitted(artifact, test)
        validate_probability = _predict_fitted(artifact, validate)
        threshold = chosen_threshold
        test_report = _score(labels(test.target_delay_s), test_probability, threshold)
        pd.DataFrame({"sample_id": test.sample_id.astype(str), "tr_id": test.tr_id.astype(str),
                      "actual_class": labels(test.target_delay_s), "probability_delay_over_150s": test_probability,
                      "predicted_class": (test_probability >= threshold).astype(int)}).to_csv(
                          output_dir / "test_predictions.csv", index=False, lineterminator="\n")
        pd.DataFrame({"sample_id": validate.sample_id.astype(str),
                      "probability_delay_over_150s": validate_probability,
                      "predicted_class": (validate_probability >= threshold).astype(int)}).to_csv(
                          output_dir / "validate_predictions.csv", index=False, lineterminator="\n")
    with dataset_zip.open("rb") as stream:
        dataset_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"dataset_sha256": dataset_hash, "config": asdict(config), "features": list(PROGRESS_FEATURES),
              "families": real_ids, "oof_n_real": int(len(oof)), "comparison": comparison,
              "selected_by_oof_ap": chosen, "oof_by_family": by_family, "test": test_report,
              "limits": "One day and 13 original trajectories; OOF model selection/threshold tuning make selected OOF F1 optimistic. Published test shares day and vehicles with train."}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--cache", type=Path, default=Path("data/progress-cache"))
    parser.add_argument("--output", type=Path, default=Path("data/risk-comparison"))
    args = parser.parse_args()
    report = run(args.dataset, args.processed, args.cache, args.output)
    print(json.dumps({"comparison": report["comparison"], "selected": report["selected_by_oof_ap"],
                      "test": report["test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
