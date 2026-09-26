"""Group-safe probability calibration for delay greater than 120 seconds.

Run: python -m mos_trans.modeling.calibrated_probability
The saved ensemble returns probabilities, without a classification threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from mos_trans.modeling.classifier import CATEGORICAL, FEATURES, family_map, model_input


DELAY_THRESHOLD_S = 120.0
METHODS = ("raw", "sigmoid", "isotonic")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _fit_calibrator(method: str, raw: np.ndarray, actual: np.ndarray):
    if method == "sigmoid":
        return LogisticRegression(C=1.0, max_iter=1000).fit(_logit(raw), actual)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, actual)
    return None


def _apply(method: str, calibrator, raw: np.ndarray) -> np.ndarray:
    if method == "sigmoid":
        return calibrator.predict_proba(_logit(raw))[:, 1]
    if method == "isotonic":
        return calibrator.predict(raw)
    return raw


def _model(seed: int, trees: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss", iterations=trees, learning_rate=0.03,
        depth=6, l2_leaf_reg=8, random_seed=seed, thread_count=4,
        allow_writing_files=False, verbose=False,
    )


def _ensemble_fit_predict(
    x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, real: np.ndarray,
    fit_idx: np.ndarray, prediction_x: pd.DataFrame, trees: int, folds: int,
):
    """Each base model is calibrated only on unseen, real vehicle families."""
    split = GroupKFold(n_splits=folds)
    predictions = {method: [] for method in METHODS}
    members = []
    for fold, (train_pos, cal_pos) in enumerate(split.split(fit_idx, groups=groups[fit_idx])):
        train_idx, cal_idx = fit_idx[train_pos], fit_idx[cal_pos]
        cal_idx = cal_idx[real[cal_idx]]
        if len(cal_idx) == 0 or len(np.unique(y[cal_idx])) != 2:
            raise ValueError(f"Calibration fold {fold} has no real examples of both classes")
        model = _model(fold + 42, trees)
        model.fit(x.iloc[train_idx], y[train_idx], cat_features=list(CATEGORICAL))
        raw_cal = model.predict_proba(x.iloc[cal_idx])[:, 1]
        raw_pred = model.predict_proba(prediction_x)[:, 1]
        calibrators = {method: _fit_calibrator(method, raw_cal, y[cal_idx])
                       for method in METHODS}
        for method in METHODS:
            predictions[method].append(_apply(method, calibrators[method], raw_pred))
        members.append({"model": model, "calibrators": calibrators,
                        "calibration_n": int(len(cal_idx)),
                        "calibration_positives": int(y[cal_idx].sum()),
                        "calibration_families": sorted(set(groups[cal_idx].tolist()))})
    return {method: np.mean(predictions[method], axis=0) for method in METHODS}, members


def _scores(actual: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": int(len(actual)), "positives": int(actual.sum()),
        "brier": float(brier_score_loss(actual, p)),
        "log_loss": float(log_loss(actual, np.clip(p, 1e-8, 1 - 1e-8), labels=[0, 1])),
        "mean_shown": float(np.mean(p)), "actual_rate": float(np.mean(actual)),
        "roc_auc": float(roc_auc_score(actual, p)),
    }


def _reliability_table(actual: np.ndarray, p: np.ndarray,
                       groups: np.ndarray) -> pd.DataFrame:
    bins = pd.cut(p, bins=np.linspace(0, 1, 6), include_lowest=True)
    return pd.DataFrame({"bin": bins, "probability": p, "actual": actual,
                         "family": groups}).groupby(
        "bin", observed=False
    ).agg(n=("actual", "size"), families=("family", "nunique"),
          mean_shown=("probability", "mean"),
          actual_rate=("actual", "mean")).reset_index().astype({"bin": str})


def notebook_feature_baseline(labeled: pd.DataFrame, y: np.ndarray,
                              groups: np.ndarray, real: np.ndarray,
                              trees: int) -> dict:
    """Use the user's six notebook features with the same group evaluation."""
    now = pd.to_datetime(labeled["T"])
    planned = pd.to_datetime(labeled["target_time_begin"])
    x = pd.DataFrame({
        "cur_dev_s": labeled.cur_dev_s.fillna(0),
        "planned_duration_s": (planned - now).dt.total_seconds(),
        "hour": now.dt.hour, "minute": now.dt.minute,
        "dayofweek": now.dt.dayofweek,
        "target_stop_id": labeled.target_stop_id.astype(str),
    })
    predictions = np.full(len(labeled), np.nan)
    for fold, (fit_idx, hold_idx) in enumerate(GroupKFold(n_splits=4).split(x, groups=groups)):
        hold_idx = hold_idx[real[hold_idx]]
        model = _model(fold + 42, trees)
        model.fit(x.iloc[fit_idx], y[fit_idx], cat_features=["target_stop_id"])
        predictions[hold_idx] = model.predict_proba(x.iloc[hold_idx])[:, 1]
    return _scores(y[real], predictions[real])


def predict(bundle: dict, frame: pd.DataFrame, method: str | None = None) -> np.ndarray:
    method = method or bundle["selected_method"]
    if method not in METHODS:
        raise ValueError(f"Unknown calibration method: {method}")
    x = model_input(frame, bundle["features"])
    values = []
    for member in bundle["members"]:
        raw = member["model"].predict_proba(x)[:, 1]
        values.append(_apply(method, member["calibrators"][method], raw))
    return np.mean(values, axis=0)


def run(input_dir: str | Path = "data/processed",
        dataset_zip: str | Path = "data/dataset.zip",
        output_dir: str | Path = "artifacts/calibrated_probability",
        trees: int = 250) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    train = pd.read_parquet(input_dir / "train_features.parquet")
    test = pd.read_parquet(input_dir / "test_features.parquet")
    validate = pd.read_parquet(input_dir / "validate_features.parquet")
    labeled = pd.concat([train, test], ignore_index=True)
    if validate.target_delay_s.notna().any():
        raise ValueError("Validate contains target values")
    if labeled.target_delay_s.isna().any():
        raise ValueError("Labeled rows contain missing target values")
    columns = list(FEATURES)
    x = model_input(labeled, columns)
    model_input(validate, columns)
    y = (labeled.target_delay_s.to_numpy(dtype=float) > DELAY_THRESHOLD_S).astype(int)
    vehicle = labeled.tr_id.astype(str).to_numpy()
    real = ~np.char.startswith(vehicle.astype(str), "900")
    families = family_map(dataset_zip, set(vehicle[real]))
    groups = np.asarray([families[v] for v in vehicle])
    outer = GroupKFold(n_splits=4)
    results = {method: np.full(len(labeled), np.nan) for method in METHODS}
    for fold, (fit_idx, hold_idx) in enumerate(outer.split(x, groups=groups)):
        hold_real = hold_idx[real[hold_idx]]
        fold_predictions, _ = _ensemble_fit_predict(
            x, y, groups, real, fit_idx, x.iloc[hold_real], trees, folds=3,
        )
        for method in METHODS:
            results[method][hold_real] = fold_predictions[method]
        print(f"outer fold {fold + 1}/4: {len(hold_real)} real points, "
              f"{len(set(groups[hold_real]))} families", flush=True)
    scored = np.flatnonzero(real)
    report = {method: _scores(y[scored], results[method][scored]) for method in METHODS}
    baseline = notebook_feature_baseline(labeled, y, groups, real, trees)
    # Brier score directly rewards accurate displayed probabilities.
    selected = min(METHODS, key=lambda method: report[method]["brier"])
    all_idx = np.arange(len(labeled))
    _, members = _ensemble_fit_predict(
        x, y, groups, real, all_idx, model_input(validate, columns), trees, folds=4,
    )
    bundle = {"selected_method": selected, "members": members, "features": columns,
              "delay_threshold_s": DELAY_THRESHOLD_S}
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "model.joblib")
    held = pd.DataFrame({"sample_id": labeled.sample_id.iloc[scored].astype(str).to_numpy(),
                         "family": groups[scored], "actual": y[scored]})
    for method in METHODS:
        held[method] = results[method][scored]
        _reliability_table(y[scored], results[method][scored], groups[scored]).to_csv(
            output_dir / f"reliability_{method}.csv", index=False)
    held.to_csv(output_dir / "group_holdout_predictions.csv", index=False)
    p_validate = predict(bundle, validate)
    pd.DataFrame({"sample_id": validate.sample_id.astype(str),
                  "probability_delay_over_120s": p_validate}).to_csv(
        output_dir / "validate_probabilities.csv", index=False)
    summary = {"event": "actual delay > 120 seconds at target stop",
               "target_population": "real vehicles; synthetic relatives held with parent family",
               "evaluation": "nested 4-fold outer / 3-fold inner GroupKFold; no outer family in model or calibrator",
               "selection_metric": "Brier score on pooled outer real-vehicle predictions",
               "selected_method": selected, "scores": report,
               "notebook_feature_baseline": baseline,
               "n_real_families": int(len(set(groups[real]))),
               "limitations": "One day and 13 real vehicle families; estimates for 80%+ bins may be sparse. "
                              "Method choice on outer predictions makes selected score mildly optimistic."}
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed")
    parser.add_argument("--dataset", default="data/dataset.zip")
    parser.add_argument("--output", default="artifacts/calibrated_probability")
    parser.add_argument("--trees", type=int, default=250)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.dataset, args.output, args.trees),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
