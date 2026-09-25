from pathlib import Path
import argparse
import json
import zipfile

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ID_COLUMN = "sample_id"
CURRENT_DELAY_COLUMN = "cur_dev_s"
TARGET_COLUMN = "target_delay_s"
RESIDUAL_TARGET_COLUMN = "target_delta_s"

EXCLUDED_COLUMNS = {
    "sample_id",
    "T",
    "target_time_begin",
    "target_delay_s",
    "target_delta_s",
    "target_class",
    "time_fact_begin",
}

CATEGORICAL_COLUMNS = [
    "target_stop_id",
    "schedule_target_address",
]

def load_datasets(input_dir: Path):
    train = pd.read_parquet(input_dir / "train_features.parquet")
    test = pd.read_parquet(input_dir / "test_features.parquet")
    validate = pd.read_parquet(input_dir / "validate_features.parquet")

    return train, test, validate

def add_residual_target(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame[RESIDUAL_TARGET_COLUMN] = (frame[TARGET_COLUMN] - frame[CURRENT_DELAY_COLUMN])

    return frame

def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    if "time_fact_begin" in frame or "target_class" in frame:
        raise ValueError("Factual arrival time or target class cannot be a feature")
    return [column for column in frame.columns
            if column not in EXCLUDED_COLUMNS]

def select_categorical_columns(frame: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [column for column in CATEGORICAL_COLUMNS
            if column in frame.columns and column in feature_columns]

def median_delta_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    median_delta = float(train[RESIDUAL_TARGET_COLUMN].median())
    prediction = (test[CURRENT_DELAY_COLUMN] + median_delta)

    return{
        "median_delta_s": median_delta,
        "mae": float(mean_absolute_error(test[TARGET_COLUMN], prediction)),
        "rmse": float(np.sqrt(mean_squared_error(test[TARGET_COLUMN], prediction))),
    }

def create_model(iterations: int = 3000, verbose: int | bool = 100) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=iterations,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=8.0,
        random_seed=42,
        allow_writing_files=False,
        verbose=verbose,
    )

def train_model(train: pd.DataFrame, test: pd.DataFrame,
                feature_columns: list[str], categorical_columns: list[str]) -> CatBoostRegressor:
    model = create_model()
    model.fit(
        train[feature_columns],
        train[RESIDUAL_TARGET_COLUMN],
        cat_features=categorical_columns,
        eval_set=(test[feature_columns], test[RESIDUAL_TARGET_COLUMN]),
        early_stopping_rounds=150,
        use_best_model=True,
    )

    return model

def predict_final_delay(model: CatBoostRegressor, frame: pd.DataFrame, 
                        feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    predicted_delta = model.predict(frame[feature_columns])
    predicted_delay = (frame[CURRENT_DELAY_COLUMN].to_numpy() + predicted_delta)

    return predicted_delta, predicted_delay

def regression_metrics(actual_delay: pd.Series, predicted_delay: np.ndarray, 
                       actual_delta: pd.Series, predicted_delta: np.ndarray) -> dict:
    errors = (actual_delay.to_numpy() - predicted_delay)

    return {
        "mae_delay_s": float(mean_absolute_error(actual_delay, predicted_delay)),
        "rmse_delay_s": float(np.sqrt(mean_squared_error(actual_delay, predicted_delay))),
        "mae_delta_s": float(mean_absolute_error(actual_delta, predicted_delta)),
        "mean_error_s": float(errors.mean()),
        "median_absolute_error_s": float(np.median(np.abs(errors))),
        "p90_absolute_error_s": float(np.quantile(np.abs(errors), 0.90)),
        "predicted_delta_mean_s": float(np.mean(predicted_delta)),
    }

def save_feature_importance(model:CatBoostRegressor, feature_columns:list[str],
                            output_dir: Path):
    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.get_feature_importance(),
    })

    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)


def run(input_dir: Path, output_dir: Path, dataset_zip: Path | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test, validate = load_datasets(input_dir)
    if (not train[TARGET_COLUMN].notna().all() or not test[TARGET_COLUMN].notna().all()
            or validate[TARGET_COLUMN].notna().any()):
        raise ValueError("Expected labeled train/test and unlabeled validate")

    train = add_residual_target(train)
    test = add_residual_target(test)

    feature_columns = select_feature_columns(train)

    categorical_columns = select_categorical_columns(train, feature_columns)
    
    print("Feature columns:")
    for column in feature_columns:
        print(" -", column)

    print("Categorical columns:", categorical_columns)

    baseline_median = median_delta_baseline(train, test)

    model = train_model(train, test, feature_columns, categorical_columns)
    selection_model_path = output_dir / "catboost_residual_local_test.cbm"
    model.save_model(str(selection_model_path))

    test_delta_prediction, test_delay_prediction = (
        predict_final_delay(
            model,
            test,
            feature_columns,
        )
    )

    model_metrics = regression_metrics(
        actual_delay=test[TARGET_COLUMN],
        predicted_delay=test_delay_prediction,
        actual_delta=test[RESIDUAL_TARGET_COLUMN],
        predicted_delta=test_delta_prediction,
    )

    best_iteration = model.get_best_iteration()
    selected_trees = max(1, int(best_iteration) + 1)
    final = create_model(iterations=selected_trees, verbose=False)
    final_training = pd.concat([train, test], ignore_index=True)
    final.fit(final_training[feature_columns], final_training[RESIDUAL_TARGET_COLUMN],
              cat_features=categorical_columns)
    validate_delta, validate_delay = predict_final_delay(final, validate, feature_columns)
    if not np.isfinite(validate_delay).all():
        raise ValueError("Non-finite validate predictions")

    metrics = {
        "target": (
            "target_delay_s - cur_dev_s"
        ),
        "rows": {
            "train": len(train),
            "test": len(test),
            "validate": len(validate),
        },
        "feature_count": len(feature_columns),
        "categorical_columns": categorical_columns,
        "best_iteration": best_iteration,
        "selected_trees": selected_trees,
        "local_test_model_training_splits": ["train"],
        "final_model_training_splits": ["train", "test"],
        "final_model_training_rows": int(len(final_training)),
        "test_metric_note": "Test was used for early stopping, so its MAE is a tuning metric, not an unbiased final estimate.",
        "baselines": {
            "median_train_delta": baseline_median,
        },
        "catboost": model_metrics,
    }

    test_predictions = pd.DataFrame(
        {
            "sample_id": test[ID_COLUMN],
            "cur_dev_s": test[CURRENT_DELAY_COLUMN],
            "target_delay_s": test[TARGET_COLUMN],
            "actual_delta_s": test[RESIDUAL_TARGET_COLUMN],
            "predicted_delta_s": test_delta_prediction,
            "prediction": test_delay_prediction,
        }
    )

    test_predictions["error_s"] = (test_predictions["target_delay_s"] - test_predictions["prediction"])
    test_predictions["absolute_error_s"] = (test_predictions["error_s"].abs())

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    final.save_model(str(output_dir / "catboost_residual.cbm"))

    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    worst_predictions = (test_predictions.sort_values("absolute_error_s", ascending=False).head(50))
    print("\nWorst predictions:")
    print(worst_predictions.head(20).to_string(index=False))
    worst_predictions.to_csv(output_dir / "worst_predictions.csv", index=False)
    
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    save_feature_importance(final, feature_columns, output_dir)

    validate_predictions = pd.DataFrame({"sample_id": validate[ID_COLUMN].astype(str),
                                         "cur_dev_s": validate[CURRENT_DELAY_COLUMN],
                                         "predicted_delta_s": validate_delta,
                                         "prediction": validate_delay})
    validate_predictions.to_csv(output_dir / "validate_predictions.csv", index=False)
    submission = validate_predictions[["sample_id", "prediction"]].copy()
    if dataset_zip is not None:
        with zipfile.ZipFile(dataset_zip) as archive:
            with archive.open("sample_submission.csv") as stream:
                template = pd.read_csv(stream, sep=";", dtype={"sample_id": str})
        if submission.sample_id.duplicated().any() or template.sample_id.duplicated().any():
            raise ValueError("Duplicate sample_id in validate or submission template")
        if set(submission.sample_id) != set(template.sample_id):
            raise ValueError("Validate sample IDs do not match submission template")
        submission = template[["sample_id"]].merge(submission, on="sample_id", validate="one_to_one")
    submission.to_csv(output_dir / "submission.csv", sep=";", index=False)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/catboost_base"))
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.zip"))
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.input, args.output, args.dataset)


if __name__ == "__main__":
    main()
