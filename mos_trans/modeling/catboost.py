from pathlib import Path
import argparse
import json
import zipfile

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mos_trans.modeling.weights import (build_sample_weights)

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
    "target_stop_id",
    "section_from_time",
    "section_to_time"
}

CATEGORICAL_COLUMNS = [
    "tr_id",
    "route_signature",
    "section_id",
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

def prepare_categorical_columns(frame: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in categorical_columns:
        frame[column] = (frame[column].fillna("__MISSING__").astype(str))

    return frame


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
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

def create_model(iterations: int=3000) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=iterations,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=8.0,
        random_seed=42,
        allow_writing_files=False,
        verbose=100,
    )

def split_train_by_time(frame: pd.DataFrame, validation_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = (frame.sort_values(["tr_id", "T"]).reset_index(drop=True))

    fit_parts = []
    validation_parts = []

    for _, vehicle_rows in frame.groupby("tr_id", sort=False):
        row_count = len(vehicle_rows)
        split_index = int(row_count * (1.0 - validation_fraction))
        split_index = max(1, min(split_index, row_count - 1))

        fit_parts.append(vehicle_rows.iloc[:split_index])
        validation_parts.append(vehicle_rows.iloc[split_index:])

    fit_train = pd.concat(fit_parts, ignore_index=True)
    early_stop_validation = pd.concat(validation_parts, ignore_index=True)

    return fit_train, early_stop_validation

def train_model(fit_train: pd.DataFrame, early_stop_validation: pd.DataFrame,
                feature_columns: list[str], categorical_columns: list[str],
                synthetic_weight:float) -> CatBoostRegressor:
    model = create_model()

    sample_weights = build_sample_weights(fit_train, synthetic_weight=synthetic_weight)

    model.fit(
        fit_train[feature_columns],
        fit_train[RESIDUAL_TARGET_COLUMN],
        cat_features=categorical_columns,
        sample_weight=sample_weights,
        eval_set=(early_stop_validation[feature_columns], early_stop_validation[RESIDUAL_TARGET_COLUMN]),
        early_stopping_rounds=150,
        use_best_model=True,
    )

    return model

def train_final_model(train: pd.DataFrame, test: pd.DataFrame,
                      feature_columns: list[str], categorical_columns: list[str],
                      best_iteration: int) -> CatBoostRegressor:
    full_train = pd.concat([train, test], ignore_index=True)
    final_iterations = max(1, best_iteration + 1)

    print("\nFinal training:")
    print("full_train rows:", len(full_train))
    print("iterations:", final_iterations)

    final_model = create_model(iterations=final_iterations)
    final_model.fit(full_train[feature_columns], full_train[RESIDUAL_TARGET_COLUMN], cat_features=categorical_columns)

    return final_model


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

def load_submission_template(dataset_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(dataset_path) as archive:
        candidates = [name for name in archive.namelist()
                      if name.endswith("sample_submission.csv")]

        if len(candidates) != 1:
            raise ValueError("Не удалось однозначно найти ""sample_submission.csv в архиве. "f"Найдено: {candidates}")

        with archive.open(candidates[0]) as file:
            template = pd.read_csv(file, sep=";", dtype={"sample_id": str})

    expected_columns = [
        "sample_id",
        "prediction",
    ]

    if template.columns.tolist() != expected_columns:
        raise ValueError("Неверные столбцы sample_submission.csv: "f"{template.columns.tolist()}")

    return template

def save_submission(template: pd.DataFrame, validate: pd.DataFrame,
                    predicted_delay: np.ndarray, output_path: Path):
    predictions = pd.DataFrame(
        {
            "sample_id": (
                validate[ID_COLUMN]
                .astype(str)
            ),
            "prediction": predicted_delay,
        }
    )

    if predictions["sample_id"].duplicated().any():
        duplicated = predictions.loc[
            predictions["sample_id"].duplicated(),
            "sample_id",
        ].tolist()

        raise ValueError(
            "В прогнозах повторяются sample_id: "
            f"{duplicated[:10]}"
        )

    template_ids = set(
        template["sample_id"].astype(str)
    )

    prediction_ids = set(
        predictions["sample_id"]
    )

    missing_ids = template_ids - prediction_ids
    extra_ids = prediction_ids - template_ids

    if missing_ids:
        raise ValueError(
            "Для некоторых sample_id нет прогноза: "
            f"{sorted(missing_ids)[:10]}"
        )

    if extra_ids:
        raise ValueError(
            "Обнаружены лишние sample_id: "
            f"{sorted(extra_ids)[:10]}"
        )

    submission = (
        template[["sample_id"]]
        .merge(
            predictions,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    )

    submission["prediction"] = pd.to_numeric(
        submission["prediction"],
        errors="coerce",
    )

    if submission["prediction"].isna().any():
        raise ValueError(
            "В submission есть пустые "
            "или нечисловые prediction"
        )

    if not np.isfinite(
        submission["prediction"].to_numpy()
    ).all():
        raise ValueError(
            "В submission есть inf или -inf"
        )

    if len(submission) != len(template):
        raise ValueError(
            "Количество строк изменилось "
            "после объединения с шаблоном"
        )

    submission.to_csv(
        output_path,
        sep=";",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.6f",
    )

    print("\nSubmission saved:")
    print(output_path)
    print("rows:", len(submission))
    print(
        "prediction min:",
        submission["prediction"].min(),
    )
    print(
        "prediction max:",
        submission["prediction"].max(),
    )
    print(
        "prediction mean:",
        submission["prediction"].mean(),
    )

def run(input_dir: Path, output_dir: Path, dataset_path: Path,
        make_submission: bool, synthetic_weight:float):
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test, validate = load_datasets(input_dir)

    train = add_residual_target(train)
    test = add_residual_target(test)

    feature_columns = select_feature_columns(train)
    categorical_columns = select_categorical_columns(train, feature_columns)

    train = prepare_categorical_columns(train, categorical_columns)
    test = prepare_categorical_columns(test, categorical_columns)
    validate = prepare_categorical_columns(validate, categorical_columns)

    fit_train, early_stop_validation = (split_train_by_time(train))
    print("Split sizes:")
    print("fit_train:", fit_train.shape)
    print("early_stop_validation:", early_stop_validation.shape)
    print("test:", test.shape)
    print("Feature columns:")

    for column in feature_columns:
        print(" -", column)

    print("Categorical columns:", categorical_columns)

    baseline_median = median_delta_baseline(train, test)

    model = train_model(fit_train, early_stop_validation, feature_columns, categorical_columns,
                        synthetic_weight)

    test_delta_prediction, test_delay_prediction = (
        predict_final_delay(
            model,
            test,
            feature_columns,
        )
    )

    validation_delta_prediction, validation_delay_prediction = (
        predict_final_delay(
            model,
            early_stop_validation,
            feature_columns,
        )
    )

    validation_metrics = regression_metrics(actual_delay=early_stop_validation[TARGET_COLUMN],
                                            predicted_delay=validation_delay_prediction,
                                            actual_delta=early_stop_validation[RESIDUAL_TARGET_COLUMN],
                                            predicted_delta=validation_delta_prediction,
    )

    model_metrics = regression_metrics(
        actual_delay=test[TARGET_COLUMN],
        predicted_delay=test_delay_prediction,
        actual_delta=test[RESIDUAL_TARGET_COLUMN],
        predicted_delta=test_delta_prediction,
    )

    metrics = {
        "target": (
            "target_delay_s - cur_dev_s"
        ),
        "rows": {
            "full_train": len(train),
            "fit_train": len(fit_train),
            "early_stop_validation": len(early_stop_validation),
            "test": len(test),
            "validate": len(validate),
        },
        "feature_count": len(feature_columns),
        "categorical_columns": categorical_columns,
        "synthetic_weights": synthetic_weight,
        "best_iteration": model.get_best_iteration(),
        "baselines": {
            "median_train_delta": baseline_median,
        },
        "early_stop_validation": validation_metrics,
        "test": model_metrics,
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

    model.save_model(str(output_dir / "catboost_residual.cbm"))

    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    worst_predictions = (test_predictions.sort_values("absolute_error_s", ascending=False).head(50))
    print("\nWorst predictions:")
    print(worst_predictions.head(20).to_string(index=False))
    worst_predictions.to_csv(output_dir / "worst_predictions.csv", index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    save_feature_importance(model, feature_columns, output_dir)

    if make_submission:
        best_iteration = model.get_best_iteration()

        final_model = train_final_model(train=train, test=test, feature_columns=feature_columns,
                                        categorical_columns=categorical_columns, best_iteration=best_iteration)
        validate_delta_prediction, validate_delay_prediction = (predict_final_delay(final_model, validate, feature_columns))

        submission_template = load_submission_template(dataset_path)
        save_submission(template=submission_template, validate=validate,
                        predicted_delay=validate_delay_prediction, output_path=output_dir / "submission.csv")

        final_model.save_model(str(output_dir / "catboost_residual_final.cbm"))
        validate_diagnostics = pd.DataFrame(
            {
                "sample_id": validate[ID_COLUMN],
                "cur_dev_s": validate[
                    CURRENT_DELAY_COLUMN
                ],
                "predicted_delta_s": (
                    validate_delta_prediction
                ),
                "prediction": (
                    validate_delay_prediction
                ),
            }
        )

        validate_diagnostics.to_csv(
            output_dir
            / "validate_predictions_diagnostics.csv",
            index=False,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/catboost_base"))
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--make-submission", action="store_true")
    parser.add_argument("--synthetic-weight", type=float,default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.input, args.output, args.dataset,
        args.make_submission, args.synthetic_weight)


if __name__ == "__main__":
    main()
