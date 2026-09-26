"""The shared, point-in-time prediction contract for batch and API inference."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from mos_trans.modeling.calibrated_probability import predict as predict_probability
from mos_trans.modeling.catboost import prepare_categorical_columns


FORBIDDEN = {"target_delay_s", "target_delta_s", "time_fact_begin", "target_class"}
DEFAULT_REGRESSOR = Path("artifacts/route_delay_trend_submission/catboost_residual_final.cbm")
DEFAULT_PROBABILITY = Path("artifacts/calibrated_probability/model.joblib")


class Predictor:
    """Load both released models once and return aligned predictions."""

    def __init__(self, regressor_path: str | Path = DEFAULT_REGRESSOR,
                 probability_path: str | Path = DEFAULT_PROBABILITY):
        self.regressor = CatBoostRegressor()
        self.regressor.load_model(str(regressor_path))
        self.regressor_features = list(self.regressor.feature_names_)
        self.probability_bundle = joblib.load(probability_path)
        self.probability_features = list(self.probability_bundle["features"])
        if float(self.probability_bundle["delay_threshold_s"]) != 120:
            raise ValueError("Probability artifact must predict delay > 120 seconds")

    @property
    def input_columns(self) -> list[str]:
        return list(dict.fromkeys([
            "sample_id", "T", "target_time_begin", "target_stop_id",
            *self.regressor_features, *self.probability_features,
        ]))

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if FORBIDDEN & set(features.columns):
            raise ValueError(f"Future or target columns: {sorted(FORBIDDEN & set(features.columns))}")
        missing = set(self.input_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Missing prediction columns: {sorted(missing)}")
        if features.sample_id.astype(str).duplicated().any():
            raise ValueError("Duplicate sample_id")
        at = pd.to_datetime(features["T"])
        target = pd.to_datetime(features["target_time_begin"])
        horizon = (target - at).dt.total_seconds()
        if not ((horizon > 600) & (horizon <= 900)).all():
            raise ValueError("Target stop must be in (T+10min, T+15min]")

        reg_frame = prepare_categorical_columns(
            features, ["tr_id", "route_signature", "section_id", "schedule_target_address"]
        )
        residual = np.asarray(self.regressor.predict(reg_frame[self.regressor_features]), dtype=float)
        delay = features.cur_dev_s.to_numpy(dtype=float) + residual
        probability = np.asarray(predict_probability(self.probability_bundle, features), dtype=float)
        if not np.isfinite(delay).all() or not np.isfinite(probability).all():
            raise ValueError("Non-finite model output")
        if ((probability < 0) | (probability > 1)).any():
            raise ValueError("Probability outside [0, 1]")
        return pd.DataFrame({
            "sample_id": features.sample_id.astype(str).to_numpy(),
            "predicted_delay_s": delay,
            "probability_delay_over_120s": probability,
        })


def prediction_frame(path: str | Path, predictor: Predictor) -> pd.DataFrame:
    """Select exactly the online-available fields from the shared feature table."""
    frame = pd.read_parquet(path)
    return frame.loc[:, predictor.input_columns].copy()


def json_records(frame: pd.DataFrame) -> list[dict]:
    """Encode nullable numeric and timestamp features safely for the ML HTTP API."""
    return json.loads(frame.to_json(orient="records", date_format="iso", date_unit="us", double_precision=15))
