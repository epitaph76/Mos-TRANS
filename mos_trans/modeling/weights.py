import numpy as np
import pandas as pd


SYNTHETIC_TR_ID_START = 9_000_000


def is_synthetic_vehicle(frame: pd.DataFrame) -> pd.Series:
    numeric_id = pd.to_numeric(frame["tr_id"], errors="coerce")

    return (numeric_id >= SYNTHETIC_TR_ID_START)


def build_sample_weights(frame: pd.DataFrame, synthetic_weight: float) -> np.ndarray:
    if not 0.0 <= synthetic_weight <= 1.0:
        raise ValueError("synthetic_weight должен быть в диапазоне [0, 1]")

    synthetic_mask = is_synthetic_vehicle(frame)

    return np.where(synthetic_mask, synthetic_weight, 1.0)