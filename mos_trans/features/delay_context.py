import numpy as np
import pandas as pd


MAX_LAG_AGE_S = 900


def add_delay_trend_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["_delay_original_order"] = np.arange(len(frame))
    frame = (frame.sort_values(["tr_id", "T"]).reset_index(drop=True))
    grouped = frame.groupby("tr_id", sort=False)

    frame["previous_T"] = (grouped["T"].shift(1))
    frame["previous_cur_dev_s"] = (grouped["cur_dev_s"].shift(1))
    frame["previous_delay_age_s"] = (frame["T"] - frame["previous_T"]).dt.total_seconds()

    valid_previous = (frame["previous_delay_age_s"] > 0) & (frame["previous_delay_age_s"] <= MAX_LAG_AGE_S)

    frame["previous_delay_available"] = (valid_previous.astype(int))
    frame.loc[~valid_previous, "previous_cur_dev_s"] = np.nan
    frame["delay_change_since_previous_s"] = (frame["cur_dev_s"]
                                            - frame["previous_cur_dev_s"])
    frame["delay_change_per_min"] = (frame["delay_change_since_previous_s"]
                                     / frame["previous_delay_age_s"] * 60.0)

    frame["cur_dev_lag_2"] = (grouped["cur_dev_s"].shift(2))
    frame["T_lag_2"] = (grouped["T"].shift(2))
    frame["lag_2_age_s"] = (frame["T"] - frame["T_lag_2"]).dt.total_seconds()

    valid_lag_2 = (frame["lag_2_age_s"] > 0) & (frame["lag_2_age_s"] <= 1800)
    frame["lag_2_available"] = (valid_lag_2.astype(int))
    frame.loc[ ~valid_lag_2, "cur_dev_lag_2"] = np.nan

    frame["delay_change_lag_2_s"] = (frame["cur_dev_s"]- frame["cur_dev_lag_2"])
    frame["delay_change_lag_2_per_min"] = (frame["delay_change_lag_2_s"]
                                           / frame["lag_2_age_s"] * 60.0)

    frame = frame.drop(columns=["previous_T", "T_lag_2"])

    frame = (frame.sort_values("_delay_original_order")
             .drop(columns="_delay_original_order")
             .reset_index(drop=True)
    )

    return frame