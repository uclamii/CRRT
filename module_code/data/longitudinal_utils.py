from typing import Callable, Dict, List
import numpy as np
import pandas as pd
from scipy.stats import skew
from datetime import timedelta

# For continuous valued repeated measurements, how to aggregate across a window of time
AGGREGATE_FUNCTIONS = [min, max, np.mean, np.std, skew, len]

# What window of time to limit our analysis to (relative to end date of outcome)
TIME_WINDOW = {"YEARS": 1, "MONTHS": 0}

# TODO: patients that are in the other files but not in the outcome files will not have a corresponding entry/time.
def time_window_mask(
    outcomes_df: pd.DataFrame,
    df: pd.DataFrame,
    timestamp_feature_name: str,
    time_window: Dict[str, int] = TIME_WINDOW,
) -> pd.DataFrame:
    """Mask the given feature df to entries within some time frame/window from the end date of the outcome."""

    # Merge feature with end date of outcome
    merged_df = df.merge(
        outcomes_df[["IP_PATIENT_ID", "End Date"]], on="IP_PATIENT_ID", how="left"
    )

    # Enforce date columnn is a datetime object
    dates = pd.to_datetime(merged_df[timestamp_feature_name])

    # Mask: keep entries for feature with a date within time_window years and months from end date of outcome
    mask = dates >= (
        merged_df["End Date"]
        - timedelta(days=360 * time_window["YEARS"], weeks=4 * time_window["MONTHS"])
    )

    print(f"Dropping {df.shape[0] - sum(mask)} rows outside of time window.")

    # Remove the merged end date used for masking and return
    return merged_df[mask].drop("End Date", axis=1)


def hcuppy_map_code(
    df: pd.DataFrame,
    code_col: str,
    exploded_cols: List[str],
    hcuppy_converter_function: Callable[[str], str],
) -> pd.DataFrame:
    mapped_dict = df[code_col].apply(lambda code: hcuppy_converter_function(code))

    # series of dicts, explode each dict into its own column
    mapped_dict = pd.DataFrame(mapped_dict.tolist())
    mapped_dict.columns = exploded_cols

    # combine the granular procedure cpt codes with the higher level ones from hcuppy
    df = pd.concat([df, mapped_dict], axis=1)
    return df
