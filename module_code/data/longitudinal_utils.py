import logging
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import skew
from datetime import timedelta

# For continuous valued repeated measurements, how to aggregate across a window of time
AGGREGATE_FUNCTIONS = [min, max, np.mean, np.std, skew, len]

# What window of time to limit our analysis to (relative to end date of outcome)
TIME_BEFORE_START_DATE = {"YEARS": 0, "MONTHS": 0, "DAYS": 14}

# TODO: patients that are in the other files but not in the outcome files will not have a corresponding entry/time.
def time_window_mask(
    outcomes_df: pd.DataFrame,
    df: pd.DataFrame,
    time_col: str,
    time_before_start_date: Dict[str, int] = TIME_BEFORE_START_DATE,
    mask_end: str = "Start Date",
) -> pd.DataFrame:
    """Mask the given feature df to entries within some time frame/window from the end date of the outcome.
    Assumes timecol, Start Date, and End Date are already dtype = DateTime."""
    outcome_date_cols = (
        ["Start Date", "End Date"] if mask_end == "End Date" else ["Start Date"]
    )

    # Merge feature with end date of outcome
    merged_df = df.merge(
        outcomes_df[["IP_PATIENT_ID"] + outcome_date_cols],
        on="IP_PATIENT_ID",
        how="right",
    )

    # Mask: keep entries for feature with a date within time_before_start_date years and months from start date of crrt
    # while relativedelta is more accurate, timedelta is much faster
    mask_start_interval = merged_df["Start Date"] - timedelta(
        days=365 * time_before_start_date.get("YEARS", 0)
        + 30 * time_before_start_date.get("MONTHS", 0)
        + time_before_start_date.get("DAYS", 0)
    )
    # Add a day to include dates that might be later in the day
    # e.g. End Date: 1/4/yyyy, want to keep 1/4/yyyy 4 PM
    # Simply checking date <= end date won't work if granularity is whole days
    mask_end_interval = merged_df[mask_end] + timedelta(days=1)

    # Mask: keep entries within requisite interval
    mask = (merged_df[time_col] >= mask_start_interval) & (
        merged_df[time_col] < mask_end_interval
    )

    # Remove the merged end date used for masking and return
    logging.info(f"Dropping {df.shape[0] - sum(mask)} rows outside of time window.")
    return merged_df[mask].drop(outcome_date_cols, axis=1)


def hcuppy_map_code(
    df: pd.DataFrame,
    code_col: str,
    exploded_cols: List[str],
    hcuppy_converter_function: Callable[[str], str],
) -> pd.DataFrame:
    """Use hcuppy lib to map ICD to CCS or CPT."""
    mapped_dict = df[code_col].apply(lambda code: hcuppy_converter_function(code))

    # series of dicts, explode each dict into its own column
    mapped_dict = pd.DataFrame(mapped_dict.tolist())
    mapped_dict.columns = exploded_cols

    # combine the granular procedure cpt codes with the higher level ones from hcuppy
    df = pd.concat([df, mapped_dict], axis=1)
    return df


def aggregate_cat_feature(
    cat_df: pd.DataFrame,
    agg_on: str,
    outcomes_df: Optional[pd.DataFrame] = None,  # used for timing
    time_col: Optional[str] = None,
    time_interval: Optional[str] = None,
    time_before_start_date: Dict[str, int] = TIME_BEFORE_START_DATE,
    time_window_end: str = "Start Date",
) -> pd.DataFrame:
    """
    Aggregate a categorical feature. Basically "Bag of words".
    Will onehot encode, and sum up occurrences for a given patient for a given time window (if given, else all time points) over each time interval.
    Time window is traditionally time_before_start_date -> start_date.
    Time interval comes from the pd.resample (e.g. "1D" is daily.)
    """
    if time_col:
        # Enforce date columnn is a datetime object
        cat_df[time_col] = pd.to_datetime(cat_df[time_col])

        # mask for time if we have a time_col
        cat_df = time_window_mask(
            outcomes_df, cat_df, time_col, time_before_start_date, time_window_end
        )

    cat_df_cols = (
        ["IP_PATIENT_ID", agg_on, time_col] if time_col else ["IP_PATIENT_ID", agg_on]
    )

    # Get dummies for the categorical column
    cat_feature = pd.get_dummies(cat_df[cat_df_cols], columns=[agg_on])

    # Sum across a patient and per time interval (if specified) over a whole time window
    cat_feature_grouped = cat_feature.groupby("IP_PATIENT_ID")
    # resample by time interval if exists
    if time_interval and time_col:
        # Uniform date column name if aggregating by time_interval
        cat_feature.rename(columns={time_col: "DATE"}, inplace=True)
        cat_feature_grouped = cat_feature_grouped.resample(time_interval, on="DATE")

    return cat_feature_grouped.sum()


def aggregate_ctn_feature(
    outcomes_df: pd.DataFrame,
    ctn_df: pd.DataFrame,
    agg_on: str,
    agg_values_col: str,
    time_col: str,
    time_interval: Optional[str] = None,
    time_before_start_date: Dict[str, int] = TIME_BEFORE_START_DATE,
    time_window_end: str = "Start Date",
) -> pd.DataFrame:
    """Aggregate a continuous longitudinal feature (e.g., vitals, labs).
    Filter time window based on a column name provided.
    Aggregate on a column name provided:
        need a column for the name to group by, and the corresponding value column name.
    Time window is traditionally time_before_start_date -> start_date.
    Time interval comes from the pd.resample (e.g. "1D" is daily.)
    """
    # Enforce date columnn is a datetime object
    ctn_df[time_col] = pd.to_datetime(ctn_df[time_col])
    # filter to window
    ctn_df = time_window_mask(
        outcomes_df, ctn_df, time_col, time_before_start_date, time_window_end
    )

    # Apply aggregate functions (within time window)
    # reset index due to strange error from mask:
    # https://github.com/pandas-dev/pandas/issues/35275#issuecomment-658208648
    ctn_feature_grouped = ctn_df.reset_index().groupby(
        ["IP_PATIENT_ID", agg_on, time_col]
    )
    # resample by time interval if exists
    if time_interval:
        ctn_feature_grouped = ctn_feature_grouped.resample(time_interval, on=time_col)
    # apply all aggregate functions to the target values column
    ctn_feature = ctn_feature_grouped.agg({agg_values_col: AGGREGATE_FUNCTIONS})
    # by resampling by day / etc it will create a duplicate time_col, get rid of granular version
    # Does nothing if not resampling by time interval
    ctn_feature.reset_index(level=2, drop=True, inplace=True)

    # unstack at the agg_values_col instead of time_col (level=2, after dropping dup)
    ctn_feature = ctn_feature.unstack(level=1)
    ctn_feature.columns = ctn_feature.columns.map("_".join)
    ctn_feature.reset_index(inplace=True)

    # Uniform date column if aggregating by time_interval
    if time_interval:
        ctn_feature.rename(columns={time_col: "DATE"}, inplace=True)
    return ctn_feature
