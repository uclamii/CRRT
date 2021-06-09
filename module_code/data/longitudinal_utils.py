import logging
from typing import Callable, Dict, List, Optional
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
        outcomes_df[["IP_PATIENT_ID", "End Date"]], on="IP_PATIENT_ID", how="right"
    )

    # Enforce date columnn is a datetime object
    end_date = pd.to_datetime(merged_df["End Date"])
    dates = pd.to_datetime(merged_df[timestamp_feature_name])

    # Mask: keep entries for feature with a date within time_window years and months from end date of outcome
    mask = dates >= (
        end_date
        - timedelta(days=360 * time_window["YEARS"], weeks=4 * time_window["MONTHS"])
    )

    logging.info(f"Dropping {df.shape[0] - sum(mask)} rows outside of time window.")

    # Remove the merged end date used for masking and return
    return merged_df[mask].drop("End Date", axis=1)


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
    time_window: Dict[str, int] = TIME_WINDOW,
) -> pd.DataFrame:
    """
    Aggregate a categorical feature. Basically "Bag of words".
    Will onehot encode, and sum up occurrences for a given patient for a given time window (if given, else all time points).
    """
    # mask for time if we have a time_col
    cat_df = (
        time_window_mask(outcomes_df, cat_df, time_col, time_window)
        if time_col
        else cat_df
    )

    # Get dummies for the categorical column
    cat_feature = pd.get_dummies(cat_df[["IP_PATIENT_ID", agg_on]], columns=[agg_on])
    # Sum across a patient (within a time window)
    cat_feature = cat_feature.groupby("IP_PATIENT_ID").apply(lambda df: df.sum(axis=0))

    # fix indices ruined by groupby
    cat_feature = cat_feature.drop("IP_PATIENT_ID", axis=1).reset_index()

    return cat_feature


def aggregate_ctn_feature(
    outcomes_df: pd.DataFrame,
    ctn_df: pd.DataFrame,
    agg_on: str,
    agg_values_col: str,
    time_col: str,
    time_window: Dict[str, int] = TIME_WINDOW,
) -> pd.DataFrame:
    """Aggregate a continuous longitudinal feature (e.g., vitals, labs).
    Filter time window based on a column name provided.
    Aggregate on a column name provided:
        need a column for the name to group by, and the corresponding value column name.
    """
    # filter to window
    ctn_df = time_window_mask(outcomes_df, ctn_df, time_col, time_window)

    # Apply aggregate functions (within time window)
    ctn_feature = ctn_df.groupby(["IP_PATIENT_ID", agg_on]).agg(
        {agg_values_col: AGGREGATE_FUNCTIONS}
    )

    # Flatten aggregations from multi_index into a feature vector
    ctn_feature = ctn_feature.unstack()
    ctn_feature.columns = ctn_feature.columns.map("_".join)
    ctn_feature.reset_index(inplace=True)

    return ctn_feature
