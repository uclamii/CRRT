import logging
from typing import Callable, Dict, List, Optional, Union
import numpy as np
from scipy.stats import skew
from datetime import timedelta

from pandas import DataFrame, Grouper, Series, concat, get_dummies, to_datetime

UNIVERSAL_TIME_COL_NAME = "DATE"


def std(df: DataFrame) -> float:
    """agg: np.std does not work properly if passed directly."""
    return np.std(df)


# For continuous valued repeated measurements, how to aggregate across a window of time
AGGREGATE_FUNCTIONS = [min, max, np.mean, std, skew, len]


def get_delta(delta: Optional[Dict[str, int]] = None) -> timedelta:
    """Get timedelta if dict is specified, else delta is nothing."""
    if delta:
        # while relativedelta is more accurate, timedelta is much faster
        time_delta = timedelta(
            days=365 * delta.get("YEARS", 0)
            + 30 * delta.get("MONTHS", 0)
            + delta.get("DAYS", 0)
        )
    else:
        time_delta = timedelta(days=0)
    return time_delta


def get_time_window_mask(
    outcomes_df: DataFrame,
    pre_start_delta: Optional[Dict[str, int]] = None,
    post_start_delta: Optional[Dict[str, int]] = None,
    mask_end: str = "End Date",
    slide_window_by: int = 0,
) -> DataFrame:
    """
    Assumes outcomes_df index is [pt, start date]
    Assumes Start Date, and End Date are already dtype = DateTime.
    Mask is [start date - pre_start_delta, min(end date, start date + post_start_delta)] if both are specified.
    If neither are specified, [start date, mask_end].

    If slide_window_by is specified (positive) it will slide the window up by n days.
    NOTE: Even if a patient doesn't have pre_start_delta days of data,
    if after sliding their measurements are still in the new/slided range,
    then they will be included.
    E.g., pt has 1 day data before start, and 1 day after.
        prestart=2, slide=0 => not included
        prestart=2, slide=1 => included
    """
    # get df indexed by pt and then columns ["start", "end"]
    window_dates = outcomes_df["End Date"].reset_index(level="Start Date")
    mask_start_interval = window_dates["Start Date"] - get_delta(pre_start_delta)

    # don't include "Start/End Date" data if they are the endpoints.
    if post_start_delta:
        # Pick whichever is earlier:
        # mask_end, or however many days after start is specified (if specified)
        mask_end_interval = concat(
            [
                window_dates["End Date"],
                # Add a day to include dates that might be later in the day
                # e.g. cutoff Date: 1/4/yyyy, want to keep 1/4/yyyy 4 PM
                # Simply checking date <= end date won't work if granularity is whole days
                (
                    window_dates["Start Date"]
                    + get_delta(post_start_delta)
                    + timedelta(days=1)
                ),
            ],
            axis=1,
        ).min(axis=1)
    else:  # just use mask_end
        # if I modify mask_end_interval it will modify window_dates, so copy
        mask_end_interval = window_dates[mask_end].copy()

    if slide_window_by:  # Ignore null
        # Slide the window (by default 0 days, so no sliding occurs)
        # slide window up from start
        mask_start_interval += timedelta(days=slide_window_by)
        mask_end_interval += timedelta(days=slide_window_by)

        # # slide window down from end
        # mask_start_interval = (
        #     mask_end_interval
        #     - get_delta(pre_start_delta)
        #     - timedelta(days=slide_window_by)
        # )
        # mask_end_interval -= timedelta(days=slide_window_by)

    time_window = concat(
        [window_dates["Start Date"], mask_start_interval, mask_end_interval],
        axis=1,
        keys=["Start Date", "Window Start", "Window End"],
    )

    # filter to ensure all patients have X days of data needed to aggregate (static)
    cutoff = window_dates["End Date"]
    if slide_window_by:
        # When the patients are controls the end date == start date (or 0 days on CRRT)
        no_crrt = (outcomes_df["CRRT Total Days"] == 0).values
        # Need to slide the end date as well (it's artificial anyway)
        # Or else, the window will be empty when we slide since
        cutoff[no_crrt] += timedelta(days=slide_window_by)
    time_window = time_window[time_window["Window End"] <= cutoff]

    # starts: list of all corresponding start dates to the windows, window start = list of all starts, window end = list of all ends (same size) per patient
    return time_window.groupby("IP_PATIENT_ID").agg(tuple).applymap(list)


def dates_are_in_range(dates: Series, start: Series, end: Series) -> Series:
    """
    Checks if dates are in in the range between start and end.
    If start == end, just check that date is the start date (only 1 day of data).
    """
    return ((dates >= start) & (dates < end)).where(start != end, dates == start)


def apply_time_window_mask(
    longitudinal_df: DataFrame, time_col: str, time_window: DataFrame
) -> DataFrame:
    """
    Assumes time_col dtype is DateTime.
    Mask the given feature df to entries within some time frame/window per pt and per start date.
    A pt can have multiple treatements.
    Check that the entry date is in any of the windows from the treatment dates.
    """

    def get_treatment_i_window_fn(
        treatment_idx: int,
    ) -> Callable[[DataFrame], Series]:
        """
        df per treatment (if a pt has multiple treatments there'll be multiple dfs)
        If a pt only has 1 treatment the following dfs will have NaT entries.
        Treatment index = idx
        """

        # function to map the section of the time_window df for a pt
        def get_window(patient_time_window_df: DataFrame) -> Series:
            try:
                return patient_time_window_df[treatment_idx]
            except IndexError:  # Ignore patients that don't have outcomes.
                return np.nan

        return get_window

    # window start len == window end len, just get the max of either
    max_num_treatments = time_window.iloc[:, 0].map(len).max()
    # 1st df = 1st start and end dates for all pts, 2nd df = 2nd "", etc.
    window_dfs = [
        time_window.applymap(get_treatment_i_window_fn(idx), na_action="ignore")
        for idx in range(max_num_treatments)
    ]
    # merge on pt id to combine with time_window
    longitudinal_df = longitudinal_df.set_index("IP_PATIENT_ID")
    # Mask: keep entries within requisite interval
    df_entries_in_range = concat(
        map(
            # comparisons with NaT are always false, which is the behavior we want
            # apply mask to window_i, keep intact which start_date window it fell in
            lambda df: df[
                dates_are_in_range(
                    df[time_col], start=df["Window Start"], end=df["Window End"]
                )
                # keep only Start Date, not Window Start and Window End
            ].drop(["Window Start", "Window End"], axis=1),
            # keep only pts with outcomes
            map(
                lambda window: longitudinal_df.merge(
                    window,
                    on="IP_PATIENT_ID",
                    how="inner",  # , left_index=True, right_index=True
                ),
                window_dfs,
            ),
        ),
    )
    logging.info(
        f"Dropping {longitudinal_df.shape[0] - df_entries_in_range.shape[0]} rows outside of time window."
    )
    # the aggs don't expect pt id to be in the index, reset the index.
    return df_entries_in_range.reset_index()


def hcuppy_map_code(
    df: DataFrame,
    code_col: str,
    exploded_cols: List[str],
    hcuppy_converter_function: Callable[[str], Dict[str, str]],
) -> DataFrame:
    """Use hcuppy lib to map ICD to CCS or CPT."""
    mapped_dict = df[code_col].apply(lambda code: hcuppy_converter_function(code))

    # series of dicts, explode each dict into its own column
    mapped_dict = DataFrame(mapped_dict.tolist())
    mapped_dict.columns = exploded_cols

    # combine the granular procedure cpt codes with the higher level ones from hcuppy
    df = concat([df, mapped_dict], axis=1)
    return df


def aggregate_cat_feature(
    cat_df: DataFrame,
    agg_on: str,
    time_col: Optional[str] = None,
    time_interval: Optional[str] = None,
    time_window: Optional[Union[DataFrame, str]] = None,
) -> DataFrame:
    """
    Aggregate a categorical feature. Basically "Bag of words".
    Will onehot encode, and sum up occurrences for a given patient for a given time window (if given, else all time points) over each time interval.
    Time window is traditionally time_before_start_date -> start_date.
    Time interval comes from the resample/Grouper (e.g. "1D" is daily.)

    Produce multindex: patient > treatment # > day > features.
    """
    if time_col:
        # Enforce date columnn is a datetime object
        cat_df[time_col] = to_datetime(cat_df[time_col])

        # mask for time if we have a time_col
        cat_df = apply_time_window_mask(cat_df, time_col, time_window)
        cat_df_cols = ["IP_PATIENT_ID", agg_on, time_col, "Start Date"]
    else:
        cat_df_cols = ["IP_PATIENT_ID", agg_on]

    # Get dummies for the categorical column
    cat_feature = get_dummies(cat_df[cat_df_cols], columns=[agg_on])
    # TODO: CategoricalDtype with all the CCS codes here

    # Sum across a patient and per CRRT treatment (indicated by Start Date):
    # per time interval (if specified) over a whole time window
    group_by = ["IP_PATIENT_ID", "Start Date"]
    # chunk by time interval if exists
    if time_interval and time_col:
        # Uniform date column name if aggregating by time_interval
        cat_feature.rename(columns={time_col: UNIVERSAL_TIME_COL_NAME}, inplace=True)
        # use with Grouper instead of resample
        # Ref: https://stackoverflow.com/a/32012129/1888794
        group_by.append(Grouper(key=UNIVERSAL_TIME_COL_NAME, freq=time_interval))

    return cat_feature.groupby(group_by).sum()


def aggregate_ctn_feature(
    ctn_df: DataFrame,
    agg_on: str,  # specify what is the name of the value (e.g. sbp vs dbp)
    agg_values_col: str,
    time_col: str,
    time_interval: Optional[str] = None,
    time_window: Optional[Union[DataFrame, str]] = None,
) -> DataFrame:
    """Aggregate a continuous longitudinal feature (e.g., vitals, labs).
    Filter time window based on a column name provided.
    Aggregate on a column name provided:
        need a column for the name to group by, and the corresponding value column name.
    Time window is traditionally time_before_start_date -> start_date.
    Time interval comes from the resample (e.g. "1D" is daily.)
    """
    # Enforce date columnn is a datetime object
    ctn_df[time_col] = to_datetime(ctn_df[time_col])
    # filter to window
    ctn_df = apply_time_window_mask(ctn_df, time_col, time_window)

    # Sum across a patient and per time interval (if specified) over a whole time window per feature
    group_by = ["IP_PATIENT_ID", "Start Date"]
    # group_by = ["IP_PATIENT_ID"]
    # chunk by time interval if exists
    if time_interval:
        # Uniform date column name if aggregating by time_interval
        ctn_df.rename(columns={time_col: UNIVERSAL_TIME_COL_NAME}, inplace=True)
        # use with Grouper instead of resample
        # Ref: https://stackoverflow.com/a/32012129/1888794
        group_by.append(Grouper(key=UNIVERSAL_TIME_COL_NAME, freq=time_interval))
    group_by.append(agg_on)

    # index: [ID, time(?), agg_on], columns: [agg_values_col: [agg_fns]]
    ctn_feature = (
        ctn_df.groupby(group_by)
        .agg({agg_values_col: AGGREGATE_FUNCTIONS})
        .droplevel(0, axis=1)  # want to just keep AGG_FN names
    )

    # stack:=  index: [id, time(?), agg_on, agg_fn]
    # unstack: 1st agg_on names (e.g. sbp dbp) (level -2 in index)
    # then under that, agg_fn (level -1 in index)
    ctn_feature = ctn_feature.stack().unstack([-2, -1])
    ctn_feature.columns = ctn_feature.columns.map("_".join)

    return ctn_feature
