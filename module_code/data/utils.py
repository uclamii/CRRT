import logging
import os
import numpy as np
from functools import reduce
from typing import List, Optional, Tuple, Dict
from scipy.stats import pearsonr

from pandas import DataFrame, Series, merge, get_dummies, concat, read_csv
from pandas.errors import ParserError

from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.feature_selection._univariate_selection import _BaseFilter, _clean_nans

# Local
from data.longitudinal_utils import aggregate_cat_feature

FILE_NAMES = {
    "cpt": "Procedures.txt",
    "pr": "Problem_Lists.txt",
    "pr_dx": "Problem_List_Diagnoses.txt",
    "labs": "Labs.txt",
    "rx": "Medications.txt",
    "vitals": "Flowsheet_Vitals.txt",
    "dx": "Encounter_Diagnoses.txt",
    "enc": "Encounters.txt",
}


def time_delta_to_str(delta: Dict[str, int]) -> str:
    """
    Coverts timedelta dict to str form: 5 years, 4 months, and 3 days => 5y4m3d
    Ignore values of 0: 4months and 3 days => 4m3d
    Assumes order of keys are: years, months, then days.
    """
    delta_str = ""
    for time_name, amount in delta.items():
        if amount > 0:
            delta_str += f"{amount}{time_name[0].lower()}"
    return delta_str


def get_preprocessed_file_name(
    pre_start_delta: Optional[Dict[str, int]] = None,
    post_start_delta: Optional[Dict[str, int]] = None,
    time_interval: Optional[str] = None,
    time_window_end: Optional[str] = None,
    slide_window_by: Optional[int] = None,
    preprocessed_df_file: Optional[str] = None,
    serialization: str = "feather",
) -> str:
    """
    Uses preprocessed_df_file for file name for preprocessed dataframe.
    However, if it's not provided it will automatically generate a name based on the arguments used to generate the file.
    The name will be the same for all cohorts because we assume each cohort is in a separate directory.

    df_{time interval the features are aggregated in}agg_[{time window start},{time window end}].extension
    If providing deltas: [startdate-pre_start_delta,startdate+post_start_delta]
    If providing neither [startdate,time_window_end].
    If sliding the window (window > 0): [start + i - pre], (start+post | end) + i]
    """
    if preprocessed_df_file:
        return preprocessed_df_file + f".{serialization}"
    fname = "df"
    if time_interval:
        fname += f"_{time_interval}agg"
    # time window
    fname += "_[startdate"
    if slide_window_by:
        sign = "+" if slide_window_by > 0 else ""
        fname += f"{sign}{slide_window_by}"

    if pre_start_delta:
        # subtracting the delta time
        fname += f"-{time_delta_to_str(pre_start_delta)}"
    fname += ","
    # end of window:
    if post_start_delta:
        fname += f"startdate+{time_delta_to_str(post_start_delta)}"
    else:
        fname += f"{time_window_end.replace(' ', '').lower()}"
    if slide_window_by:
        sign = "+" if slide_window_by > 0 else ""
        fname += f"{sign}{slide_window_by}"

    # Close
    fname += "]"

    return fname + "." + serialization


def onehot(
    df: DataFrame,
    cols_to_onehot: List[str],
    sum_across_patient: bool = False,
) -> DataFrame:
    """
    One-hot encodes list of features and add it back into the df.
    If summing across the patient, it will aggregate across that patient (which shouldn't affect columns that don't have more than one entry anyway.)
    """
    if sum_across_patient:
        onehot_dfs = [
            aggregate_cat_feature(df, onehot_col) for onehot_col in cols_to_onehot
        ]
        # add back into the df, drop the original columns since we have onehot version now
        # we have to merge instead of concat bc agg_cat_features leaves in patient id (for summation)
        return reduce(
            lambda df1, df2: merge(df1, df2, on="IP_PATIENT_ID"),
            onehot_dfs,
        )

    # otherwise, just do normal dummies and concat
    onehot_dfs = [
        get_dummies(df[onehot_col], prefix=onehot_col) for onehot_col in cols_to_onehot
    ]
    # add back into the df, drop the original columns since we have onehot version now
    return concat([df.drop(cols_to_onehot, axis=1)] + onehot_dfs, axis=1)


def loading_message(what_is_loading: str) -> None:
    """Helper function to know what table is being loaded during preprocessing."""
    logging.info("*" * 5 + f"Loading {what_is_loading}..." + "*" * 5)


def read_files_and_combine(
    files: List[str],
    raw_data_dir: str,
    on: List[str] = ["IP_PATIENT_ID"],
    how: str = "inner",
) -> DataFrame:
    """
    Takes one or more files in a list and returns a combined dataframe.
    Deals with strangely formatted files from the dataset.
    """
    dfs = []

    for file in files:
        path = os.path.join(raw_data_dir, file)
        try:
            # Try normally reading the csv with pandas, if it fails the formatting is strange
            df = read_csv(path)
        except ParserError as e:
            logging.warn(e)
            logging.warn("Skipping bad lines...")
            df = read_csv(path, on_bad_lines="skip")
        except Exception:
            logging.warning(f"Unexpected encoding in {file}. Encoding with cp1252.")
            df = read_csv(path, encoding="cp1252")

        # Enforce all caps column names
        dfs.append(df.set_axis(df.columns.str.upper(), axis=1))

    combined = reduce(lambda df1, df2: merge(df1, df2, on=on, how=how), dfs)
    return combined


def f_pearsonr(X: DataFrame, labels: Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use correlation of features to the labels as a scoring function
    for sklearn feature selection.
    """
    if isinstance(X, DataFrame):
        scores_and_pvalues = X.apply(lambda feature: pearsonr(feature, labels), axis=0)
        scores, pvalues = zip(*scores_and_pvalues)
        scores = scores.fillna(0)
    else:
        # Numpy will automatically unpack the 2 rows into each var
        scores, pvalues = np.apply_along_axis(
            lambda feature: pearsonr(feature, labels), 0, X
        )
        scores = np.nan_to_num(scores, nan=0)

    return (scores, pvalues)


class Preselected(SelectorMixin, BaseEstimator):
    """Select features according preset mask."""

    # Do nothing score func
    def __init__(self, support_mask: List[bool]):
        self.support_mask = support_mask

    def fit(self, X, y):
        return self

    def _get_support_mask(self):
        return self.support_mask


class SelectThreshold(_BaseFilter):
    """Select features according to a threshold of the highest scores.
    Sklearn compatible: https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/feature_selection/_univariate_selection.py#L430
    """

    def __init__(self, score_func=f_classif, *, threshold=0):
        super().__init__(score_func=score_func)
        self.threshold = threshold

    def _get_support_mask(self):
        check_is_fitted(self)

        scores = abs(_clean_nans(self.scores_))
        mask = scores > self.threshold
        ties = np.where(scores == self.threshold)[0]
        if len(ties):
            max_feats = int(len(scores) * self.percentile / 100)
            kept_ties = ties[: max_feats - mask.sum()]
            mask[kept_ties] = True
        return mask


def get_pt_type_indicators(df: DataFrame) -> DataFrame:
    """Look in diagnoses and problems for ccs codes related to heart, liver, and infection."""
    tables = ["dx", "pr"]
    types = [
        {"name": "liver", "codes": [6, 16, 149, 150, 151, 222]},
        # TODO: should these be mutually exclusive
        # could potentially add "shock"/249 to heart too
        {
            "name": "heart",
            "codes": [
                96,
                97,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                114,
                115,
                116,
                117,
            ],
        },
        {"name": "infection", "codes": [1, 2, 3, 4, 5, 7, 8, 249]},
    ]

    for pt_type in types:
        masks = [0]  # in case there are no masks, then filter just becomes all zeros
        for code in pt_type["codes"]:
            for table_name in tables:
                column_name = f"{table_name}_CCS_CODE_{code}"
                # codes may not be in the dataset
                if column_name in df:
                    masks.append((df[column_name] > 0).astype(int))

        df[f"{pt_type['name']}_pt_indicator"] = reduce(
            lambda maska, maskb: maska | maskb, masks
        )
    return df
