import logging
import os
import numpy as np
from functools import reduce
from typing import List, Tuple
from scipy.stats import pearsonr

from pandas import DataFrame, Series, merge, get_dummies, concat, read_csv

from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.feature_selection._univariate_selection import _BaseFilter, _clean_nans

# Local
from data.longitudinal_utils import aggregate_cat_feature


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
        try:
            # Try normally reading the csv with pandas, if it fails the formatting is strange
            df = read_csv(os.path.join(raw_data_dir, file))
        except Exception:
            logging.warning(f"Unexpected encoding in {file}. Encoding with cp1252.")
            df = read_csv(os.path.join(raw_data_dir, file), encoding="cp1252")

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
    Sklearn compatible: https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/feature_selection/_univariate_selection.py#L430"""

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
        masks = []
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
