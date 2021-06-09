import logging
import os
import pandas as pd
from functools import reduce
from typing import List
from data.longitudinal_utils import aggregate_cat_feature

# Just uncomment for the data_dir that makes sense for your machine
DATA_DIR = "/home/davina/Private/dialysis-data"
# DATA_DIR = r"C:\Users\arvin\Documents\ucla research\CRRT project"


def onehot(
    df: pd.DataFrame, cols_to_onehot: List[str], sum_across_patient: bool = False,
) -> pd.DataFrame:
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
            lambda df1, df2: pd.merge(df1, df2, on="IP_PATIENT_ID"), onehot_dfs,
        )

    # otherwise, just do normal dummies and concat
    onehot_dfs = [
        pd.get_dummies(df[onehot_col], prefix=onehot_col)
        for onehot_col in cols_to_onehot
    ]
    # add back into the df, drop the original columns since we have onehot version now
    return pd.concat([df.drop(cols_to_onehot, axis=1)] + onehot_dfs, axis=1)


def loading_message(what_is_loading: str) -> None:
    """Helper function to know what table is being loaded during preprocessing."""
    logging.info("*" * 5 + f"Loading {what_is_loading}..." + "*" * 5)


def read_files_and_combine(
    files: List[str], on: List[str] = ["IP_PATIENT_ID"], how: str = "inner"
) -> pd.DataFrame:
    """
    Takes one or more files in a list and returns a combined dataframe.
    Deals with strangely formatted files from the dataset.
    """
    dfs = []

    for file in files:
        try:
            # Try normally reading the csv with pandas, if it fails the formatting is strange
            df = pd.read_csv(os.path.join(DATA_DIR, file))
        except IOError:
            logging.warning(f"Unexpected encoding in {file}. Encoding with cp1252.")
            df = pd.read_csv(os.path.join(DATA_DIR, file), encoding="cp1252")

        # Enforce all caps column names
        dfs.append(df.set_axis(df.columns.str.upper(), axis=1))

    combined = reduce(lambda df1, df2: pd.merge(df1, df2, on=on, how=how), dfs)
    return combined
