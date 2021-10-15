import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import skew

from module_code.data.longitudinal_utils import (
    aggregate_cat_feature,
    aggregate_ctn_feature,
    time_window_mask,
    UNIVERSAL_TIME_COL_NAME,
)


class TestAggregateFeature(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.time_col_values = [  # yyyy-mm-dd
            "2019-03-05",
            "2018-02-04",
            "2020-04-06",
            "2020-04-06",
            "2020-04-06",
            "2019-03-05",
            "2018-02-04",
        ]
        self.patient_ids = [2, 1, 3, 3, 1, 4, 1]
        self.categorical_values = [
            "apple",
            "apple",
            "orange",
            "apple",
            "orange",
            "apple",
            "apple",
        ]
        self.continuous_values = [0.2, 0.1, 0.1, 0.3, 0.2, 0.4, 0.3]
        self.continuous_value_names = ["SBP", "SBP", "SBP", "DBP", "SBP", "SBP", "SBP"]
        self.time_col = "Time Column"
        self.agg_on = "Values"
        self.cat_df = pd.DataFrame(
            {
                "IP_PATIENT_ID": self.patient_ids,
                self.agg_on: self.categorical_values,
                self.time_col: pd.to_datetime(self.time_col_values),
            }
        )
        self.ctn_df = pd.DataFrame(
            {
                "IP_PATIENT_ID": self.patient_ids,
                "Value Names": self.continuous_value_names,
                self.agg_on: self.continuous_values,
                self.time_col: pd.to_datetime(self.time_col_values),
            }
        )

    @patch("module_code.data.longitudinal_utils.time_window_mask")
    def test_no_time_interval_ctn(self, mock_masked_df):
        """
        Tests aggregation of continuous features without using a time interval.
        """
        pt1_values = [0.1, 0.2, 0.3]
        correct_df = pd.DataFrame.from_dict(
            {
                1: [
                    0.1,
                    0.3,
                    np.mean(pt1_values),
                    np.std(pt1_values),
                    skew(pt1_values),
                    3,
                ]  # SBP
                + [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # DBP
                2: [0.2, 0.2, 0.2, 0, 0, 1]  # SBP
                + [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # DBP
                3: [0.1, 0.1, 0.1, 0, 0, 1] + [0.3, 0.3, 0.3, 0, 0, 1],
                4: [0.4, 0.4, 0.4, 0, 0, 1]  # SBP
                + [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # DBP
            },
            orient="index",
            columns=[
                name
                for value_name in ["SBP", "DBP"]
                for name in [
                    f"{value_name}_min",
                    f"{value_name}_max",
                    f"{value_name}_mean",
                    f"{value_name}_std",
                    f"{value_name}_skew",
                    f"{value_name}_len",
                ]
            ],
            dtype=np.float,  # dtype has to match or else this will falsely fail
        )
        correct_df.index.name = "IP_PATIENT_ID"
        # drop DBP_std since it won't exist for the test data we've written
        # There's no patient that has 2 DBP readings (so std is np.nan for all)
        # correct_df.drop("DBP_std", axis=1, inplace=True)
        mock_masked_df.return_value = self.ctn_df

        df = aggregate_ctn_feature(
            self.ctn_df,
            self.ctn_df,
            agg_on="Value Names",
            agg_values_col=self.agg_on,
            time_col=self.time_col,
        )
        self.assertTrue(correct_df.equals(df))

    @patch("module_code.data.longitudinal_utils.time_window_mask")
    def test_no_time_interval_cat(self, mock_masked_df):
        """
        Tests aggregation of categorical features without using a time interval.
        Mock the df returned after time window masking, so we don't need outcomes_df.
        Also we don't want to mix tests for time_window_mask and this function.
        Expect patients to be in the order of their values, regardless of order seen.
        Expected order: patient >  feature
        """
        correct_df = pd.DataFrame.from_dict(
            {1: [2, 1], 2: [1, 0], 3: [1, 1], 4: [1, 0]},
            orient="index",
            columns=[f"{self.agg_on}_apple", f"{self.agg_on}_orange"],
            dtype=np.uint8,  # dtype has to match or else this will falsely fail
        )
        correct_df.index.name = "IP_PATIENT_ID"
        mock_masked_df.return_value = self.cat_df

        df = aggregate_cat_feature(self.cat_df, self.agg_on, time_col=self.time_col,)
        self.assertTrue(correct_df.equals(df))

    @patch("module_code.data.longitudinal_utils.time_window_mask")
    def test_time_interval_ctn(self, mock_masked_df):
        """
        Tests aggregation of continuous features using a time interval.
        """
        correct_df = pd.DataFrame.from_dict(
            {
                (1, np.datetime64("2018-02-04")): [
                    0.1,
                    0.3,
                    0.2,
                    np.std([0.1, 0.3]),
                    skew([0.1, 0.3]),
                    2,
                ],
                (1, np.datetime64("2020-04-06")): [0.2, 0.2, 0.2, 0, 0, 1],
                (2, np.datetime64("2019-03-05")): [0.2, 0.2, 0.2, 0, 0, 1]  # SBP
                + [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # DBP
                (3, np.datetime64("2020-04-06")): [0.1, 0.1, 0.1, 0, 0, 1]  # SBP
                + [0.3, 0.3, 0.3, 0, 0, 1],  # DBP
                (4, np.datetime64("2019-03-05")): [0.4, 0.4, 0.4, 0, 0, 1]  # SBP
                + [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # DBP
            },
            orient="index",
            columns=[
                name
                for value_name in ["SBP", "DBP"]
                for name in [
                    f"{value_name}_min",
                    f"{value_name}_max",
                    f"{value_name}_mean",
                    f"{value_name}_std",
                    f"{value_name}_skew",
                    f"{value_name}_len",
                ]
            ],
            dtype=np.float,  # dtype has to match or else this will falsely fail
        )

        correct_df.index = pd.MultiIndex.from_tuples(
            correct_df.index, names=["IP_PATIENT_ID", UNIVERSAL_TIME_COL_NAME]
        )
        # ctn_df doesn't matter since this will replace inside aggregate_...
        mock_masked_df.return_value = self.ctn_df

        time_interval = "1D"
        df = aggregate_ctn_feature(
            self.ctn_df,
            self.ctn_df,
            agg_on="Value Names",
            agg_values_col=self.agg_on,
            time_col=self.time_col,
            time_interval=time_interval,
        )
        self.assertTrue(correct_df.equals(df))

    @patch("module_code.data.longitudinal_utils.time_window_mask")
    def test_time_interval_cat(self, mock_masked_df):
        """
        Tests categorical aggregation of features using a time interval.
        Mock the df returned after time window masking, so we don't need outcomes_df.
        Also we don't want to mix tests for time_window_mask and this function.

        Expect patients to be in order of value.
        Expect dates to be in order for each patient.
        Expected order: patient >  date >  feature
        """
        correct_df = pd.DataFrame.from_dict(
            {
                (1, np.datetime64("2018-02-04")): [2, 0],
                (1, np.datetime64("2020-04-06")): [0, 1],
                (2, np.datetime64("2019-03-05")): [1, 0],
                (3, np.datetime64("2020-04-06")): [1, 1],
                (4, np.datetime64("2019-03-05")): [1, 0],
            },
            orient="index",
            columns=[f"{self.agg_on}_apple", f"{self.agg_on}_orange"],
            dtype=np.uint8,  # dtype has to match or else this will falsely fail
        )
        correct_df.index = pd.MultiIndex.from_tuples(
            correct_df.index, names=["IP_PATIENT_ID", UNIVERSAL_TIME_COL_NAME]
        )
        # cat_df doesn't matter since this will replace inside aggregate_...
        mock_masked_df.return_value = self.cat_df

        # arg values
        time_interval = "1D"
        df = aggregate_cat_feature(
            self.cat_df,
            self.agg_on,
            time_col=self.time_col,
            time_interval=time_interval,
        )
        self.assertTrue(correct_df.equals(df))


class TestWindowMask(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    # TODO: add test cases for picking "End Date" as end point for window
    def test_edge_cases(self):
        # TODO: add comment explaining what behavior this test is testing for
        start_dates = [  # yyyy-mm-dd
            "2018-02-04",
            "2019-03-05",
            "2020-04-06",
            "2020-04-06",
            "2018-02-04",
            "2019-03-05",
            "2020-04-06",
        ]
        hour_deltas = [23, 24, 25, 48, -23, -24, -25]
        outcomes_df = pd.DataFrame(
            {
                "IP_PATIENT_ID": range(len(start_dates)),
                "Start Date": pd.to_datetime(start_dates),
            }
        )
        # TODO: is the 1s for the 24 hour deltas? to send if over the edge?
        hour_deltas = pd.Series(
            [timedelta(hours=delta, seconds=1) for delta in hour_deltas]
        )
        df = pd.DataFrame(
            {
                "IP_PATIENT_ID": range(len(start_dates)),
                "TIME_COL": outcomes_df["Start Date"] + hour_deltas,
            }
        )

        time_windows = [
            {"DAYS": 0, "MONTHS": 0, "YEARS": 0},
            {"DAYS": 1, "MONTHS": 0, "YEARS": 0},
        ]
        correct_df_rows = [[0], [0, 4, 5]]
        for time_before_start_date, correct_rows in zip(time_windows, correct_df_rows):
            masked_df = time_window_mask(
                outcomes_df,
                df,
                "TIME_COL",
                time_before_start_date,
                mask_end="Start Date",
            )
            self.assertTrue(df.iloc[correct_rows].equals(masked_df))
