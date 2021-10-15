import unittest
import pandas as pd
from datetime import timedelta

from module_code.data.longitudinal_utils import time_window_mask


class TestWindowMask(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

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
