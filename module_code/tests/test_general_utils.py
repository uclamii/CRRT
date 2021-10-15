import unittest

from module_code.utils import get_preprocessed_file_name


class TestLoading(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_get_preprocessed_file_name(self):
        time_interval = "1D"
        time_before_start_date = {"YEARS": 0, "MONTHS": 0, "DAYS": 14}
        time_window_end = "End Date"

        name = "df_1Dagg_14d_enddate.feather"

        self.assertEqual(
            get_preprocessed_file_name(
                time_before_start_date, time_interval, time_window_end
            ),
            name,
        )

        self.assertEqual(
            get_preprocessed_file_name(
                time_before_start_date, time_interval, time_window_end, "override_name"
            ),
            "override_name",
        )

        self.assertEqual(
            get_preprocessed_file_name(preprocessed_df_file="override_name"),
            "override_name",
        )

        time_before_start_date = {"YEARS": 1, "MONTHS": 3, "DAYS": 14}
        name = "df_1Dagg_1y3m14d_enddate.feather"

        self.assertEqual(
            get_preprocessed_file_name(
                time_before_start_date, time_interval, time_window_end
            ),
            name,
        )
