import unittest

from module_code.utils import get_preprocessed_file_name


class TestLoading(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_get_preprocessed_file_name(self):
        time_interval = "1D"
        pre_start_delta = {"YEARS": 0, "MONTHS": 0, "DAYS": 14}
        time_window_end = "End Date"
        # test pre_start_delta but no post delta
        self.assertEqual(
            get_preprocessed_file_name(
                pre_start_delta, None, time_interval, time_window_end
            ),
            "df_1Dagg_[startdate-14d,enddate].feather",
        )

        # Test override with other args
        self.assertEqual(
            get_preprocessed_file_name(
                pre_start_delta,
                None,
                time_interval,
                time_window_end,
                preprocessed_df_file="override_name",
            ),
            "override_name.feather",
        )

        # test override with only override arg
        self.assertEqual(
            get_preprocessed_file_name(preprocessed_df_file="override_name"),
            "override_name.feather",
        )

        # test all years, months, and days specified, test no years specified
        # test both pre_start and post_start deltas
        pre_start_delta = {"YEARS": 1, "MONTHS": 3, "DAYS": 14}
        post_start_delta = {"YEARS": 0, "MONTHS": 1, "DAYS": 7}
        self.assertEqual(
            get_preprocessed_file_name(
                pre_start_delta, post_start_delta, time_interval, time_window_end
            ),
            "df_1Dagg_[startdate-1y3m14d,startdate+1m7d].feather",
        )

        # test neither deltas specified
        self.assertEqual(
            get_preprocessed_file_name(None, None, time_interval, time_window_end),
            "df_1Dagg_[startdate,enddate].feather",
        )

        # Test slide_window_by
        with self.subTest("slide_window_by"):
            # Test null and 0
            self.assertEqual(
                get_preprocessed_file_name(
                    None, None, time_interval, time_window_end, 0
                ),
                "df_1Dagg_[startdate,enddate].feather",
            )
            self.assertEqual(
                get_preprocessed_file_name(
                    None, None, time_interval, time_window_end, None
                ),
                "df_1Dagg_[startdate,enddate].feather",
            )

            # No deltas
            self.assertEqual(
                get_preprocessed_file_name(
                    None, None, time_interval, time_window_end, 1
                ),
                "df_1Dagg_[startdate+1,enddate+1].feather",
            )
            # Both deltas
            pre_start_delta = {"YEARS": 1, "MONTHS": 3, "DAYS": 14}
            post_start_delta = {"YEARS": 0, "MONTHS": 1, "DAYS": 7}
            self.assertEqual(
                get_preprocessed_file_name(
                    pre_start_delta, post_start_delta, time_interval, time_window_end, 1
                ),
                "df_1Dagg_[startdate+1-1y3m14d,startdate+1m7d+1].feather",
            )

            # test pre_start_delta but no post delta
            self.assertEqual(
                get_preprocessed_file_name(
                    pre_start_delta, None, time_interval, "Start Date", 1
                ),
                "df_1Dagg_[startdate+1-1y3m14d,startdate+1].feather",
            )
