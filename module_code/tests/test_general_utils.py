from argparse import Namespace
import unittest
import unittest.mock as mock
from os.path import join
import pandas as pd

from module_code.data.utils import get_preprocessed_file_name
from module_code.data.load import (
    process_and_serialize_raw_data,
    get_preprocessed_df_path,
)


class TestLoading(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    @mock.patch("module_code.data.load.DataFrame.to_parquet")
    def test_different_cohorts(self, mock_serialization):  # with everything turned on
        # Ensure serialization doesn't happen
        mock_serialization.return_value = None

        pre_start_delta = {"YEARS": 1, "MONTHS": 3, "DAYS": 14}
        post_start_delta = {"YEARS": 0, "MONTHS": 1, "DAYS": 7}
        args = Namespace(
            pre_start_delta=pre_start_delta,
            post_start_delta=post_start_delta,
            time_window_end="Start Date",
            slide_window_by=1,
            serialization="parquet",
            ucla_control_data_dir="ucla_control",
            ucla_crrt_data_dir="ucla_crrt",
            # these don't matter for these tests but need to exist
            model_type="",
            max_days_on_crrt=0,
            time_interval=None,
            preprocessed_df_file=None,
        )

        cohorts = ["ucla_crrt", "ucla_control"]
        correct_universal_fname = "df_[startdate+1-1y3m14d,startdate+1m7d+1].parquet"
        for cohort in cohorts:
            with self.subTest(cohort):
                path = get_preprocessed_df_path(args, cohort)
                self.assertEqual(path, join(cohort, correct_universal_fname))

                # ensure the desired function is called
                with mock.patch(
                    f"module_code.data.load.preproc_{cohort}"
                ) as mock_cohort_load_fn:
                    mock_cohort_load_fn.return_value = pd.DataFrame()
                    process_and_serialize_raw_data(args, path, cohort=cohort)
                    mock_cohort_load_fn.assert_called_once()

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
