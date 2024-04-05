"""

Testing of dataloaders

"""

from argparse import Namespace
from typing import Dict, List
import unittest
from unittest.mock import patch
from numpy.random import default_rng
import numpy as np
import pandas as pd

from module_code.data.sklearn_loaders import SklearnCRRTDataModule

SEED = 0


class TestSklearnLoaders(unittest.TestCase):
    # TODO[LOW]: test auxiliary info like coulmns and ctn_columns
    # TODO: test reference_ids
    # TODO: test reference_cols
    def setUp(self) -> None:
        super().setUp()
        self.nsamples = 10
        self.feature_names = ["f1", "f2"]
        self.outcome_col_name = "outcome"
        self.val_split_size = 0.5
        self.test_split_size = 0.5
        pre_start_delta = {"YEARS": 1, "MONTHS": 3, "DAYS": 14}
        post_start_delta = {"YEARS": 0, "MONTHS": 1, "DAYS": 7}
        self.args = Namespace(
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
            preselect_features=[],
            new_eval_cohort=False,
        )

    def multiindex_from_indices(self, indices: List[int]) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples(
            [(idx, 0) for idx in indices], names=["IP_PATIENT_ID", "Start Date"]
        )

    def create_mock_df(
        self, feature_names: List[str], indices: List[int]
    ) -> pd.DataFrame:
        rng = default_rng(SEED)
        nsamples = len(indices)
        nfeatures = len(feature_names)
        data = rng.integers(0, 100, (nsamples, nfeatures))
        outcome = rng.integers(0, 1, (nsamples, 1))

        # pd.DataFrame(np.concatenate([data,outcome],axis=1), columns=['f1','f2','outcome'], index = pd.MultiIndex.from_tuples([(idx, 0) for idx in [0,2,4,6,8]], names=["IP_PATIENT_ID", "Start Date"]))

        return pd.DataFrame(
            np.concatenate([data, outcome], axis=1),
            columns=feature_names + ["outcome"],
            index=self.multiindex_from_indices(indices),
        )

    def load_data_side_effect(self, feature_names: List[List[str]], nsamples: int):
        def load_data(*args, **kwargs):
            """Load depending on cohort, make up data."""
            # have same columns different indices. required to have outcome
            if args[-1] == "ucla_crrt" or kwargs.get("cohort", "") == "ucla_crrt":
                return (
                    self.create_mock_df(feature_names[0], list(range(1, nsamples, 2))),
                    feature_names[0],
                )  # indices odd
            elif (
                args[-1] == "ucla_control" or kwargs.get("cohort", "") == "ucla_control"
            ):
                return (
                    self.create_mock_df(feature_names[1], list(range(0, nsamples, 2))),
                    feature_names[1],
                )  # indices even

        return load_data

    @patch(
        "module_code.data.sklearn_loaders.SklearnCRRTDataModule.get_post_split_transform"
    )
    @patch("module_code.data.sklearn_loaders.load_data")
    def test_mixed_train_eval_cohorts(self, mock_load_data, mock_transforms):
        # They have shared and unique columns
        ucla_crrt_cols = ["f1", "f2"]
        ucla_control_cols = ["f2", "f3"]
        all_cols = ["f1", "f2", "f3"]

        mock_load_data.side_effect = self.load_data_side_effect(
            [ucla_crrt_cols, ucla_control_cols],
            self.nsamples,
        )

        data = SklearnCRRTDataModule(
            SEED,
            outcome_col_name=self.outcome_col_name,
            train_val_cohort="ucla_crrt+ucla_control",
            eval_cohort="ucla_crrt+ucla_control",
            val_split_size=self.val_split_size,
            test_split_size=self.test_split_size,
        )

        # default. no selection or reference. should be union
        with self.subTest("Different Columns Union"):
            mock_transforms.side_effect = lambda x: lambda y: y

            data.setup(self.args)

            # should be outer join
            np.testing.assert_array_equal(all_cols, data.test[0].columns)
            np.testing.assert_array_equal(
                data.test[0].shape[0], self.nsamples * self.test_split_size
            )

        # select feature in only one of the datasets
        with self.subTest("Different Columns With Single Feature Selection"):
            mock_transforms.side_effect = lambda x: lambda y: y[["f1"]]

            data.setup(self.args)

            # should only have selected feature
            np.testing.assert_array_equal(["f1"], data.test[0].columns)
            np.testing.assert_array_equal(
                data.test[0].shape[0], self.nsamples * self.test_split_size
            )

        # select features that occur in each dataset uniquely
        with self.subTest("Different Columns With Multiple Feature Selection"):
            mock_transforms.side_effect = lambda x: lambda y: y[["f1", "f3"]]

            data.setup(self.args)

            # should only have selected feature
            np.testing.assert_array_equal(["f1", "f3"], data.test[0].columns)
            np.testing.assert_array_equal(
                data.test[0].shape[0], self.nsamples * self.test_split_size
            )

        # select features that occur in dataset intersection
        with self.subTest("Intersect Columns With Multiple Feature Selection"):
            mock_transforms.side_effect = lambda x: lambda y: y[["f2"]]

            data.setup(self.args)

            # should only have selected feature
            np.testing.assert_array_equal(["f2"], data.test[0].columns)
            np.testing.assert_array_equal(
                data.test[0].shape[0], self.nsamples * self.test_split_size
            )

        # select single feature with reference columns
        with self.subTest("Different Columns Reference Cols"):
            mock_transforms.side_effect = lambda x: lambda y: y

            data.setup(self.args, reference_cols=["f1"])

            # should only have reference feature
            np.testing.assert_array_equal(["f1"], data.test[0].columns)
            np.testing.assert_array_equal(
                data.test[0].shape[0], self.nsamples * self.test_split_size
            )

        # select multiple features with reference columns
        with self.subTest("Different Columns Reference Cols Mixed"):
            mock_transforms.side_effect = lambda x: lambda y: y

            data.setup(self.args, reference_cols=["f1", "f3"])

            # should only have reference feature
            np.testing.assert_array_equal(["f1", "f3"], data.test[0].columns)
            np.testing.assert_array_equal(
                data.test[0].shape[0], self.nsamples * self.test_split_size
            )

        # select non-existing with reference
        with self.subTest("Different Columns Non-existing Reference Cols Mixed"):
            mock_transforms.side_effect = lambda x: lambda y: y

            data.setup(self.args, reference_cols=["f4"])

            # should only have reference feature
            np.testing.assert_array_equal(["f4"], data.test[0].columns)
            np.testing.assert_array_equal(
                [np.nan] * int(self.nsamples * self.test_split_size),
                data.test[0]["f4"].to_list(),
            )
            np.testing.assert_array_equal(
                data.test[0].shape[0], self.nsamples * self.test_split_size
            )

    @patch("module_code.data.sklearn_loaders.load_data")
    def test_same_train_eval_cohorts(self, mock_load_data_fn):
        mock_load_data_fn.side_effect = self.load_data_side_effect(
            [self.feature_names, self.feature_names], self.nsamples
        )

        with self.assertRaises(AssertionError):  # need test_split_size
            SklearnCRRTDataModule(
                SEED,
                outcome_col_name=self.outcome_col_name,
                train_val_cohort="ucla_crrt",
                eval_cohort="ucla_crrt",
                val_split_size=self.val_split_size,
            )
        data = SklearnCRRTDataModule(
            SEED,
            outcome_col_name=self.outcome_col_name,
            train_val_cohort="ucla_crrt",
            eval_cohort="ucla_crrt",
            val_split_size=self.val_split_size,
            test_split_size=self.test_split_size,
        )
        data.setup(self.args)
        for split in ["train", "val", "test"]:  # crrt is all odd indices
            self.assertTrue(all(data.split_pt_ids[split] % 2 == 1))

    @patch(
        "module_code.data.sklearn_loaders.SklearnCRRTDataModule.get_post_split_transform"
    )
    @patch("module_code.data.sklearn_loaders.load_data")
    def test_different_train_eval_cohorts(self, mock_load_data, mock_transforms):
        mock_load_data.side_effect = self.load_data_side_effect(
            [self.feature_names, self.feature_names], self.nsamples
        )
        mock_transforms.side_effect = lambda x: lambda y: y
        data = SklearnCRRTDataModule(
            SEED,
            outcome_col_name=self.outcome_col_name,
            train_val_cohort="ucla_crrt",
            eval_cohort="ucla_control",
            val_split_size=self.val_split_size,
        )
        data.setup(self.args)
        for split in ["train", "val"]:  # crrt is all odd indices
            self.assertTrue(all(data.split_pt_ids[split] % 2 == 1))
        # control is even indices
        self.assertTrue(all(data.split_pt_ids["test"] % 2 == 0))

        # the entire test X, y should be the entire ucla control
        df, _ = self.load_data_side_effect(
            [self.feature_names, self.feature_names], self.nsamples
        )("ucla_control")
        for df1, df2 in [
            (df.drop(self.outcome_col_name, axis=1), data.test[0]),
            (df[self.outcome_col_name], data.test[1]),
        ]:
            np.testing.assert_array_equal(df1, df2)

        with self.subTest("Different Columns"):
            # Keep only one column. add one new column
            ucla_control_cols = self.feature_names[:1] + ["somerandomcolumn"]
            all_cols = self.feature_names + ["somerandomcolumn"]
            mock_load_data.side_effect = self.load_data_side_effect(
                [self.feature_names, ucla_control_cols],
                self.nsamples,
            )
            data = SklearnCRRTDataModule(
                SEED,
                outcome_col_name=self.outcome_col_name,
                train_val_cohort="ucla_crrt",
                eval_cohort="ucla_control",
                val_split_size=self.val_split_size,
            )
            data.setup(self.args)
            for split in ["train", "val"]:  # crrt is all odd indices
                self.assertTrue(all(data.split_pt_ids[split] % 2 == 1))
            # control is even indices
            self.assertTrue(all(data.split_pt_ids["test"] % 2 == 0))

            # should be outer join
            np.testing.assert_array_equal(all_cols, data.test[0].columns)

    @patch(
        "module_code.data.sklearn_loaders.SklearnCRRTDataModule.get_post_split_transform"
    )
    @patch("module_code.data.sklearn_loaders.load_data")
    def test_filters(self, mock_load_data, mock_transforms):
        mock_load_data.side_effect = self.load_data_side_effect(
            [self.feature_names, self.feature_names], self.nsamples
        )
        # override transform since filtering happens BEFORE transformation
        mock_transforms.side_effect = lambda x: lambda y: y

        data = SklearnCRRTDataModule(
            SEED,
            outcome_col_name=self.outcome_col_name,
            train_val_cohort="ucla_crrt",
            eval_cohort="ucla_control",
            # val_split_size=self.val_split_size,
            val_split_size=0.01,
        )
        data.setup(self.args)
        train_df = data.train[0].to_numpy()

        def test_filters_equal(
            true_filter: Dict[str, np.ndarray], est_filter: Dict[str, np.ndarray]
        ):
            self.assertEqual(true_filter.keys(), est_filter.keys())
            for true, est in zip(true_filter.values(), est_filter.values()):
                np.testing.assert_array_equal(true, est)

        with self.subTest("Single Column + Exact"):
            # 0,0 : first row, first column
            data.filters = {"group": {"filter": ("f1", train_df[0, 0])}}
            data.setup(self.args)

            # should just keep the first value
            f = np.full(train_df.shape[0], False, dtype=bool)
            f[0] = True
            true_train_filter = {"filter": f}
            test_filters_equal(true_train_filter, data.train_filters)

        with self.subTest("Single Column + Range"):
            # 0, [min, max+1) -> first column, keep the whole range
            data.filters = {
                "group": {
                    "filter": ("f1", (train_df[:, 0].min(), train_df[:, 0].max() + 1))
                }
            }
            data.setup(self.args)

            # should just keep all (since we're keeping the whole range)
            true_train_filter = {"filter": np.full(train_df.shape[0], True, dtype=bool)}
            test_filters_equal(true_train_filter, data.train_filters)

            # 0, [max+1, max+2) -> first column, include nothing
            maxval = train_df[:, 0].max()
            data.filters = {"group": {"filter": ("f1", (maxval + 1, maxval + 2))}}
            data.setup(self.args)

            # should just keep all (since we're keeping the whole range)

            true_train_filter = {
                "filter": np.full(train_df.shape[0], False, dtype=bool)
            }
            test_filters_equal(true_train_filter, data.train_filters)

        with self.subTest("Multiple Column + Exact"):
            # match first column first row, but not the second (so nothing should match)
            data.filters = {
                "group": {"filter": (["f1", "f2"], [train_df[0, 0], -train_df[1, 0]])}
            }
            data.setup(self.args)

            true_train_filter = {
                "filter": np.full(train_df.shape[0], False, dtype=bool)
            }
            test_filters_equal(true_train_filter, data.train_filters)

            # match first column first row, AND the second (so one should match)
            data.filters = {
                "group": {"filter": (["f1", "f2"], [train_df[0, 0], train_df[0, 1]])}
            }
            data.setup(self.args)

            f = np.full(train_df.shape[0], False, dtype=bool)
            f[0] = True
            true_train_filter = {"filter": f}
            test_filters_equal(true_train_filter, data.train_filters)

        with self.subTest("Multiple Filters"):
            data.filters = {
                "group": {
                    "filter1": ("f1", train_df[0, 0]),
                    "filter2": ("f2", train_df[0, 1]),
                }
            }
            data.setup(self.args)

            # should just keep the first value
            f = np.full(train_df.shape[0], False, dtype=bool)
            f[0] = True
            true_train_filter = {"filter1": f, "filter2": f}
            test_filters_equal(true_train_filter, data.train_filters)

            data.filters = {
                "group": {
                    "filter1": ("f1", train_df[0, 0]),
                    "filter2": ("f2", train_df[1, 1]),
                }
            }
            data.setup(self.args)

            # should just keep the first value
            f1 = np.full(train_df.shape[0], False, dtype=bool)
            f2 = f1.copy()
            f1[0] = True
            f2[1] = True

            true_train_filter = {"filter1": f1, "filter2": f2}
            test_filters_equal(true_train_filter, data.train_filters)
