from argparse import Namespace
from typing import List
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

        return pd.DataFrame(
            np.concatenate([data, outcome], axis=1),
            columns=feature_names + ["outcome"],
            index=self.multiindex_from_indices(indices),
        )

    def load_data_side_effect(self, *args, **kwargs):
        """Load depending on cohort, make up data."""
        # have same columns different indices. required to have outcome
        if args[-1] == "ucla_crrt" or kwargs.get("cohort", "") == "ucla_control":
            return self.create_mock_df(
                self.feature_names, list(range(1, self.nsamples, 2))
            )  # indices odd
        elif args[-1] == "ucla_control" or kwargs.get("cohort", "") == "ucla_control":
            return self.create_mock_df(
                self.feature_names, list(range(0, self.nsamples, 2))
            )  # indices even

    @patch("module_code.data.sklearn_loaders.load_data")
    def test_same_train_eval_cohorts(self, mock_load_data_fn):
        mock_load_data_fn.side_effect = self.load_data_side_effect

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

    @patch("module_code.data.sklearn_loaders.load_data")
    def test_different_train_eval_cohorts(self, mock_load_data):
        mock_load_data.side_effect = self.load_data_side_effect
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
        df = self.load_data_side_effect("ucla_control")
        for df1, df2 in [
            (df.drop(self.outcome_col_name, axis=1), data.test[0]),
            (df[self.outcome_col_name], data.test[1]),
        ]:
            np.testing.assert_array_equal(df1, df2)
