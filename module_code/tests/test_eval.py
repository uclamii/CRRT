from argparse import Namespace
from typing import List
import unittest
from unittest.mock import MagicMock, patch
import unittest.mock as mock
from numpy.random import default_rng
import numpy as np
import pandas as pd

from module_code.data.sklearn_loaders import SklearnCRRTDataModule
from module_code.evaluate.explainability import shap_explainability
from module_code.models.static_models import CRRTStaticPredictor
from module_code.evaluate.utils import log_figure

SEED = 0


class TestEvalModel(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.nsamples = 100
        self.feature_names = ["f1", "f2"]
        self.outcome_col_name = "outcome"
        self.val_split_size = 0.5
        self.test_split_size = 0.5
        self.data_args = {
            "pre_start_delta": {},
            "post_start_delta": {},
            "time_window_end": "",
            "slide_window_by": 0,
            "serialization": "parquet",
            "ucla_crrt_data_dir": "",
            # these don't matter for these tests but need to exist
            "model_type": "",
            "max_days_on_crrt": 0,
            "time_interval": None,
            "preprocessed_df_file": None,
            "preselect_features": False,
        }
        self.data = SklearnCRRTDataModule(
            SEED,
            outcome_col_name=self.outcome_col_name,
            train_val_cohort="ucla_crrt",
            eval_cohort="ucla_crrt",
            val_split_size=self.val_split_size,
            test_split_size=self.test_split_size,
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
        return self.create_mock_df(
            self.feature_names, list(range(1, self.nsamples, 2))
        )  # indices odd

    def do_nothing_side_effect(self, *args, **kwargs):
        return

    @patch("module_code.data.sklearn_loaders.load_data")
    def test_explain(self, mock_load_data_fn):
        mock_load_data_fn.side_effect = self.load_data_side_effect

        with self.subTest("SHAP"):
            args = Namespace(
                seed=SEED,
                modeln="xgb",
                metric_names=[],
                curve_names=[],
                plot_names=["shap_explain"],
                top_k_feature_importance=5,
                model_kwargs={},
                **self.data_args,
            )
            self.data.setup(args)

            model = CRRTStaticPredictor.from_argparse_args(args)
            model.fit(self.data)

            self.data.train_filters = None
            model.static_model.use_shap_for_feature_importance = False

            model.evaluate(self.data, "train")

            # TODO: idk why the # of calls is 0, it looks like patch isn't working the way I am expecting it to.
            with self.subTest("With Feature Importance"):
                model.static_model.use_shap_for_feature_importance = True
                with patch(  # ref: https://stackoverflow.com/a/55960830
                    "module_code.evaluate.explainability.log_figure", wraps=log_figure
                ) as mock_log_fig:
                    model.evaluate(self.data, "train")
                    # called 3 times: waterfall, beeswarm, bar
                    # self.assertGreaterEqual(3, mock_log_fig.call_count)

    @patch("module_code.data.sklearn_loaders.load_data")
    def test_curves(self, mock_load_data_fn):
        mock_load_data_fn.side_effect = self.load_data_side_effect

        with self.subTest("SHAP"):
            args = Namespace(
                seed=SEED,
                modeln="xgb",
                metric_names=[],
                curve_names=["calibration_curve"],
                plot_names=[],
                top_k_feature_importance=5,
                model_kwargs={},
                **self.data_args,
            )
            self.data.setup(args)

            model = CRRTStaticPredictor.from_argparse_args(args)
            model.fit(self.data)

            self.data.train_filters = None
            model.evaluate(self.data, "train")

    @patch("module_code.models.static_models.PLOT_MAP")
    @patch("module_code.data.sklearn_loaders.load_data")
    def test_eval_conditions(self, mock_load_data_fn, mock_plot_map):
        mock_load_data_fn.side_effect = self.load_data_side_effect
        true_plot_map = {
            "randomness": MagicMock(name="model_randomness"),
            "shap_explain": MagicMock(name="shap_explain"),
            "roc_curve": MagicMock(name="roc_curve"),
        }
        mock_plot_map.__getitem__.side_effect = true_plot_map.__getitem__
        mock_plot_map.items.side_effect = true_plot_map.items

        with self.subTest("SHAP"):
            args = Namespace(
                seed=SEED,
                modeln="xgb",
                metric_names=[],
                curve_names=[],
                plot_names=["randomness", "shap_explain", "roc_curve"],
                top_k_feature_importance=5,
                model_kwargs={},
                **self.data_args,
            )
            data = SklearnCRRTDataModule(
                SEED,
                outcome_col_name=self.outcome_col_name,
                train_val_cohort="ucla_crrt",
                eval_cohort="ucla_crrt",
                val_split_size=self.val_split_size,
                test_split_size=self.test_split_size,
                filters={"group": {"f": (self.feature_names[0], (50, 100))}},
            )
            data.setup(args)

            model = CRRTStaticPredictor.from_argparse_args(args)
            model.fit(data)

            # should not be called in train and val
            model.evaluate(data, "train")
            self.assertEqual(0, true_plot_map["randomness"].call_count)
            self.assertEqual(0, true_plot_map["shap_explain"].call_count)
            self.assertEqual(0, true_plot_map["roc_curve"].call_count)
            model.evaluate(data, "val")
            self.assertEqual(0, true_plot_map["randomness"].call_count)
            self.assertEqual(0, true_plot_map["shap_explain"].call_count)
            self.assertEqual(0, true_plot_map["roc_curve"].call_count)

            model.evaluate(data, "test")
            # this is model level should only be called once
            self.assertEqual(1, true_plot_map["randomness"].call_count)
            self.assertEqual(1, true_plot_map["roc_curve"].call_count)
            # this is not model level, 1 for model, 1 for the 1 subgroup filter
            self.assertEqual(2, true_plot_map["shap_explain"].call_count)
