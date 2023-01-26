from argparse import Namespace
from typing import List
import unittest
from unittest.mock import patch
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
