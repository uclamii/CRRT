import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np


class TestDataLoad(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        features = ["F1", "F2"]
        small_index = pd.MultiIndex.from_tuples(
            [(1, 1), (1, 2), (2, 1)], names=["pt_id", "day"]
        )
        data = [[1, 2], [np.nan, 3], [4, 5]]
        self.small_df = pd.DataFrame(data, index=small_index, columns=features)

    # TODO: write tests

    @patch("module_code.data.longitudinal_features.load_procedures")
    @patch("module_code.data.longitudinal_features.load_problems")
    @patch("module_code.data.longitudinal_features.load_medications")
    @patch("module_code.data.longitudinal_features.load_labs")
    @patch("module_code.data.longitudinal_features.load_vitals")
    @patch("module_code.data.longitudinal_features.load_diagnoses")
    @patch("module_code.data.longitudinal_features.load_outcomes")
    def test_merge_features_with_outcome(
        self,
        mock_outcomes,
        mock_diagnoses,
        mock_vitals,
        mock_labs,
        mock_medications,
        mock_problems,
        mock_procedures,
    ):
        """
        pt1: even features. pt2: odd features. pt3: all features.
        """
        mock_outcomes.return_value = pd.DataFrame()
        mock_diagnoses.return_value = pd.DataFrame()
        mock_vitals.return_value = pd.DataFrame()
        mock_labs.return_value = pd.DataFrame()
        mock_medications.return_value = pd.DataFrame()
        mock_problems.return_value = pd.DataFrame()
        mock_procedures.return_value = pd.DataFrame()
