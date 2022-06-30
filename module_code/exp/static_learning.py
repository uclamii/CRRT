from argparse import Namespace
import pandas as pd

from data.sklearn_loaders import SklearnCRRTDataModule
from models.static_models import CRRTStaticPredictor
from data.utils import get_pt_type_indicators


def static_learning(df: pd.DataFrame, args: Namespace):
    # need to update CRRTDataModule
    filters = {
        "heart": lambda df: df["heart_pt_indicator"] == 1,
        "liver": lambda df: df["liver_pt_indicator"] == 1,
        "infection": lambda df: df["infection_pt_indicator"] == 1,
    }
    data = SklearnCRRTDataModule.from_argparse_args(df, args, filters=filters)
    data.setup()
    # Then need to update the predictor
    model = CRRTStaticPredictor.from_argparse_args(args)
    model.fit(data)

    model.static_model.curve_names = None
    model.static_model.error_analysis = None
    model.static_model.top_k_feature_importance = None
    return model.evaluate("val")
    # model.evaluate("train", filters)
    # model.evaluate("test", filters)
