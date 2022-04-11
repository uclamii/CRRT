from argparse import Namespace
import pandas as pd

from data.sklearn_loaders import SklearnCRRTDataModule
from models.static_models import CRRTStaticPredictor
from data.load import get_pt_type_indicators


def static_learning(df: pd.DataFrame, args: Namespace):
    # need to update CRRTDataModule
    data = SklearnCRRTDataModule.from_argparse_args(df, args)
    data.setup()
    # Then need to update the predictor
    model = CRRTStaticPredictor.from_argparse_args(args)
    model.fit(data)

    df = get_pt_type_indicators(df)
    filters = {
        "heart": lambda df: df["heart_pt_indicator"] == 1,
        "liver": lambda df: df["liver_pt_indicator"] == 1,
        "infection": lambda df: df["infection_pt_indicator"] == 1,
    }

    # TODO: mlflow autolog is supposed to take care of train, but isn't for some reason, so we do it manually here for now.
    model.evaluate("train", filters)
    model.evaluate("val", filters)
    model.evaluate("test", filters)
