from argparse import Namespace

from data.standard_loaders import SklearnCRRTDataModule
from models.static_models import CRRTStaticPredictor
import pandas as pd


def static_learning(df: pd.DataFrame, args: Namespace):
    # need to update CRRTDataModule
    data = SklearnCRRTDataModule.from_argparse_args(df, args)
    data.setup()
    # Then need to update the predictor
    model = CRRTStaticPredictor.from_argparse_args(args)
    model.fit(data)
