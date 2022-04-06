from argparse import Namespace

from data.torch_loaders import TorchCRRTDataModule
from models.longitudinal_models import CRRTDynamicPredictor
import pandas as pd


def continuous_learning(df: pd.DataFrame, args: Namespace):
    data = TorchCRRTDataModule.from_argparse_args(df, args)
    data.setup()
    model = CRRTDynamicPredictor.from_argparse_args(args, nfeatures=data.nfeatures)
    model.fit(data)
