from argparse import Namespace
import pandas as pd
import os
import torch

from data.pytorch_loaders import CRRTDataModule
from models.longitudinal_models import CRRTPredictor


def seed_everything(seed: int):
    """Sets seeds and also makes cuda deterministic for pytorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def continuous_learning(df: pd.DataFrame, args: Namespace):
    data = CRRTDataModule.from_argparse_args(df, args)
    data.setup()
    model = CRRTPredictor.from_argparse_args(args, nfeatures=data.nfeatures)
    model.fit(data)
