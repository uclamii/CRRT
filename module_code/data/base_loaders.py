from argparse import ArgumentParser, Namespace
import inspect
from typing import Callable, Optional, Tuple, Union
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from cli_utils import CLIInitialized

# X, y
DataLabelTuple = Tuple[pd.DataFrame, pd.Series]


class AbstractCRRTDataModule(ABC, CLIInitialized):
    @abstractmethod
    def get_post_split_transform(self, train: DataLabelTuple) -> Callable:
        pass

    @abstractmethod
    def split_dataset(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> Tuple[DataLabelTuple, DataLabelTuple, DataLabelTuple]:
        pass

    @staticmethod
    @abstractmethod
    def add_data_args(parent_parser: ArgumentParser) -> ArgumentParser:
        pass


class CRRTDataset:
    def __init__(
        self,
        split_data: DataLabelTuple,
        transform: Optional[Callable] = None,
    ) -> None:
        # split can be data from train, val, or test
        self.split_data = split_data
        self.transform = transform

    def __len__(self):
        return len(self.split_data[1])

    def __getitem__(self, index: int):
        X = self.split_data[0][index]
        y = self.split_data[1][index]
        if self.transform:
            X = self.transform(X)
        return (X, y)
