from argparse import ArgumentParser, Namespace
import inspect
from typing import Callable, Optional, Tuple, Union
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

# X, y
SplitDataTuple = Tuple[pd.DataFrame, pd.Series]


class AbstractCRRTDataModule(ABC):

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        pass

    @abstractmethod
    def get_post_split_transform(self, train: SplitDataTuple) -> Callable:
        pass

    @abstractmethod
    def split_dataset(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
    ) -> Tuple[SplitDataTuple, SplitDataTuple, SplitDataTuple]:
        pass

    @staticmethod
    @abstractmethod
    def add_data_args(parent_parser: ArgumentParser) -> ArgumentParser:
        pass

    @classmethod
    @abstractmethod
    def add_data_args(parent_parser: ArgumentParser) -> ArgumentParser:
        pass

    @classmethod
    @abstractmethod
    def from_argparse_args(cls,
        preprocessed_df: np.ndarray,
        args: Union[Namespace, ArgumentParser],
        **kwargs):
        pass


class CRRTDataset:

    def __init__(
        self, split: SplitDataTuple, data_transform: Optional[Callable] = None,
            ft_select_transform: Optional[Callable] = None) -> None:
        self.split = split
        self.data_transform = data_transform
        self.ft_select_transform = ft_select_transform

    def __len__(self):
        return len(self.split[1])

    def __getitem__(self, index: int):
        X = self.split[0][index]
        y = self.split[1][index]
        if self.data_transform:
            X = self.data_transform(X)
        if self.ft_select_transform:
            X = self.ft_select_transform(X)
        return (X, y)
