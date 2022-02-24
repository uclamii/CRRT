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
    def from_argparse_args(
        cls,
        preprocessed_df: np.ndarray,
        args: Union[Namespace, ArgumentParser],
        **kwargs
    ) -> "CRRTDataModule":
        """
        Create an instance from CLI arguments.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
        # Ref: https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.8.3/pytorch_lightning/trainer/trainer.py#L750
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        data_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        data_kwargs.update(**kwargs)

        return cls(preprocessed_df, **data_kwargs)


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
