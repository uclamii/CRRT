from argparse import ArgumentParser, Namespace
import inspect
from typing import Callable, Optional, Tuple, Union, List
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# X, y
DataLabelTuple = Tuple[pd.DataFrame, pd.Series]


class CLIInitialized:
    @classmethod
    def from_argparse_args(
        cls,
        args: Union[Namespace, ArgumentParser],
        inner_classes: List = None,
        **kwargs,
    ):
        """
        Create an instance from CLI arguments.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
        If there are inner classes to instantiate and you need their signature, you can overload this method and pass a list of classes:
            e.g. return super().from_argparse_args(cls, args, [AEDitto], **kwargs)
        # Ref: https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.8.3/pytorch_lightning/trainer/trainer.py#L750
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid args, the rest may be user specific
        # returns a immutable dict MappingProxyType, want to combine so copy
        valid_kwargs = inspect.signature(cls.__init__).parameters.copy()
        if inner_classes is not None:
            for inner_class in inner_classes:  # Update with inner classes
                valid_kwargs.update(
                    inspect.signature(inner_class.__init__).parameters.copy()
                )
        data_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        data_kwargs.update(**kwargs)

        return cls(**data_kwargs)


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
