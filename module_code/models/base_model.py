from typing import Callable, List, Union
from argparse import ArgumentParser, Namespace
from abc import ABC, ABCMeta, abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd


class AbstractModel(ABC):

    @abstractmethod
    def build_model(self,):
        pass

    @abstractmethod
    def configure_metrics(self, metric_names: List[str]) -> List[Callable]:
        """Pick metrics."""
        pass

    @staticmethod
    @abstractmethod
    def add_model_args(parent_parser: ArgumentParser) -> ArgumentParser:
        pass


class AbstractCRRTPredictor(TransformerMixin, BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def load_model(self, serialized_model_path: str) -> None:
        pass

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def transform(self, X: Union[np.ndarray, pd.DataFrame],) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs
    ) -> "AbstractCRRTPredictor":
        """
        Create an instance from CLI arguments.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
        # Ref: https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.8.3/pytorch_lightning/trainer/trainer.py#L750
        """
        pass