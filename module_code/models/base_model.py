from typing import Callable, List, Type, Union
from argparse import ArgumentParser
from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator

from data.base_loaders import CLIInitialized


class AbstractModel(ABC):
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def configure_metrics(self, metric_names: List[str]) -> List[Callable]:
        """Pick metrics."""
        pass

    @staticmethod
    @abstractmethod
    def add_model_args(parent_parsers: List[ArgumentParser]) -> ArgumentParser:
        pass


class BaseSklearnPredictor(TransformerMixin, BaseEstimator, CLIInitialized):
    @abstractmethod
    def load_model(self, serialized_model_path: str) -> None:
        pass
