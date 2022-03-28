import inspect
from typing import Callable, List, Type, Union
from argparse import ArgumentParser, Namespace
from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator


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
    def add_model_args(parent_parsers: List[ArgumentParser]) -> ArgumentParser:
        pass


class BaseSklearnPredictor(TransformerMixin, BaseEstimator):
    @abstractmethod
    def load_model(self, serialized_model_path: str) -> None:
        pass

    @classmethod
    def from_argparse_args(
        cls,
        wrapped_model_class: Type[AbstractModel],
        args: Union[Namespace, ArgumentParser],
        **kwargs
    ) -> "BaseSklearnPredictor":
        """
        Create an instance from CLI arguments.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
        # Ref: https://github.com/PyTorchLightning/PyTorch-Lightning/blob/-2.8.3/pytorch_lightning/trainer/trainer.py#L750
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid args, the rest may be user specific
        # returns a immutable dict MappingProxyType, want to combine so copy
        valid_kwargs = inspect.signature(cls.__init__).parameters.copy()
        # sklearn predictor wraps some model inside of it, we want to include its params
        valid_kwargs.update(
            inspect.signature(wrapped_model_class.__init__).parameters.copy()
        )
        data_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        data_kwargs.update(**kwargs)

        return cls(**data_kwargs)
