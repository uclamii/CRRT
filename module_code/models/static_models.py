from argparse import ArgumentParser, Namespace
import inspect
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    average_precision_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

from data.standard_loaders import StdCRRTDataModule
from data.argparse_utils import YAMLStringListToList

from exp.utils import seed_everything
from models.base_model import AbstractCRRTPredictor, AbstractModel


alg_map = {
    "lgr": LogisticRegression,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "nb": MultinomialNB,
    "dt": DecisionTreeClassifier,
    "rf": RandomForestClassifier,
    "lgb": LGBMClassifier,
    "xgb": XGBClassifier
}

metric_map = {
    "auroc": lambda gt, pred_probs, decision_thresh: roc_auc_score(gt, pred_probs),
    "ap": lambda gt, pred_probs, decision_thresh: average_precision_score(
        gt, pred_probs
    ),
    "brier": lambda gt, pred_probs, decision_thresh: brier_score_loss(gt, pred_probs),
    "accuracy": lambda gt, pred_probs, decision_thresh: accuracy_score(
        gt, (pred_probs >= decision_thresh).astype(int)
    ),
    "f1": lambda gt, pred_probs, decision_thresh: f1_score(
        gt, (pred_probs >= decision_thresh).astype(int)
    ),
    "recall": lambda gt, pred_probs, decision_thresh: recall_score(
        gt, (pred_probs >= decision_thresh).astype(int)
    ),
    "specificity": lambda gt, pred_probs, decision_thresh: recall_score(
        gt, (pred_probs >= decision_thresh).astype(int), pos_label=0
    ),
    "precision": lambda gt, pred_probs, decision_thresh: precision_score(
        gt, (pred_probs >= decision_thresh).astype(int)
    ),
    "conf_matrix": lambda gt, pred_probs, decision_thresh: confusion_matrix(
        gt, (pred_probs >= decision_thresh).astype(int)
    ),
    "TN": lambda gt, pred_probs, decision_thresh: confusion_matrix(
        gt, (pred_probs >= decision_thresh).astype(int)
    )[0, 0],
    "FN": lambda gt, pred_probs, decision_thresh: confusion_matrix(
        gt, (pred_probs >= decision_thresh).astype(int)
    )[1, 0],
    "TP": lambda gt, pred_probs, decision_thresh: confusion_matrix(
        gt, (pred_probs >= decision_thresh).astype(int)
    )[1, 1],
    "FP": lambda gt, pred_probs, decision_thresh: confusion_matrix(
        gt, (pred_probs >= decision_thresh).astype(int)
    )[0, 1],
}


class StaticModel(AbstractModel):
    def __init__(
        self,
        seed: int,
        modeln: str,
        metrics: List[str],
        **model_kwargs
    ):
        super().__init__()
        self.seed = seed,
        self.modeln = modeln
        self.model_kwargs = model_kwargs
        self.model = self.build_model()
        self.metrics = self.configure_metrics(metrics)

    def build_model(self,):
        if self.modeln in alg_map:
            model_cls = alg_map[self.modeln]
        else:
            raise ValueError("The {} is not a valid model type".format(self.modeln))
        return model_cls(**self.model_kwargs)

    def configure_metrics(self, metric_names: List[str]) -> List[Callable]:
        """Pick metrics."""
        for metric in metric_names:
            assert metric in metric_map, (
                f"{metric} is not valid metric name."
                " Must match a key in `metric_map`"
            )
        return [metric_map[metric] for metric in metric_names]

    @staticmethod
    def add_model_args(parent_parser: ArgumentParser) -> ArgumentParser:
        p = ArgumentParser(parents=[parent_parser], add_help=False)
        p.add_argument(
            "--modeln",
            type=str,
            default="lgr",
            choices=["lgr", "svm", "knn", "nb", "dt", "rf", "lgb", "xgb"],
            help="Name of model to use for learning.",
        )
        p.add_argument(
            "--metrics",
            type=str,
            action=YAMLStringListToList(str),
            help="(List of comma-separated strings) Name of Pytorch Metrics from torchmetrics.",
        )
        # TODO: mechanism for incorporating any additional arguments? Actually, Can just be passed by YAML? Can test later
        return p


class CRRTStaticPredictor(AbstractCRRTPredictor):
    """
    Wrapper predictor class, compatible with sklearn.
    Uses longitudinal model to do time series classification on tabular data.
    Implements fit and transform.
    """

    def __init__(
        self, **kwargs,
    ):
        self.static_model = StaticModel(**kwargs)


    # TODO: This needs to be changed for the serialization of the static models
    def load_model(self, serialized_model_path: str) -> None:
        """Loads the underlying autoencoder state dict from path."""
        self.static_model.load_state_dict(load(serialized_model_path))

    # TODO: this needs to be changed for sklearn-type models
    def fit(self, data: StdCRRTDataModule):
        """Trains the autoencoder for imputation."""
        seed_everything(self.seed)
        self.data = data
        # self.data.setup()

        self.trainer.fit(self.longitudinal_model, datamodule=self.data)
        if self.runtest:
            self.trainer.test(self.longitudinal_model, datamodule=self.data)

        return self

    # TODO: this also needs to be changed for static model
    def transform(self, X: Union[np.ndarray, pd.DataFrame],) -> np.ndarray:
        """Applies trained model to given data X."""
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(
                X.values * 1, device=self.longitudinal_model.device, dtype=torch.float
            )
        else:
            X = torch.tensor(
                X, device=self.longitudinal_model.device, dtype=torch.float
            )
        outputs = self.longitudinal_model(X)  # .detach().cpu().numpy()

        return outputs

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs
    ) -> "AbstractCRRTPredictor":
        """
        Create an instance from CLI arguments.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
        # Ref: https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.8.3/pytorch_lightning/trainer/trainer.py#L750
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid args, the rest may be user specific
        # returns a immutable dict MappingProxyType, want to combine so copy
        valid_kwargs = inspect.signature(cls.__init__).parameters.copy()
        valid_kwargs.update(
            inspect.signature(StaticModel.__init__).parameters.copy()
        )
        data_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        data_kwargs.update(**kwargs)

        return cls(**data_kwargs)
