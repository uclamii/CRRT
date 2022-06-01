from argparse import ArgumentParser, Namespace
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
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    average_precision_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from sklearn.calibration import CalibrationDisplay
import mlflow

## Local
from data.sklearn_loaders import SklearnCRRTDataModule
from data.argparse_utils import YAMLStringListToList

from exp.utils import seed_everything
from models.base_model import BaseSklearnPredictor, AbstractModel
from evaluate.error_viz import error_visualization
from evaluate.error_analysis import model_randomness
from evaluate.feature_importance import log_feature_importances

alg_map = {
    "lgr": LogisticRegression,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "nb": MultinomialNB,
    "dt": DecisionTreeClassifier,
    "rf": RandomForestClassifier,
    "lgb": LGBMClassifier,
    "xgb": XGBClassifier,
}

# gt = ground truth
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

# https://scikit-learn.org/stable/modules/classes.html#id3
curve_map = {
    "calibration_curve": CalibrationDisplay,
    "roc_curve": RocCurveDisplay,
    "pr_curve": PrecisionRecallDisplay,
    "det_curve": DetCurveDisplay,
    "confusion_matrix": ConfusionMatrixDisplay,
}

error_analysis_map = {"visualize": error_visualization, "randomness": model_randomness}


class StaticModel(AbstractModel):
    def __init__(
        self,
        seed: int,
        modeln: str,
        metrics: List[str],
        curves: List[str],
        error_analysis: List[str],
        top_k_feature_importance: int,
        **model_kwargs,
    ):
        super().__init__()
        self.seed = seed
        self.modeln = modeln
        self.model_kwargs = model_kwargs
        self.model_kwargs["random_state"] = seed
        self.model = self.build_model()
        self.metrics = self.configure_metrics(metrics)
        self.metric_names = metrics
        self.curves = self.configure_curves(curves)
        self.curve_names = curves
        self.error_analysis = error_analysis
        self.top_k_feature_importance = top_k_feature_importance

    def build_model(self):
        if self.modeln in alg_map:
            model_cls = alg_map[self.modeln]
        else:
            raise ValueError("The {} is not a valid model type".format(self.modeln))
        return model_cls(**self.model_kwargs)

    def configure_metrics(self, metric_names: List[str]) -> List[Callable]:
        """Pick metrics."""
        if metric_names is None:
            return None
        return [metric_map[metric] for metric in metric_names]

    def configure_curves(self, curve_names: List[str]) -> List[Callable]:
        """Pick plots."""
        if curve_names is None:
            return None
        for curve in curve_names:
            assert (
                curve in curve_map
            ), f"{curve} is not valid plot name. Must match a key in `plot_map`"
        return [curve_map[curve] for curve in curve_names]

    @staticmethod
    def add_model_args(p: ArgumentParser) -> ArgumentParser:
        p.add_argument(
            "--static-modeln",
            dest="modeln",
            type=str,
            default="lgr",
            choices=["lgr", "svm", "knn", "nb", "dt", "rf", "lgb", "xgb"],
            help="Name of model to use for learning.",
        )
        p.add_argument(
            "--static-metrics",
            dest="metrics",
            type=str,
            action=YAMLStringListToList(str, choices=list(metric_map.keys())),
            help="(List of comma-separated strings) Name of metrics from sklearn.",
        )
        p.add_argument(
            "--static-curves",
            dest="curves",
            type=str,
            action=YAMLStringListToList(str, choices=list(curve_map.keys())),
            help="(List of comma-separated strings) Name of curves/plots from sklearn.",
        )
        p.add_argument(
            "--static-error-analysis",
            dest="error_analysis",
            action=YAMLStringListToList(str, choices=list(error_analysis_map.keys())),
            help="(List of comma-separated strings) Name of which error analyses desired for static prediction.",
        )
        p.add_argument(
            "--static-top-k-feature-importance",
            dest="top_k_feature_importance",
            type=int,
            default=0,
            help="Number of features to limit feature importances to.",
        )
        return p


class CRRTStaticPredictor(BaseSklearnPredictor):
    """
    Wrapper predictor class, compatible with sklearn.
    Uses longitudinal model to do time series classification on tabular data.
    Implements fit and transform.
    """

    def __init__(self, seed: int, runtest: bool, **kwargs):
        self.seed = seed
        self.runtest = runtest
        self.static_model = StaticModel(seed=seed, **kwargs)

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs
    ) -> "CRRTStaticPredictor":
        return super().from_argparse_args(StaticModel, args, **kwargs)

    # TODO: This needs to be changed for the serialization of the static models
    def load_model(self, serialized_model_path: str) -> None:
        pass

    def fit(self, data: SklearnCRRTDataModule):
        seed_everything(self.seed)
        self.data = data
        # self.data.setup()

        self.static_model.model.fit(*self.data.train)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Applies trained model to given data X."""
        outputs = self.static_model.model.transform(X)
        return outputs

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.static_model.model.predict_proba(X)

    def evaluate(
        self,
        stage: str,
        filters: Optional[
            Dict[str, Union[pd.Series, Callable[[pd.DataFrame], pd.Series]]]
        ] = None,
    ):
        """
        Can additionally pass filters for subsets to evaluate performance.
        The filters can be either the bool series filter itself, or a function that produces one given the df.
        """
        X, y = getattr(self.data, stage)
        ## Evaluate on split of dataset (train, val, test) ##
        self.eval_and_log(X, y, prefix=f"{self.static_model.modeln}_{stage}_")

        X = pd.DataFrame(X, columns=self.data.columns)
        ## Evaluate for each filter ##
        if filters is not None:
            for filter_n, filter in filters.items():
                if isinstance(filter, Callable):
                    filter = filter(X)

                # If we don't ask for values sklearn will complain it was fitted without feature names
                self.eval_and_log(
                    X.values[filter],
                    y.values[filter],
                    prefix=f"{self.static_model.modeln}_{stage}_{filter_n}_",
                )

    def eval_and_log(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        prefix: str,
        decision_threshold: float = 0.5,
    ):
        """Logs metrics and curves/plots."""
        # Metrics
        if self.static_model.metric_names is not None:
            mlflow.log_metrics(
                {
                    f"{prefix}_{metric_name}": metric_fn(
                        labels, self.predict_proba(data)[:, 1], decision_threshold
                    )
                    for metric_name, metric_fn in zip(
                        self.static_model.metric_names, self.static_model.metrics
                    )
                }
            )

        # Curves/Plots
        if self.static_model.curve_names is not None:
            for curve_name, curve in zip(
                self.static_model.curve_names, self.static_model.curves
            ):
                mlflow.log_figure(
                    curve.from_predictions(labels, self.predict_proba(data)[:, 1])
                    .plot()
                    .figure_,
                    f"{prefix}_{curve_name}.png",
                )

        # Error analysis
        if self.static_model.error_analysis is not None:
            for analysis_name in self.static_model.error_analysis:
                error_analysis_map[analysis_name](
                    data,
                    labels,
                    prefix,
                    self.static_model.model,
                    self.data.columns,
                    self.seed,
                )

        # Feature importance
        # Ref: https://machinelearningmastery.com/calculate-feature-importance-with-python/
        if self.static_model.top_k_feature_importance:
            log_feature_importances(
                self.static_model.top_k_feature_importance,
                data,
                labels,
                prefix,
                self.static_model.model,
                self.data.columns,
                self.seed,
            )
