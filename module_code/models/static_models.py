from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Dict, List, Union
from os.path import join, dirname
from os import makedirs
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
from pandas import Index

from pytorch_lightning.core.mixins import HyperparametersMixin
from pytorch_lightning.core.saving import save_hparams_to_yaml, load_hparams_from_yaml

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
from data.argparse_utils import YAMLStringDictToDict, YAMLStringListToList

from models.utils import has_gpu, seed_everything
from models.base_model import BaseSklearnPredictor, AbstractModel
from evaluate.error_viz import error_visualization
from evaluate.error_analysis import model_randomness
from evaluate.explanability import lime_explainability
from evaluate.feature_importance import log_feature_importances

STATIC_MODEL_FNAME = "static_model.pkl"
STATIC_HPARAM_FNAME = "static_hparams.yml"

ALG_MAP = {
    "lgr": LogisticRegression,
    # "knn": KNeighborsClassifier,
    # "dt": DecisionTreeClassifier,
    "rf": RandomForestClassifier,
    "lgb": LGBMClassifier,
    "xgb": XGBClassifier,
}

# gt = ground truth
METRIC_MAP = {
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

error_analysis_map = {
    "visualize": error_visualization,
    "randomness": model_randomness,
    "explain": lime_explainability,
}


class StaticModel(AbstractModel, HyperparametersMixin):
    def __init__(
        self,
        seed: int,
        modeln: str,
        metric_names: List[str],
        curve_names: List[str],
        error_analysis: List[str],
        top_k_feature_importance: int,
        model_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        if self.hparams["modeln"] not in {"knn", "nb"}:
            self.hparams["model_kwargs"]["random_state"] = seed
        if self.hparams["modeln"] == "xgb" and has_gpu():
            # Get rid of warnings
            self.hparams["model_kwargs"]["use_label_encoder"] = False
            self.hparams["model_kwargs"]["eval_metric"] = "logloss"
            # self.model_kwargs["tree_method"] = "gpu_hist"  # installing py-xgboost-gpu is not working
        self.setup()

    def setup(self):
        self.model = self.build_model()
        self.metrics = self.configure_metrics(self.hparams["metric_names"])
        self.curves = self.configure_curves(self.hparams["curve_names"])
        self.error_analysis = self.configure_error_analysis(
            self.hparams["error_analysis"]
        )

    def build_model(self):
        if self.hparams["modeln"] in ALG_MAP:
            model_cls = ALG_MAP[self.hparams["modeln"]]
        else:
            raise ValueError(
                "The {} is not a valid model type".format(self.hparams["modeln"])
            )
        return model_cls(**self.hparams["model_kwargs"])

    def configure_metrics(self, metric_names: List[str]) -> List[Callable]:
        """Pick metrics."""
        if metric_names is None:
            return None
        return [METRIC_MAP[metric] for metric in metric_names]

    def configure_curves(self, curve_names: List[str]) -> List[Callable]:
        """Pick plots."""
        if curve_names is None:
            return None
        for curve in curve_names:
            assert (
                curve in curve_map
            ), f"{curve} is not valid plot name. Must match a key in `plot_map`"
        return [curve_map[curve] for curve in curve_names]

    def configure_error_analysis(self, error_analysis: List[str]) -> List[Callable]:
        if error_analysis is None:
            return None
        return [error_analysis_map[analysis_name] for analysis_name in error_analysis]

    def log_model(self):
        """Log to MLFlow."""
        # Log hparams
        hparams_file = join("static_model", STATIC_HPARAM_FNAME)
        mlflow.log_dict(self.hparams, hparams_file)

        # Log model
        # Select which logging function from mlflow
        if self.hparams["modeln"] == "xgb":
            log_fn = mlflow.xgboost.log_model
        elif self.hparams["modeln"] == "lgb":
            log_fn = mlflow.lightgbm.log_model
        else:
            log_fn = mlflow.sklearn.log_model
        # Log to MlFlow
        log_fn(self.model, join("static_model", STATIC_MODEL_FNAME))

    def save_model(self, serialized_static_model_dir: str):
        """Vs logging with mlflow, saves to selected path"""
        # Save hparams
        hparams_file = join(serialized_static_model_dir, STATIC_HPARAM_FNAME)
        # make if does not exist, otherwise overwrite
        makedirs(dirname(hparams_file), exist_ok=True)
        save_hparams_to_yaml(hparams_file, self.hparams)

        # Save model
        if self.hparams["modeln"] == "xgb":
            save_fn = self.model.save_model
        else:
            save_fn = lambda path: mlflow.sklearn._save_model(
                self.model, path, "pickle"
            )
        save_fn(join(serialized_static_model_dir, STATIC_MODEL_FNAME))

    def load_model(self, serialized_static_model_dir: str) -> None:
        """Loads model regardless if saved locally or in mlflow."""
        # Load Hparams (hparams is a property not an attribute so I cannot assign it directly)
        self._set_hparams(
            load_hparams_from_yaml(
                join(serialized_static_model_dir, STATIC_HPARAM_FNAME)
            )
        )

        # Load Model
        model_path = join(serialized_static_model_dir, STATIC_MODEL_FNAME)
        try:
            if self.hparams["modeln"] == "xgb":
                load_fn = mlflow.xgboost.load_model
            elif self.hparams["modeln"] == "lgb":
                load_fn = mlflow.lightgbm.load_model
            else:
                load_fn = mlflow.sklearn.load_model
            self.model = load_fn(model_path)
        except mlflow.exceptions.MlflowException:  # not mlflow logged
            if self.hparams["modeln"] == "xgb":
                self.model = XGBClassifier()
                self.model.load_model(model_path)
            else:
                self.model = mlflow.sklearn._load_model_from_local_file(
                    model_path, "pickle"
                )

    @staticmethod
    def add_model_args(p: ArgumentParser) -> ArgumentParser:
        p.add_argument(
            "--static-modeln",
            dest="modeln",
            type=str,
            default="lgr",
            choices=list(ALG_MAP.keys()),
            help="Name of model to use for learning.",
        )
        p.add_argument(
            "--static-model-kwargs",
            dest="model_kwargs",
            action=YAMLStringDictToDict(),
            default={},
            help="Model kwargs corresponding to the model specified in modeln.",
        )
        p.add_argument(
            "--static-metrics",
            dest="metric_names",
            type=str,
            action=YAMLStringListToList(str, choices=list(METRIC_MAP.keys())),
            help="(List of comma-separated strings) Name of metrics from sklearn.",
        )
        p.add_argument(
            "--static-curves",
            dest="curve_names",
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
            default=None,
            help="Number of features to limit feature importances to.",
        )
        return p


class CRRTStaticPredictor(BaseSklearnPredictor):
    """
    Wrapper predictor class, compatible with sklearn.
    Uses longitudinal model to do time series classification on tabular data.
    Implements fit and transform.
    """

    def __init__(self, seed: int, **kwargs):
        self.seed = seed
        self.static_model = StaticModel(seed=seed, **kwargs)

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs
    ) -> "CRRTStaticPredictor":
        return super().from_argparse_args(args, [StaticModel], **kwargs)

    def log_model(self):
        self.static_model.log_model()

    def load_model(self, serialized_static_model_path: str) -> None:
        self.static_model.load_model(serialized_static_model_path)

    def save_model(self, serialized_static_model_path: str) -> None:
        self.static_model.save_model(serialized_static_model_path)

    def fit(self, data: SklearnCRRTDataModule):
        seed_everything(self.seed)
        self.static_model.model.fit(*data.train)
        self.log_model()
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Applies trained model to given data X."""
        outputs = self.static_model.model.transform(X)
        return outputs

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.static_model.model.predict_proba(X)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.static_model.model.predict(X)

    def evaluate(
        self,
        data: SklearnCRRTDataModule,
        stage: str,
    ) -> Dict[str, Any]:
        """
        Can additionally pass filters for subsets to evaluate performance.
        The filters can be either the bool series filter itself, or a function that produces one given the df.
        """
        X, y = getattr(data, stage)
        if hasattr(data, "data_transform"):
            selected_columns_mask = data.data_transform.__self__.named_steps[
                "feature-selection"
            ].get_support()
            columns = data.columns[selected_columns_mask]
        else:
            columns = data.columns
        categorical_columns = data.categorical_columns.intersection(columns)
        X = pd.DataFrame(X, columns=columns)
        ## Evaluate on split of dataset (train, val, test) ##
        metrics = self.eval_and_log(
            X,
            y,
            prefix=f"{self.static_model.hparams['modeln']}_{stage}_",
            categorical_columns=categorical_columns,
        )

        ## Evaluate for each filter ##
        filters = getattr(data, f"{stage}_filters")
        if filters is not None:
            for filter_n, filter in filters.items():
                # If we don't ask for values sklearn will complain it was fitted without feature names
                self.eval_and_log(
                    X[filter.values],
                    y[filter.values],
                    prefix=f"{self.static_model.hparams['modeln']}_{stage}_{filter_n}_",
                    categorical_columns=categorical_columns,
                )

        return metrics

    def eval_and_log(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        prefix: str,
        categorical_columns: Union[List[str], Index],
        decision_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Logs metrics and curves/plots."""
        metrics = None
        predict_proba = self.predict_proba(data.values)[:, 1]
        # np.save(predict_probas_file, predict_proba)

        # log predict probabilities
        predict_probas_file = join("predict_probas", f"{prefix}_predict_probas.pkl")
        makedirs(dirname(predict_probas_file), exist_ok=True)  # ensure dir exists
        predict_probas = pd.Series(predict_proba, index=labels.index)
        predict_probas.to_pickle(predict_probas_file)
        mlflow.log_artifact(predict_probas_file, dirname(predict_probas_file))

        # Metrics
        if self.static_model.hparams["metric_names"] is not None:
            metrics = {
                f"{prefix}_{metric_name}": metric_fn(
                    labels, predict_proba, decision_threshold
                )
                for metric_name, metric_fn in zip(
                    self.static_model.hparams["metric_names"],
                    self.static_model.metrics,
                )
            }
            mlflow.log_metrics(metrics)

        # Curves/Plots
        if self.static_model.hparams["curve_names"] is not None:
            for curve_name, curve in zip(
                self.static_model.hparams["curve_names"],
                self.static_model.curves,
            ):
                mlflow.log_figure(
                    curve.from_predictions(labels, predict_proba).plot().figure_,
                    join("img_artifacts", "curves", f"{prefix}_{curve_name}.png"),
                )

        # Error analysis
        if self.static_model.hparams["error_analysis"] is not None:
            for analysis_fn in self.static_model.error_analysis:
                analysis_fn(
                    data.values,
                    labels,
                    prefix,
                    self.static_model.model,
                    data.columns,
                    categorical_columns,
                    self.seed,
                )

        # Feature importance
        # Ref: https://machinelearningmastery.com/calculate-feature-importance-with-python/
        if self.static_model.hparams["top_k_feature_importance"] is not None:
            log_feature_importances(
                self.static_model.hparams["top_k_feature_importance"],
                data.values,
                labels,
                prefix,
                self.static_model.model,
                data.columns,
                self.seed,
            )
        return metrics
