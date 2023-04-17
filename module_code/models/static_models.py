from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Dict, List, Optional, Union
from posixpath import join, dirname
from os import makedirs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from evaluate.explainability import shap_explainability, lime_explainability
from evaluate.feature_importance import feature_importance
from evaluate.utils import log_figure, bootstrap_metric, confidence_interval

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
CURVE_MAP = {
    "calibration_curve": CalibrationDisplay,
    "roc_curve": RocCurveDisplay,
    "pr_curve": PrecisionRecallDisplay,
    "det_curve": DetCurveDisplay,
}

PLOT_MAP = {
    "confusion_matrix": ConfusionMatrixDisplay,
    "error_viz": error_visualization,
    "randomness": model_randomness,
    "shap_explain": shap_explainability,
    "lime_explain": lime_explainability,
    "feature_importance": feature_importance,
}

MODEL_LEVEL_METRICS = [
    "error_viz",
    "randomness",
    "feature_importance",
] + list(CURVE_MAP.keys())

# Dont' want to do extensive evaluation on the validation or training set
TEST_ONLY_METRICS = list(CURVE_MAP.keys()) + list(PLOT_MAP.keys())


class StaticModel(AbstractModel, HyperparametersMixin):
    def __init__(
        self,
        seed: int,
        modeln: str,
        metric_names: List[str],
        curve_names: List[str],
        plot_names: List[str],
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
        self.plots = self.configure_plots(self.hparams["plot_names"])

    def build_model(self):
        if self.hparams["modeln"] in ALG_MAP:
            model_cls = ALG_MAP[self.hparams["modeln"]]
        else:
            raise ValueError(
                "The {} is not a valid model type".format(self.hparams["modeln"])
            )

        # hparams that can't be passed to rf are saved, causing issues in post-hoc testing
        # this doesn't solve the root problem, but a temporary fix
        if self.hparams["modeln"] in ["rf"]:
            import inspect

            valid_kwargs = inspect.signature(model_cls.__init__).parameters.copy()
            data_kwargs = dict(
                (name, self.hparams["model_kwargs"][name])
                for name in valid_kwargs
                if name in self.hparams["model_kwargs"]
            )
            return model_cls(**data_kwargs)

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
        return [CURVE_MAP[curve] for curve in curve_names]

    def configure_plots(self, plot_names: List[str]) -> List[Callable]:
        if plot_names is None:
            return None
        self.use_shap_for_feature_importance = True
        plots = []
        for plot in plot_names:
            if plot == "feature_importance" and self.use_shap_for_feature_importance:
                pass
            else:
                plots.append(PLOT_MAP[plot])
        return plots

    def log_model(self):
        """Log to MLFlow."""

        # Log hparams
        # manually get path from uri and save using pytorch lightning
        # windows has issues when using mlflow to save and load hyperparameters from yaml
        hparam_path = mlflow.get_artifact_uri().split("file://")[1]
        hparams_file = join(hparam_path, "static_model", STATIC_HPARAM_FNAME)
        makedirs(dirname(hparams_file), exist_ok=True)
        save_hparams_to_yaml(hparams_file, self.hparams)
        # mlflow.log_dict(self.hparams, hparams_file)

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

    @classmethod
    def load(cls, serialized_static_model_dir: str) -> "StaticModel":
        """Different from load model, this creates an instance from the pickle instead of just loading in"""
        hparams = load_hparams_from_yaml(
            join(serialized_static_model_dir, STATIC_HPARAM_FNAME)
        )
        static_model = cls(**hparams)
        static_model.load_model(serialized_static_model_dir)
        return static_model

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
            action=YAMLStringListToList(str, choices=list(CURVE_MAP.keys())),
            help="(List of comma-separated strings) Name of curves/plots from sklearn.",
        )
        p.add_argument(
            "--static-plots",
            dest="plot_names",
            action=YAMLStringListToList(str, choices=list(PLOT_MAP.keys())),
            help="(List of comma-separated strings) Name of which plots aside from curves are desired for static prediction.",
        )
        p.add_argument(
            "--static-top-k-feature-importance",
            dest="top_k_feature_importance",
            type=int,
            default=10,
            help="Number of features to limit feature importances to. Does NOT turn on/off the plot.",
        )
        return p


class CRRTStaticPredictor(BaseSklearnPredictor):
    """
    Wrapper predictor class, compatible with sklearn.
    Uses longitudinal model to do time series classification on tabular data.
    Implements fit and transform.
    """

    def __init__(self, seed: int, static_model: StaticModel = None, **kwargs):
        self.seed = seed
        if static_model is not None:
            self.static_model = static_model
        else:
            self.static_model = StaticModel(seed=seed, **kwargs)

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs
    ) -> "CRRTStaticPredictor":
        return super().from_argparse_args(args, [StaticModel], **kwargs)

    def log_model(self):
        self.static_model.log_model()

    @classmethod
    def load(cls, args: Namespace, serialized_model_path: str) -> "CRRTStaticPredictor":
        """Different from load model, this creates an instance from the pickle instead of just loading in"""

        """
        Somethign is broken with pyyaml: https://github.com/yaml/pyyaml/issues/266
        The fix is to change `load_hparams_from_yaml` utility itself.
        ```
        with fs.open(config_yaml, "r") as fp:
            # hparams = yaml.full_load(fp)
            hparams = yaml.load(fp, Loader=yaml.Loader)
        ```
        """
        # hparams = load_hparams_from_yaml(
        # join(serialized_model_path, STATIC_HPARAM_FNAME)
        # )
        # return cls(hparams["seed"], static_model=static_model)
        static_model = StaticModel.load(serialized_model_path)
        return cls.from_argparse_args(args, static_model=static_model)

    def load_model(self, serialized_static_model_path: str) -> None:
        self.static_model.load_model(serialized_static_model_path)

    def save_model(self, serialized_static_model_path: str) -> None:
        self.static_model.save_model(serialized_static_model_path)

    def fit(self, data: SklearnCRRTDataModule):
        seed_everything(self.seed)
        self.static_model.model.fit(*data.train)
        if mlflow.active_run() is None:
            self.save_model(join("local_data", "static_model"))
        else:
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
        pred_probas = pd.Series(self.predict_proba(X.values)[:, 1], index=y.index)
        preds = pd.Series(self.predict(X.values), index=y.index)
        # np.save(predict_probas_file, predict_proba)

        ## Evaluate on split of dataset (train, val, test) ##
        metrics = self.eval_and_log(
            stage,
            X,
            y,
            pred_probas,
            preds,
            prefix=f"{self.static_model.hparams['modeln']}_{stage}_",
            categorical_columns=categorical_columns,
        )

        ## Evaluate for each filter ##
        filters = getattr(data, f"{stage}_filters")
        if filters is not None:
            if mlflow.active_run():
                subgroup_sizes = {f"{k}_N": v.sum() for k, v in filters.items()}
                mlflow.log_metrics(subgroup_sizes)

            for filter_n, filter in filters.items():
                if filter.sum():  # don't do anything if there's no one represented
                    # If we don't ask for values sklearn will complain it was fitted without feature names
                    self.eval_and_log(
                        stage,
                        X,
                        y,
                        pred_probas,
                        preds,
                        prefix=f"{self.static_model.hparams['modeln']}_{stage}_{filter_n}_",
                        categorical_columns=categorical_columns,
                        subgroup_filter=filter.values,
                    )

        return metrics

    def eval_and_log(
        self,
        stage: str,
        data: pd.DataFrame,
        labels: pd.Series,
        pred_probas: pd.Series,
        preds: pd.Series,
        prefix: str,
        categorical_columns: Union[List[str], Index],
        subgroup_filter: Optional[pd.Series] = None,
        decision_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Logs metrics and curves/plots."""
        if subgroup_filter is not None:  # apply filter here
            data = data[subgroup_filter]
            labels = labels[subgroup_filter]
            preds = preds[subgroup_filter]
            pred_probas = pred_probas[subgroup_filter]

        def eval_conditions(name: str) -> bool:
            # if it's not a subgroup we don't care, evaluate
            # if it's a subgroup then we skip model-level metrics (like feature importance)
            subgroup_eval_conditions = (subgroup_filter is None) or (
                name not in MODEL_LEVEL_METRICS
            )

            # if the metric is test only then the stage should be test
            stage_eval_conditions = (name not in TEST_ONLY_METRICS) or (stage == "test")
            return subgroup_eval_conditions and stage_eval_conditions

        metrics = None

        # log predict probabilities
        # predict_probas_file = join("predict_probas", f"{prefix}_predict_probas.pkl")
        # makedirs(dirname(predict_probas_file), exist_ok=True)  # ensure dir exists
        # pred_probas.to_pickle(predict_probas_file)
        # if mlflow.active_run():
        #     mlflow.log_artifact(predict_probas_file, dirname(predict_probas_file))

        # log labels for convenient post-hoc analysis if required
        # if "test" in prefix:
        #     labels_file = join("predict_probas", f"{prefix}_labels.pkl")
        #     labels.to_pickle(labels_file)
        #     if mlflow.active_run():
        #         mlflow.log_artifact(labels_file, dirname(predict_probas_file))

        pred_probas = pred_probas.values

        labels_are_homog = len(labels.value_counts()) == 1
        metrics_ok_homog = {"accuracy"}

        # Metrics
        if self.static_model.hparams["metric_names"] is not None:
            metrics = {}
            for metric_name, metric_fn in zip(
                self.static_model.hparams["metric_names"],
                self.static_model.metrics,
            ):
                if eval_conditions(metric_name):
                    name = f"{prefix}_{metric_name}"
                    if labels_are_homog and metric_name not in metrics_ok_homog:
                        metrics[name] = np.nan
                    else:
                        metrics[name] = metric_fn(
                            labels, pred_probas, decision_threshold
                        )

                    if "test" in prefix:
                        bootstrapped = bootstrap_metric(
                            labels,
                            pred_probas,
                            metric_name,
                            metric_fn,
                            decision_threshold=decision_threshold,
                            random_state=self.static_model.hparams["model_kwargs"][
                                "random_state"
                            ],
                        )

                        CI = confidence_interval(bootstrapped)
                        metrics[name + "_CI_low"] = CI[0]
                        metrics[name + "_CI_high"] = CI[1]

                        bootstrapped = pd.Series(bootstrapped)
                        bootstrapped_file = join(
                            "bootstrapped", f"{name}_bootstrapped.pkl"
                        )
                        makedirs(
                            dirname(bootstrapped_file), exist_ok=True
                        )  # ensure dir exists
                        bootstrapped.to_pickle(bootstrapped_file)
                        if mlflow.active_run():
                            mlflow.log_artifact(
                                bootstrapped_file, dirname(bootstrapped_file)
                            )

            if mlflow.active_run():
                mlflow.log_metrics(metrics)
            else:
                print(metrics)

        # Curves/Plots
        if self.static_model.hparams["curve_names"] is not None:
            for curve_name, curve in zip(
                self.static_model.hparams["curve_names"],
                self.static_model.curves,
            ):
                if eval_conditions(curve_name):
                    # prevent a lot of figures from popping up on screen
                    figure = curve.from_predictions(labels, pred_probas)
                    name = f"{prefix}_{curve_name}"
                    figure.figure_.suptitle(name)
                    # log_figure(figure, join("img_artifacts", "curves", name))
                    log_figure(figure.figure_, join("img_artifacts", "curves", name))
                    plt.close()

        # Other plots
        if self.static_model.plots is not None:
            args = {
                "data": data.values,
                "labels": labels.values,
                "preds": preds.values,
                "prefix": prefix,
                "model": self.static_model.model,
                "columns": data.columns,
                "categorical_columns": categorical_columns,
                "seed": self.seed,
                "top_k": self.static_model.hparams["top_k_feature_importance"],
            }

            for analysis_fn in self.static_model.plots:
                # reverse lookup the name from the function
                name = next(key for key, fn in PLOT_MAP.items() if fn == analysis_fn)
                if eval_conditions(name):
                    # don't use shap for feature importance if:
                    # we say not to or if we're in a subgroup
                    if (
                        not self.static_model.use_shap_for_feature_importance
                        or subgroup_filter is not None
                    ) and analysis_fn == shap_explainability:
                        new_args = args.copy()
                        new_args["top_k"] = None
                        analysis_fn(**new_args)
                    else:
                        analysis_fn(**args)

        return metrics
