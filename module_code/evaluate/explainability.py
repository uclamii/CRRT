from typing import List, Optional
from numpy import ndarray
from pandas import Index
from os.path import join, dirname
from sklearn.base import ClassifierMixin
import matplotlib.pyplot as plt
import lime
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow
from os import makedirs
import pickle

from evaluate.utils import log_figure, filter_fns


def dump_shap_values(shap_values, preds, labels, prefix):
    to_store = {
        "values": shap_values.values,
        "data": shap_values.data,
        "base_values": shap_values.base_values,
        "feature_names": shap_values.feature_names,
    }

    for filt in ["tp", "tn", "fp", "fn"]:
        to_store[filt] = filter_fns[filt](preds, labels).nonzero()[0]

    array_file = join(
        "img_artifacts", "feature_importance", f"{prefix}_shap_values.pkl"
    )
    makedirs(
        dirname(join("local_data", array_file)), exist_ok=True
    )  # ensure dir exists

    with open(join("local_data", array_file), "wb") as f:
        pickle.dump(to_store, f)

    if mlflow.active_run():
        mlflow.log_artifact(join("local_data", array_file), dirname(array_file))


def shap_explainability(
    data: ndarray,
    labels: ndarray,
    preds: ndarray,
    prefix: str,
    model: ClassifierMixin,
    columns: Index,
    categorical_columns: List[int],
    seed: int,
    top_k: Optional[int] = None,
):
    if not isinstance(model, LogisticRegression):
        # algorithm = "auto"
        # if isinstance(model, RandomForestClassifier):
        #     algorithm = "tree"
        explainer = shap.Explainer(
            model,
            feature_names=columns.to_list(),
            output_names=["Do not Recommend", "Recommend"],
            categorical_columns=[columns.get_loc(col) for col in categorical_columns],
            seed=seed,
        )
    else:
        # Logistic regression has an error with not being callable. Use LinearExplainer with a masker
        # Also need to remove output_names column
        masker = shap.maskers.Independent(data=data)
        explainer = shap.LinearExplainer(
            model,
            masker=masker,
            feature_names=columns.to_list(),
            categorical_columns=[columns.get_loc(col) for col in categorical_columns],
            seed=seed,
        )

    shap_values: shap.Explanation = explainer(data)
    # label_dims = [1, 0, 1, 0]
    # for label_dim, sample_type in zip(label_dims, ["tp", "tn", "fp", "fn"]):
    #     # explain 1 (just randomly pick first) sample from tp and tn
    #     idxs = filter_fns[sample_type](preds, labels).nonzero()[0]
    #     if len(idxs) > 0:
    #         i = idxs[0]
    #         plt.clf()
    #         if len(shap_values.shape) == 3:  # samples x features x output
    #             figure = shap.plots.waterfall(shap_values[i][:, label_dim], show=False)
    #         else:  # samples x features
    #             figure = shap.plots.waterfall(shap_values[i], show=False)
    #         figure.suptitle(
    #             f"{prefix}{sample_type} Decision Explanation (SHAP, iloc: {i})", y=1
    #         )
    #         log_figure(
    #             figure,
    #             join(
    #                 "img_artifacts",
    #                 "explanation",
    #                 f"{prefix}_{sample_type}_explanation",
    #             ),
    #         )

    if top_k:
        dump_shap_values(shap_values, preds, labels, prefix)

        plt.clf()
        if len(shap_values.shape) == 3:
            shap.plots.beeswarm(shap_values[:, :, 1], show=False)
        else:
            shap.plots.beeswarm(shap_values, show=False)
        figure = plt.gcf()
        figure.suptitle(f"{prefix} SHAP Feature Impact")
        # includes impact direction
        log_figure(
            figure,
            join("img_artifacts", "feature_impact", f"{prefix}_feature_impact"),
        )

        plt.clf()
        if len(shap_values.shape) == 3:
            shap.plots.bar(shap_values[:, :, 1], show=False)
        else:
            shap.plots.bar(shap_values, show=False)
        figure = plt.gcf()
        figure.suptitle(f"{prefix} SHAP Absolute Feature Importance")
        # absolute value magnitude impact
        log_figure(
            figure,
            join(
                "img_artifacts",
                "feature_importance",
                f"{prefix}_feature_importance",
            ),
        )
        plt.close()


def lime_explainability(
    data: ndarray,
    labels: ndarray,
    preds: ndarray,
    prefix: str,
    model: ClassifierMixin,
    columns: List[str],
    categorical_columns: List[int],
    seed: int,
):
    # index corresponding to categorical columns.
    explainer = lime.lime_tabular.LimeTabularExplainer(
        data,
        feature_names=columns,
        class_names=["Do not Recommend", "Recommend"],
        # Requires indices
        categorical_features=[columns.get_loc(col) for col in categorical_columns],
        random_state=seed,
    )
    for sample_type in ["tp", "tn", "fp", "fn"]:
        sample = data[filter_fns[sample_type](preds, labels)]
        if len(sample) > 0:
            # explain 1 (just randomly pick the first) sample from tp and tn
            exp = explainer.explain_instance(sample[0], model.predict_proba)
            figure = exp.as_pyplot_figure()
            figure.suptitle(f"{prefix} {sample_type} Decision Explanation (LIME)")
            # figure.set_tight_layout(True)
            log_figure(
                figure,
                join(
                    "img_artifacts",
                    "explanation",
                    f"{prefix}_{sample_type}_explanation",
                ),
            )
