from typing import List
from numpy import ndarray, argwhere, apply_along_axis
from os.path import join
from sklearn.base import ClassifierMixin
from mlflow import log_figure
import lime
import lime.lime_tabular

from evaluate.error_analysis import filter_fns, is_ctn


def is_cat(
    feature_values: ndarray, threshold: float = 0.05, long_tailed: bool = False
) -> bool:
    # TODO: this is too basic of a heuristic, it's capturing dx_CCS_CODEs
    return not is_ctn(feature_values, threshold, long_tailed)


def lime_explainability(
    data: ndarray,
    labels: ndarray,
    prefix: str,
    model: ClassifierMixin,
    columns: List[str],
    seed: int,
):
    # index corresponding to categorical columns.
    categorical_features = argwhere(apply_along_axis(is_cat, 0, data)).squeeze()
    explainer = lime.lime_tabular.LimeTabularExplainer(
        data,
        feature_names=columns,
        class_names=["Do not Recommend", "Recommend"],
        categorical_features=categorical_features,
        # categorical_names=categorical_names,
        random_state=seed,
    )
    preds = model.predict(data)
    for sample_type in ["tp", "tn"]:
        sample = data[filter_fns[sample_type](preds, labels)]
        if len(sample) > 0:
            # explain 1 (just randomly pick the first) sample from tp and tn
            exp = explainer.explain_instance(sample[0], model.predict_proba)
            log_figure(
                exp.as_pyplot_figure(),
                join("img_artifacts", f"{prefix}_{sample_type}_explanation.png"),
            )
