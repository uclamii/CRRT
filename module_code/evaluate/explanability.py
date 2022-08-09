from typing import List
from numpy import ndarray
from os.path import join
from sklearn.base import ClassifierMixin
from mlflow import log_figure
import lime
import lime.lime_tabular

from evaluate.error_analysis import filter_fns


def lime_explainability(
    data: ndarray,
    labels: ndarray,
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
    preds = model.predict(data)
    for sample_type in ["tp", "tn", "fp", "fn"]:
        sample = data[filter_fns[sample_type](preds, labels)]
        if len(sample) > 0:
            # explain 1 (just randomly pick the first) sample from tp and tn
            exp = explainer.explain_instance(sample[0], model.predict_proba)
            figure = exp.as_pyplot_figure()
            figure.set_tight_layout(True)
            log_figure(
                figure,
                join(
                    "img_artifacts",
                    "explanation",
                    f"{prefix}_{sample_type}_explanation.png",
                ),
            )
