from typing import List
from numpy import ndarray
from os.path import join
from sklearn.base import ClassifierMixin
from mlflow import log_figure
import lime
import lime.lime_tabular


# TODO: WIP
def lime_explainability(
    train_data: ndarray,
    eval_data: ndarray,
    prefix: str,
    model: ClassifierMixin,
    columns: List[str],
    seed: int,
):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        train_data,
        feature_names=columns,
        class_names="",
        categorical_features="",
        categorical_names="",
        random_state=seed,
    )
    exp = explainer.explain_instance(eval_data[0], model.predict_proba)
    log_figure(
        exp.as_pyplot_figure(), join("img_artifacts", f"{prefix}_explanation.png")
    )
