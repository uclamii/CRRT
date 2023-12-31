from typing import List, Callable
from numpy import ndarray
from pandas import Series
from os.path import join

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from matplotlib import pyplot as plt

from sklearn.base import ClassifierMixin
from mlflow import log_figure


def feature_importance(
    top_k: int,
    data: ndarray,
    labels: ndarray,
    prefix: str,
    model: ClassifierMixin,
    columns: List[str],
    seed: int,
    **kwargs,  # ignore args other fns use in plot_names but not this one
):
    # Ref: https://machinelearningmastery.com/calculate-feature-importance-with-python/
    # TODO: inject feature names
    if isinstance(model, LogisticRegression) or isinstance(model, SVC):
        importance = model.coef_[0]
    elif (
        isinstance(model, DecisionTreeClassifier)
        or isinstance(model, RandomForestClassifier)
        or isinstance(model, XGBClassifier)
        or isinstance(model, LGBMClassifier)
    ):
        importance = model.feature_importances_
    elif isinstance(model, KNeighborsClassifier) or isinstance(model, MultinomialNB):
        importance = permutation_importance(
            model, data, labels, random_state=seed
        ).importances_mean
    plt.figure()
    Series(importance, index=columns).nlargest(top_k).plot(kind="barh")
    plt.tight_layout()
    log_figure(
        plt.gcf(),
        join("img_artifacts", "feature_importance", f"{prefix}_feature_importance.png"),
    )
    plt.close()
