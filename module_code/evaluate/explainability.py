from typing import List, Optional
from numpy import ndarray
from pandas import Index
from os.path import join
from sklearn.base import ClassifierMixin
import matplotlib.pyplot as plt
import lime
import shap

from evaluate.utils import log_figure, filter_fns


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
    explainer = shap.Explainer(
        model.predict,
        data,
        feature_names=columns.to_list(),
        output_names=["Do not Recommend", "Recommend"],
        categorical_columns=[columns.get_loc(col) for col in categorical_columns],
        max_evals=2 * len(columns) + 1,  # Error from Permutation explainer
        seed=seed,
    )
    shap_values: shap.Explanation = explainer(data)
    for sample_type in ["tp", "tn", "fp", "fn"]:
        # explain 1 (just randomly pick first) sample from tp and tn
        idxs = filter_fns[sample_type](preds, labels).nonzero()[0]
        if len(idxs) > 0:
            i = idxs[0]
            plt.clf()
            figure = shap.plots.waterfall(shap_values[i], show=False)
            figure.suptitle(
                f"{prefix}{sample_type} Decision Explanation (SHAP, iloc: {i})", y=1
            )
            log_figure(
                figure,
                join(
                    "img_artifacts",
                    "explanation",
                    f"{prefix}_{sample_type}_explanation",
                ),
            )

    if top_k:
        plt.clf()
        shap.plots.beeswarm(shap_values, show=False)
        figure = plt.gcf()
        figure.suptitle(f"{prefix} SHAP Feature Impact")
        # includes impact direction
        log_figure(
            figure,
            join("img_artifacts", "feature_impact", f"{prefix}_feature_impact"),
        )

        plt.clf()
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
