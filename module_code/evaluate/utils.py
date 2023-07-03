from typing import TYPE_CHECKING, Callable, List, Tuple, Dict, Any, Union
from posixpath import dirname, join
from os import makedirs
import mlflow
from scipy.stats import bootstrap
from sklearn.utils import resample
import pandas as pd
import numpy as np
from numpy import percentile
from numpy.random import default_rng
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import matplotlib  # pylint: disable=unused-import
    import matplotlib.figure


filter_fns = {
    "tp": lambda data, labels: (data == labels) & (labels == 1),
    "tn": lambda data, labels: (data == labels) & (labels == 0),
    "fp": lambda data, labels: (data != labels) & (data == 1),
    "fn": lambda data, labels: (data != labels) & (data == 0),
}


def log_figure(figure: "matplotlib.figure.Figure", path: str, ext: str = "svg"):
    path = f"{path}.{ext}"
    if mlflow.active_run() is None:  # log locally if mlflow not running
        # make if does not exist, otherwise overwrite
        path = join("local_data", path)
        makedirs(dirname(path), exist_ok=True)
        figure.savefig(path, format=ext, bbox_inches="tight")
    else:
        mlflow.log_figure(figure, path)


def log_text(text: str, path: str):
    if mlflow.active_run() is None:  # log locally if mlflow not running
        path = join("local_data", path)
        makedirs(dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        mlflow.log_text(text, path)


def dump_array(prefix: str, name: str, array: Union[pd.Series, np.ndarray]):
    if isinstance(array, pd.Series):
        array_file = join("local_data", name, f"{prefix}_{name}.pkl")
        makedirs(dirname(array_file), exist_ok=True)  # ensure dir exists
        array.to_pickle(array_file)
    else:  # np array
        array_file = join("local_data", name, f"{prefix}_{name}.npy")
        makedirs(dirname(array_file), exist_ok=True)  # ensure dir exists
        np.save(array_file, array)
    if mlflow.active_run():
        mlflow.log_artifact(array_file, dirname(array_file))


def eval_metric(
    labels: pd.Series,
    pred_probas: pd.Series,
    metric_name: str,
    metric_fn: Callable,
    decision_threshold: float = 0.5,
) -> float:
    labels_are_homog = len(labels.value_counts()) == 1

    if labels[0] == 1:
        metrics_ok_homog = {"accuracy", "precision", "recall", "TP", "FN"}
    else:
        metrics_ok_homog = {"accuracy", "specificity", "TN", "FP"}

    if labels_are_homog and metric_name not in metrics_ok_homog:
        return np.nan
    return metric_fn(labels, pred_probas, decision_threshold)


def bootstrap_metric(
    labels: np.ndarray,
    pred_probas: np.ndarray,
    metric_name: str,
    metric_fn: Callable,
    n_bootstrap_samples: int = 1000,
    seed: int = 42,
    decision_threshold: float = 0.5,
    mode: str = "resample",
) -> np.ndarray:
    # use default scipy bootstraping. outputs distribution of the statistic
    if mode == "scipy":
        bootstrapped_metrics = bootstrap(
            (labels, pred_probas, [decision_threshold] * len(labels)),
            metric_fn,
            vectorized=False,
            paired=True,
            random_state=seed,
            n_resamples=n_bootstrap_samples,
            batch=len(labels),
        ).bootstrap_distribution
    elif mode == "resample":  # manually resample the dataset
        bootstrapped_metrics = []

        gen = default_rng(seed)
        bootstrap_seeds = gen.integers(0, 10000, n_bootstrap_samples)
        for i in range(n_bootstrap_samples):
            # bootstrap by sampling with replacement
            pred_probas_boot, labels_boot = resample(
                pred_probas, labels, random_state=bootstrap_seeds[i], replace=True
            )

            metric = eval_metric(
                labels_boot,
                pred_probas_boot,
                metric_name,
                metric_fn,
                decision_threshold,
            )
            bootstrapped_metrics.append(metric)

    bootstrapped_metrics = np.array(bootstrapped_metrics)

    return bootstrapped_metrics


def confidence_interval(
    metrics: np.ndarray, confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Returns confidence interval at given confidence level for statistical
    distributions established with bootstrap sampling.
    """
    alpha = 1 - confidence_level
    lower = alpha / 2 * 100
    upper = (alpha / 2 + confidence_level) * 100
    return (percentile(metrics, lower), percentile(metrics, upper))


# The below is not currently used because it significantly slows down the runtime
def bootstrap_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    curve_fn: Callable,
    axes: Dict[str, Any],
    n_resamples: int = 1000,
    random_state: int = 42,
) -> Tuple[List[float], List[float]]:
    bootstrapped_y_values = []

    # get x coordinates from the original curve
    original_curve = curve_fn.from_predictions(y_true, y_pred)
    original_x = getattr(original_curve, axes["labels"][0])
    plt.close()

    random_instance = np.random.RandomState(seed=random_state)
    for i in range(n_resamples):
        # bootstrap by sampling with replacement
        X, y = resample(y_pred, y_true, random_state=random_instance, replace=True)

        # get new curve with bootstrapped samples
        curve = curve_fn.from_predictions(y, X)

        # get interpolated values using original x coordinates
        if "recall" in axes["labels"]:
            interpolated_y = np.interp(
                original_x,
                getattr(curve, axes["labels"][0])[::-1],
                getattr(curve, axes["labels"][1])[::-1],
            )
        else:
            interpolated_y = np.interp(
                original_x,
                getattr(curve, axes["labels"][0]),
                getattr(curve, axes["labels"][1]),
            )

        bootstrapped_y_values.append(interpolated_y)
        plt.close()

    return original_x, bootstrapped_y_values


def display_boostrap(
    x: List[float],
    y: List[float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    curve_fn: Callable,
    axes: Dict[str, Any],
    metrics: Dict[str, Any],
    prefix: str,
):
    # plot original figure and remove original data to keep consistent figure formatting
    figure = curve_fn.from_predictions(y_true, y_pred)
    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()

    # calculate the mean of the bootstrapped y values
    std_y = np.std(y, axis=0)
    mean_y = np.mean(y, axis=0)

    # set limits if necessary (only for roc)
    if axes["limits"] is not None:
        mean_y[-1] = axes["limits"][1]
        mean_y[0] = axes["limits"][0]

    # plot perfect calibration
    if "prob" in axes["labels"][0]:
        figure.ax_.plot([0, 1], [0, 1], ":k")

    # plot the mean
    metric_name = axes["metric"]
    figure.ax_.plot(
        x,
        mean_y,
        color="tab:blue",
        label=f"Classifier ({metric_name.upper()}={metrics[prefix+'__'+metric_name]:.3f})\n95% CI = {metrics[prefix+'__'+metric_name+'_CI_low']:.3f}-{metrics[prefix+'__'+metric_name+'_CI_high']:.3f}",
        lw=2,
    )

    # plot the std
    y_upper = np.minimum(mean_y + std_y, 1)
    y_lower = np.maximum(mean_y - std_y, 0)
    figure.ax_.fill_between(
        x,
        y_lower,
        y_upper,
        color="tab:blue",
        alpha=0.2,
    )

    figure.ax_.legend(loc="lower right")
    return figure
