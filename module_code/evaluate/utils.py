from typing import TYPE_CHECKING, Callable, List, Tuple, Dict, Any
from os.path import dirname
from os import makedirs
import mlflow
from scipy.stats import bootstrap
from sklearn.utils import resample
import numpy as np
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


def log_figure(figure: "matplotlib.figure.Figure", path: str):
    if mlflow.active_run() is None:  # log locally if mlflow not running
        # make if does not exist, otherwise overwrite
        makedirs(dirname(path), exist_ok=True)
        figure.savefig(path)
    else:
        mlflow.log_figure(figure, path)


def log_text(text: str, path: str):
    if mlflow.active_run() is None:  # log locally if mlflow not running
        makedirs(dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        mlflow.log_text(text, path)


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    metric_fn: Callable,
    n_resamples: int = 1000,
    random_state: int = 42,
    decision_threshold: float = 0.5,
    mode: str = "resample",
) -> np.ndarray:

    # use default scipy bootstraping. outputs distribution of the statistic
    if mode == "scipy":
        bootstrapped_scores = bootstrap(
            (y_true, y_pred, [decision_threshold] * len(y_true)),
            metric_fn,
            vectorized=False,
            paired=True,
            random_state=random_state,
            n_resamples=1000,
            batch=len(y_true),
        ).bootstrap_distribution

    # manually resample the dataset
    elif mode == "resample":
        bootstrapped_scores = []

        random_instance = np.random.RandomState(seed=random_state)
        for i in range(n_resamples):

            # bootstrap by sampling with replacement
            X, y = resample(y_pred, y_true, random_state=random_instance, replace=True)

            labels_are_homog = len(y.value_counts()) == 1
            metrics_ok_homog = {"accuracy"}
            if labels_are_homog and metric_name not in metrics_ok_homog:
                score = np.nan
            else:
                score = metric_fn(y, X, decision_threshold)

            bootstrapped_scores.append(score)

    bootstrapped_scores = np.array(bootstrapped_scores)

    return bootstrapped_scores


def confidence_interval(sorted_scores: np.ndarray) -> Tuple[float]:

    # first sort
    sorted_scores.sort()

    # get confidence intervals
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper


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
