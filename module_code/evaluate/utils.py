from typing import TYPE_CHECKING
from os.path import dirname
from os import makedirs
import mlflow

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
        makedirs(dirname(path), exist_ok=True)
        figure.savefig(path, format=ext, bbox_inches="tight")
    else:
        mlflow.log_figure(figure, path)


def log_text(text: str, path: str):
    if mlflow.active_run() is None:  # log locally if mlflow not running
        makedirs(dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        mlflow.log_text(text, path)
