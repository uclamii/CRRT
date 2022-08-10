from logging import warn
from typing import List
from mlflow import log_artifact, log_figure
from matplotlib import pyplot as plt
from mealy import ErrorAnalyzer, ErrorVisualizer
from os.path import join
from numpy import ndarray
from sklearn.base import ClassifierMixin


def error_visualization(
    data: ndarray,
    labels: ndarray,
    prefix: str,
    model: ClassifierMixin,
    columns: List[str],
    categorical_columns: List[str],
    seed: int,
):
    if len(data) < 100:
        warn(f"Dataset {prefix} has less than 100 rows, cannot do error visualization.")
        return
    try:
        error_analyzer = ErrorAnalyzer(
            model,
            feature_names=columns,
            random_state=seed,
        )
        error_analyzer.fit(data, labels)
        error_analyzer.evaluate(data, labels, output_format="dict")
        error_analyzer.get_error_leaf_summary(
            leaf_selector=None, add_path_to_leaves=True
        )

        error_viz = ErrorVisualizer(error_analyzer)
        # graphviz source to png so it can be logged
        tree_src = error_viz.plot_error_tree()
        tree_src.format = "png"
        tree_src.render(join("img_artifacts", f"{prefix}_tree"))
        path = join("img_artifacts", f"{prefix}_tree.png")
        log_artifact(path, path)

        leaf_id = error_analyzer._get_ranked_leaf_ids()[0]
        # TODO: tie this to feature importance top-k-features? (use same k)
        error_viz.plot_feature_distributions_on_leaves(
            leaf_selector=leaf_id, top_k_features=5
        )
        log_figure(
            plt.gcf(), join("img_artifacts", "error_viz", f"{prefix}_leave_dists.png")
        )
    except RuntimeError:
        # all predictions are correct no error analysis, skip
        pass
