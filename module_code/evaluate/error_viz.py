from logging import warn
from typing import List
from mlflow import log_artifact, active_run
from matplotlib import pyplot as plt
from mealy import ErrorAnalyzer, ErrorVisualizer
from posixpath import join, dirname
from numpy import ndarray
from sklearn.base import ClassifierMixin
import collections
import numpy as np

from evaluate.utils import log_figure


def error_visualization(
    data: ndarray,
    labels: ndarray,
    prefix: str,
    model: ClassifierMixin,
    columns: List[str],
    seed: int,
    **kwargs,
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
        if active_run():
            log_artifact(path, dirname(path))

        leaf_id = error_analyzer._get_ranked_leaf_ids()[0]

        # Return summary of leaf, including ids of path
        leaves_summary = get_leaf_summary(error_analyzer, leaf_id)
        path = leaves_summary[0]["path_to_leaf"]
        nodes_to_leaf = []
        incorrect = []
        correct = []

        # This iterates through the path to the leaf, starting from root
        for i in range(1, len(path)):
            node = path[i]
            node_id = int(node.split(" ")[0])
            parent = get_error_node_summary(error_analyzer, node_id)
            incorrect.append(parent["n_errors"])
            correct.append(parent["n_corrects"])
            nodes_to_leaf.append(" ".join(parent["path_to_leaf"][-1].split(" ")[1:]))

        # Add the leaf
        incorrect.append(leaves_summary[0]["n_errors"])
        correct.append(leaves_summary[0]["n_corrects"])
        nodes_to_leaf.append(
            " ".join(leaves_summary[0]["path_to_leaf"][-1].split(" ")[1:])
        )

        # Create stacked barchart
        fig, ax = plt.subplots()
        left = np.zeros(len(nodes_to_leaf))
        ax.barh(
            nodes_to_leaf[::-1],
            incorrect[::-1],
            0.5,
            label="incorrect",
            left=left,
            color="#CE1228",
        )
        left += incorrect[::-1]
        ax.barh(
            nodes_to_leaf[::-1],
            correct[::-1],
            0.5,
            label="correct",
            left=left,
            color="#DDDDDD",
        )
        ax.legend()
        ax.set_xlabel("Patient Counts")
        plt.show()
        log_figure(
            fig, join("img_artifacts", "error_viz", f"{prefix}_tree_summary.svg")
        )

        # TODO: tie this to feature importance top-k-features? (use same k)
        error_viz.plot_feature_distributions_on_leaves(
            leaf_selector=leaf_id, top_k_features=5
        )

        log_figure(
            plt.gcf(), join("img_artifacts", "error_viz", f"{prefix}_leave_dists")
        )

    except RuntimeError:
        # all predictions are correct no error analysis, skip
        pass


# Below is adapted from the original mealy with support for obtaining information along a path to a node
# Default mealy does not support 2 things
# 1. When calling get_leaf_summary, it does not return the IDs of the nodes along the path
# 2. Can not get the summary of any node, only leaves


def get_error_node_summary(error_analyzer, leaf_id):

    n_errors = int(
        error_analyzer.error_tree.estimator_.tree_.value[
            leaf_id, 0, error_analyzer.error_tree.error_class_idx
        ]
    )
    n_samples = error_analyzer.error_tree.estimator_.tree_.n_node_samples[leaf_id]
    local_error = n_errors / n_samples
    total_error_fraction = n_errors / error_analyzer.error_tree.n_total_errors
    n_corrects = n_samples - n_errors

    leaf_dict = {
        "id": leaf_id,
        "n_corrects": n_corrects,
        "n_errors": n_errors,
        "local_error": local_error,
        "total_error_fraction": total_error_fraction,
    }

    leaf_dict["path_to_leaf"] = get_path_to_node(error_analyzer, leaf_id)

    return leaf_dict


def get_leaf_summary(error_analyzer, leaf_id):

    leaf_nodes = error_analyzer._get_ranked_leaf_ids(leaf_selector=leaf_id)

    leaves_summary = []
    for leaf_id in leaf_nodes:
        n_errors = int(
            error_analyzer.error_tree.estimator_.tree_.value[
                leaf_id, 0, error_analyzer.error_tree.error_class_idx
            ]
        )
        n_samples = error_analyzer.error_tree.estimator_.tree_.n_node_samples[leaf_id]
        local_error = n_errors / n_samples
        total_error_fraction = n_errors / error_analyzer.error_tree.n_total_errors
        n_corrects = n_samples - n_errors

        leaf_dict = {
            "id": leaf_id,
            "n_corrects": n_corrects,
            "n_errors": n_errors,
            "local_error": local_error,
            "total_error_fraction": total_error_fraction,
        }

        leaf_dict["path_to_leaf"] = get_path_to_node(error_analyzer, leaf_id)

        leaves_summary.append(leaf_dict)

    return leaves_summary


def get_path_to_node(error_analyzer, node_id):
    """Return path to node as a list of split steps from the nodes of the sklearn Tree object"""
    feature_names = error_analyzer.pipeline_preprocessor.get_original_feature_names()
    children_left = list(error_analyzer.error_tree.estimator_.tree_.children_left)
    children_right = list(error_analyzer.error_tree.estimator_.tree_.children_right)
    threshold = error_analyzer._inverse_transform_thresholds()
    feature = error_analyzer._inverse_transform_features()

    cur_node_id = node_id
    path_to_node = collections.deque()
    while cur_node_id > 0:

        node_is_left_child = cur_node_id in children_left
        if node_is_left_child:
            parent_id = children_left.index(cur_node_id)
        else:
            parent_id = children_right.index(cur_node_id)

        feat = feature[parent_id]
        thresh = threshold[parent_id]

        is_categorical = error_analyzer.pipeline_preprocessor.is_categorical(feat)
        thresh = str(thresh) if is_categorical else format_float(thresh, 2)

        decision_rule = ""
        if node_is_left_child:
            decision_rule += " <= " if not is_categorical else " is not "
        else:
            decision_rule += " > " if not is_categorical else " is "

        decision_rule = (
            str(parent_id) + " " + str(feature_names[feat]) + decision_rule + thresh
        )
        path_to_node.appendleft(decision_rule)
        cur_node_id = parent_id

    return path_to_node


def format_float(number, decimals):
    """
    Format a number to have the required number of decimals. Ensure no trailing zeros remain.

    Args:
        number (float or int): The number to format
        decimals (int): The number of decimals required

    Return:
        formatted (str): The number as a formatted string

    """
    formatted = ("{:." + str(decimals) + "f}").format(number).rstrip("0")
    if formatted.endswith("."):
        return formatted[:-1]
    return formatted
