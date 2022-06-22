import os
import torch

from utils import time_delta_str_to_dict


def seed_everything(seed: int):
    """Sets seeds and also makes cuda deterministic for pytorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optuna_grid(experiment_name: str, trials):
    if experiment_name == "static_learning":
        feature_selection_method = trials.suggest_categorical(
            "feature_selection", ["top-k", "corr_thresh"]
        )
        params = {
            "pre_start_delta": time_delta_str_to_dict(
                trials.suggest_categorical(
                    "pre_start_delta", ["3m", "1m", "14d", "7d", "5d", "3d", "1d"]
                )
            ),
        }
        if feature_selection_method == "top-k":
            params["top_k_feature_importance"] = trials.suggest_int(
                "top_k_feature_importance", 3, 20, step=5
            )
        else:
            params["corr_thresh"] = trials.suggest_float(
                "corr_thresh", 0.1, 0.9, step=0.1
            )
        return params
