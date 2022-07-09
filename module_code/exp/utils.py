from typing import Dict, Optional
import regex

# Grid of hyperparameters for each type of model
GRID_HP_MAP = {
    "lgr": {
        "penalty": ["l2", "elasticnet"],
        "C": [0.1, 1, 10, 100, 1000],
        "n_jobs": [-1],
    },
    "knn": {
        "weights": ["uniform", "distance"],
        "leaf_size": [20, 25, 30, 35, 40],
        "p": [1, 2],
        "metric": ["minkowski", "chebyshev"],
        "n_jobs": [-1],
    },
    "dt": {
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 30, 100],
        "max_features": ["auto", "sqrt"],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2, 5],
    },
    "rf": {
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 30, 100],
        "max_features": ["auto", "sqrt"],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2, 5],
        "bootstrap": [True, False],
        "n_estimators": list(range(10, 100, 10)) + list(range(100, 1050, 100)),
    },
    "lgb": {
        "num_leaves": [20, 40, 60, 80, 100],
        "min_child_samples": [5, 10, 15],
        "max_depth": [-1, 5, 10, 20],
        "learning_rate": [0.05, 0.1, 0.2],
        "reg_alpha": [0, 0.01, 0.03],
    },
    "xgb": {
        "learning_rate": [0.001, 0.05, 0.10, 0.15],
        "max_depth": [8, 10, 12],
        "min_child_weight": [1],
        "gamma": [0.5, 0.8, 1],
        "colsample_bytree": [0.5, 0.7, 0.9],
        # this is for fit and not for the classifier itself
        # "early_stopping_rounds": [10],
        "reg_alpha": [0.001, 0.05, 0.10, 0.15],
        "reg_lambda": [0.001, 0.05, 0.10, 0.15],
        "n_estimators": list(range(10, 100, 10))
        + list(range(100, 1550, 100))
        + list(range(2000, 10050, 1000)),
    },
}


def time_delta_str_to_dict(delta_str: Optional[str]) -> Optional[Dict[str, int]]:
    """
    Inverse of time_delta_to_str.
    Converts a str of format: YyMmDd for Y years M months and D days.
    Into a dict: {YEARS: Y, MONTHS: M, DAYS: D}
    """
    if delta_str:
        time_regex = r"(?:(?<YEARS>\d+)y)?(?:(?<MONTHS>\d+)m)?(?:(?<DAYS>\d+)d)?"
        return {
            k: int(v) if v else 0
            for k, v in regex.search(time_regex, delta_str).groupdict().items()
        }
    return None


def get_optuna_grid(modeln: str, experiment_name: str, trials):
    if experiment_name == "static_learning":
        feature_selection_method = trials.suggest_categorical(
            "feature_selection", ["kbest", "corr_thresh"]
        )

        params = {
            "pre_start_delta": time_delta_str_to_dict(
                trials.suggest_categorical(
                    "pre_start_delta", ["7d", "6d", "5d", "4d", "3d", "2d", "1d"]
                )
            ),
            # "modeln": modeln,
            # Since GRID_HP_MAP is just list of choices we'll use suggest_categorical
            "model_kwargs": {
                k: trials.suggest_categorical(f"{modeln}_{k}", v)
                # If we dont have it in the grid just use default kwargs (by setting {})
                for k, v in GRID_HP_MAP.get(modeln, {}).items()
            },
        }

        # we run trials back to back so set a value for one and clear the other out
        if feature_selection_method == "kbest":
            params["kbest"] = trials.suggest_int("kbest", 3, 18, step=5)
            params["corr_thresh"] = None
        else:
            params["corr_thresh"] = trials.suggest_float(
                # "corr_thresh", 0.1, 0.9, step=0.1
                "corr_thresh",
                0.01,
                0.09,
                step=0.005,
            )
            params["kbest"] = None

        return params
