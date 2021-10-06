from argparse import ArgumentParser, Namespace
import sys
from os.path import isfile
import yaml


def load_cli_args(args_options_path: str = "options.yml"):
    """
    Modify command line args if desired, or load from YAML file.
    """
    if len(sys.argv) <= 3:
        if isfile(args_options_path):  # if file exists
            with open(args_options_path, "r") as f:
                res = yaml.safe_load(f)

    # set/override cli args below
    res = {
        "run-name": "placeholder",
        "experiment": "online_learning",
        # "experiment-tracking": True,  # any value works, comment line to toggle off
        # TODO: Change this to only be in options.yml so we're not pushing/pulling our own directories everytime
        "raw-data-dir": "/home/davina/Private/dialysis-data"
        # "raw-data-dir": r"C:\Users\arvin\Documents\ucla research\CRRT project\data_files"
    }

    sys.argv = [sys.argv[0]]
    for k, v in res.items():
        sys.argv += [f"--{k}", str(v)]


def init_cli_args() -> Namespace:
    """
    Parse commandline args needed to run experiments.
    Basically mostly hyperparams.
    """

    p = ArgumentParser()
    p.add_argument(
        "--seed", type=int, default=42, help="Seed used for reproducing results.",
    )
    p.add_argument(
        "--experiment",
        type=str,
        choices=["run_cv", "online_learning"],
        help="Name of method to run in experiments directory. Name must match exactly. Used to set experiment name in mlflow",
    )
    p.add_argument(
        "--raw-data-dir",
        type=str,
        help="Path to directory that contains the raw or unprocessed data files.",
    )

    # Logging / Tracking
    p.add_argument(
        "--experiment-tracking", type=bool, help="Toggles experiment tracking on."
    )  # Note: presence regardless of value will toggle on, e.g. "var: False"
    p.add_argument(
        "--local-log-path",
        type=str,
        help="Location on local machine to store logs (including mlflow experiments).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of run under a tracked experiment (logged to mlflow).",
    )

    # return p.parse_args()
    # Ignore unrecognized args
    return p.parse_known_args()[0]
