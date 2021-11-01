from argparse import ArgumentParser, Namespace
from os.path import isfile
import sys
import yaml
from typing import Dict, Optional

from data.longitudinal_utils import TIME_BEFORE_START_DATE
from data.argparse_utils import YAMLStringDictToDict
from data.pytorch_loaders import CRRTDataModule
from models.longitudinal_models import LongitudinalModel


def load_cli_args(args_options_path: str = "options.yml"):
    """
    Modify command line args if desired, or load from YAML file.
    """
    if len(sys.argv) <= 3:
        if isfile(args_options_path):  # if file exists
            with open(args_options_path, "r") as f:
                res = yaml.safe_load(f)

            # set/override cli args below
            """
            res = {
                "run-name": "placeholder",
                # "experiment-tracking": True,  # any value works, comment line to toggle off
            }
            """

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
        choices=["run_cv", "ctn_learning"],
        help="Name of method to run in experiments directory. Name must match exactly. Used to set experiment name in mlflow",
    )
    p.add_argument(
        "--raw-data-dir",
        type=str,
        help="Path to directory that contains the raw or unprocessed data files.",
    )
    p.add_argument(
        "--preprocessed-df-file",
        type=str,
        help="Name of file that contains a serialized DataFrame of the preprocessed raw data.",
    )
    p.add_argument(
        "--serialization",
        type=str,
        # Feather does not allow serialization of MultiIndex dataframes
        choices=["feather", "parquet"],
        default="feather",
        help="Name of serialization method to use for preprocessed df file.",
    )

    # Params for generating preprocessed df file
    p.add_argument(
        "--time-interval",
        type=str,
        default=None,
        help="Time interval in which to aggregate the raw data for preprocessing (formatted for pandas.resample()).",
    )
    p.add_argument(
        "--time-before-start-date",
        type=str,  # will be dict, to str (l30), convert to dict again
        action=YAMLStringDictToDict(),
        default=TIME_BEFORE_START_DATE,
        help="Dictionary of 'YEARS', 'MONTHS', and 'DAYS' (time) to specify the start date of the window of data to look at.",
    )
    p.add_argument(
        "--time-window-end",
        type=str,
        default="Start Date",
        choices=["Start Date", "End Date"],
        help="Specifying if to preprocess data only until the start date of CRRT, or to go all the way through the end date of CRRT.",
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

    # add args for pytorch lightning datamodule
    p = CRRTDataModule.add_data_args(p)
    # add args for pytorch lightning model
    p = LongitudinalModel.add_model_args(p)

    # return p.parse_args()
    # Ignore unrecognized args
    return p.parse_known_args()[0]


def get_preprocessed_file_name(
    time_before_start_date: Optional[Dict[str, int]] = None,
    time_interval: Optional[str] = None,
    time_window_end: Optional[str] = None,
    preprocessed_df_file: Optional[str] = None,
    serialization: str = "feather",
) -> str:
    """
    Uses preprocessed_df_file for file name for preprocessed dataframe.
    However, if it's not provided it will automatically generate a name based on the arguments used to generate the file.

    df_{time interval the features are aggregated in}_{time window start before start date of crrt}_{end time of the time window}.extension
    """
    if preprocessed_df_file:
        return preprocessed_df_file
    fname = "df"
    if time_interval:
        fname += f"_{time_interval}agg"
    # time window
    fname += "_"
    for time_name, amount in time_before_start_date.items():
        # e.g. 14d for DAYS: 14, but ignore values of 0
        if amount > 0:
            fname += f"{amount}{time_name[0].lower()}"
    # end of window:
    fname += f"_{time_window_end.replace(' ', '').lower()}"

    return fname + "." + serialization
