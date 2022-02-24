from argparse import ArgumentParser, Namespace
from os.path import isfile
import sys
import yaml
from typing import Dict, Optional

from data.argparse_utils import YAMLStringDictToDict
from data.torch_loaders import TorchCRRTDataModule
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
        "--pre-start-delta",
        type=str,  # will be dict, to str (l30), convert to dict again
        action=YAMLStringDictToDict(),
        default=None,
        help="Dictionary of 'YEARS', 'MONTHS', and 'DAYS' (time) to specify offset of days before the start date to set the start point of the time window.",
    )
    p.add_argument(
        "--post-start-delta",
        type=str,  # will be dict, to str (l30), convert to dict again
        action=YAMLStringDictToDict(),
        default=None,
        help="Dictionary of 'YEARS', 'MONTHS', and 'DAYS' (time) to specify offset of days after the start date to set the end point of the time window.",
    )
    p.add_argument(
        "--time-window-end",
        type=str,
        default="Start Date",
        choices=["Start Date", "End Date"],
        help="Specifying if to preprocess data only until the start date of CRRT, or to go all the way through the end date of CRRT, will be used if no post-start-delta passed.",
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
    """
    This needs to be updated for whatever options were chosen
    """
    # add args for pytorch lightning datamodule
    p = CRRTDataModule.add_data_args(p)
    # add args for pytorch lightning model
    p = LongitudinalModel.add_model_args(p)

    # return p.parse_args()
    # Ignore unrecognized args
    return p.parse_known_args()[0]


def time_delta_to_str(delta: Dict[str, int]) -> str:
    """
    Coverts timedelta dict to str form: 5 years, 4 months, and 3 days => 5y4m3d
    Ignore values of 0: 4months and 3 days => 4m3d
    Assumes order of keys are: years, months, then days.
    """
    delta_str = ""
    for time_name, amount in delta.items():
        if amount > 0:
            delta_str += f"{amount}{time_name[0].lower()}"
    return delta_str


def get_preprocessed_file_name(
    pre_start_delta: Optional[Dict[str, int]] = None,
    post_start_delta: Optional[Dict[str, int]] = None,
    time_interval: Optional[str] = None,
    time_window_end: Optional[str] = None,
    preprocessed_df_file: Optional[str] = None,
    serialization: str = "feather",
) -> str:
    """
    Uses preprocessed_df_file for file name for preprocessed dataframe.
    However, if it's not provided it will automatically generate a name based on the arguments used to generate the file.

    df_{time interval the features are aggregated in}agg_[{time window start},{time window end}].extension
    If providing deltas: [startdate-pre_start_delta,startdate+post_start_delta]
    If providing neither [startdate,time_window_end].
    """
    if preprocessed_df_file:
        return preprocessed_df_file + f".{serialization}"
    fname = "df"
    if time_interval:
        fname += f"_{time_interval}agg"
    # time window
    fname += "_[startdate"
    if pre_start_delta:
        # subtracting the delta time
        fname += f"-{time_delta_to_str(pre_start_delta)}"
    fname += ","
    # end of window:
    if post_start_delta:
        fname += f"startdate+{time_delta_to_str(post_start_delta)}]"
    else:
        fname += f"{time_window_end.replace(' ', '').lower()}]"

    return fname + "." + serialization
