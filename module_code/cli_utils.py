from argparse import ArgumentParser, Namespace, SUPPRESS
import inspect
from os.path import isfile
import sys
import yaml
from typing import Optional, Union, List

from data.argparse_utils import YAMLStringDictToDict
from data.torch_loaders import TorchCRRTDataModule
from data.sklearn_loaders import SklearnCRRTDataModule
from models.longitudinal_models import LongitudinalModel
from models.static_models import METRIC_MAP, StaticModel


def add_global_args(
    p: ArgumentParser, suppress_default: bool = False
) -> ArgumentParser:
    p.add_argument(
        "--seed",
        type=int,
        default=41,
        help="Seed used for reproducing results.",
    )
    p.add_argument(
        "--experiment",
        type=str,
        choices=["run_cv", "ctn_learning", "static_learning"],
        help="Name of method to run in experiments directory. Name must match exactly. Used to set experiment name in mlflow",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        help="Path to directory that contains the raw or unprocessed data files for one-off scripts.",
    )
    p.add_argument(
        "--ucla-crrt-data-dir",
        type=str,
        help="Path to directory that contains the data table files for CRRT cohort at UCLA.",
    )
    p.add_argument(
        "--ucla-control-data-dir",
        type=str,
        help="Path to directory that contains the data table files for control cohort at UCLA.",
    )
    p.add_argument(
        "--cedars-crrt-data-dir",
        type=str,
        help="Path to directory that contains the data table files for CRRT cohort at Cedars.",
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
    p.add_argument(
        "--controls-window-size",
        type=int,
        default=None,
        help="When processing controls, set the number of days you want the window size to be. This should align with the window size of CRRT patients.",
    )
    p.add_argument(
        "--patient-age",
        type=str,
        choices=["adult", "peds"],
        default="adult",
        help="Whether to evaluate on pediatric or adult patients (thresholded at age 21).",
    )
    p.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["train", "tune", "eval"],
        help="Distinguish a run where each run is a separate stage (intead of all in one). e.g., running training separate from evaluation.",
    )
    p.add_argument(
        "--best-model-path",
        type=str,
        default=None,
        help="Path to the model if stage == 'eval'.",
    )

    # Params for generating preprocessed df file
    time_p = p.add_argument_group("Time Interval and Windowing")
    time_p.add_argument(
        "--time-interval",
        type=str,
        default=None,
        help="Time interval in which to aggregate the raw data for preprocessing (formatted for pandas.resample()).",
    )
    time_p.add_argument(
        "--pre-start-delta",
        type=str,  # will be dict, to str (l29), convert to dict again
        action=YAMLStringDictToDict(),
        default=None,
        help="Dictionary of 'YEARS', 'MONTHS', and 'DAYS' (time) to specify offset of days before the start date to set the start point of the time window.",
    )
    time_p.add_argument(
        "--post-start-delta",
        type=str,  # will be dict, to str (l29), convert to dict again
        action=YAMLStringDictToDict(),
        default=None,
        help="Dictionary of 'YEARS', 'MONTHS', and 'DAYS' (time) to specify offset of days after the start date to set the end point of the time window.",
    )
    time_p.add_argument(
        "--time-window-end",
        type=str,
        default="Start Date",
        choices=["Start Date", "End Date"],
        help="Specifying if to preprocess data only until the start date of CRRT, or to go all the way through the end date of CRRT, will be used if no post-start-delta passed.",
    )
    time_p.add_argument(
        "--rolling-evaluation",
        default=False,
        type=bool,
        help="Flag for specifying if performing rolling windown analysis",
    )
    time_p.add_argument(
        "--reference-window",
        default=False,
        type=bool,
        help="If rolling-evaluation is True, this flag specifies which run is the window for training",
    )
    time_p.add_argument(
        "--new-eval-cohort",
        default=False,
        type=bool,
        help="Flag for specifying if evaluating on a new cohort",
    )
    time_p.add_argument(
        "--slide-window-by",
        type=int,
        default=None,
        help="If doing a rolling window analysis, this is the integer number of days to slide the time window mask forward. None means no sliding. 0 means none now but the following runs will be.",
    )
    time_p.add_argument(
        "--max-days-on-crrt",
        type=int,
        default=None,
        help="If doing rolling window analysis, we want to know the maximum slide, AKA the maximum days we allow someone to be on CRRT for this analysis. E.g., if max slide is 3 days, it doesn't make sense to include someone who has 20 days on CRRT for the rolling analysis as their outcome is so much farther out.",
    )
    time_p.add_argument(
        "--min-days-on-crrt",
        type=int,
        default=0,
        help="If subgrouping, we may want to know the minimum slide, AKA the minimum days we allow someone to be on CRRT for this analysis.",
    )

    # Logging / Tracking
    logging_p = p.add_argument_group("Logging / Tracking")
    logging_p.add_argument(
        "--experiment-tracking", type=bool, help="Toggles experiment tracking on."
    )  # Note: presence regardless of value will toggle on, e.g. "var: False"
    logging_p.add_argument(
        "--local-log-path",
        type=str,
        help="Location on local machine to store logs (including mlflow experiments).",
    )
    logging_p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of run under a tracked experiment (logged to mlflow).",
    )
    logging_p.add_argument(
        "--runtest",
        type=bool,
        default=False,
        help="Whether or not to run testing on the predictive model.",
    )
    logging_p.add_argument(
        "--tune-n-trials",
        type=int,
        default=None,
        help="Set to integer value to turn on Optuna tuning. Find the settings for tuning in module/exp/utils.py. It will set the number of hyperparameters to try PER predictive model. So if we're evaluating 5 predictive models, tune-n-trials==1 will run 5 experiments. if ==2, it will run 10, etc.",
    )
    logging_p.add_argument(
        "--tune-metric",
        type=str,
        default="auroc",
        choices=list(METRIC_MAP.keys()),
        help="Name of metric to use to select best trial when hyperparameter tuning.",
    )
    logging_p.add_argument(
        "--tune-direction",
        type=str,
        default="maximize",
        choices=["maximize", "minimize"],
        help="Whether the best metric is the max value or the min value.",
    )

    # To be able to add these to the subparsers without conflicts
    # Ref: https://stackoverflow.com/a/62906328/1888794
    if suppress_default:
        for action in p._actions:
            action.default = SUPPRESS

    return p


def load_cli_args(args_options_path: str = "options.yml"):
    """
    Modify command line args if desired, or load from YAML file.
    """
    if isfile(args_options_path):  # if file exists
        with open(args_options_path, "r") as f:
            res = yaml.safe_load(f)

        # set/override cli args below
        """
        res = { "run-name": "placeholder", }
        """

        # sys.argv = [sys.argv[0]]

        # add as a positional arg/command to control subparsers (instead of flag)
        # don't remove so that experiment tracking automatically logs
        if "model-type" in res:
            sys.argv.insert(1, res["model-type"])

        for k, v in res.items():
            if f"--{k}" not in sys.argv:
                sys.argv += [f"--{k}", str(v)]


def remove_help_action(p: ArgumentParser) -> ArgumentParser:
    """
    Help action will conflict with child parsers and subsequent suparsers.
    This allows us to get rid of it no matter what the parent was initialized with.
    """
    help_action = next(  # ref: https://stackoverflow.com/a/7125547/1888794
        (action for action in p._actions if action.dest == "help"), None
    )
    if help_action is not None:
        p._remove_action(help_action)

    # return p


def init_cli_args(p: Optional[ArgumentParser] = None) -> Namespace:
    """
    Parse commandline args needed to run experiments.
    Basically mostly hyperparams.
    Word for the wise: probabaly just use `click` in the future instead of `argparse`.
    """
    if p is None:
        p = ArgumentParser(add_help=False)
    else:  # Manually get rid of help action if it exists (causes conflict)
        remove_help_action(p)
    global_p = add_global_args(p)

    # Ref: passing global args through https://stackoverflow.com/a/56595689/1888794
    parser_main = ArgumentParser(parents=[global_p])
    # Allows subcommands, so args are only added based on the command
    # Ref: https://docs.python.org/3/library/argparse.html#sub-commands
    subparsers = parser_main.add_subparsers(dest="model_type", help="Model types.")
    dynamic_parser = subparsers.add_parser(
        "dynamic", help="Dynamic Model", parents=[global_p]
    )
    dynamic_parser = TorchCRRTDataModule.add_data_args(dynamic_parser)
    dynamic_parser = LongitudinalModel.add_model_args(dynamic_parser)

    static_parser = subparsers.add_parser(
        "static", help="Static Model", parents=[global_p]
    )
    static_parser = SklearnCRRTDataModule.add_data_args(static_parser)
    static_parser = StaticModel.add_model_args(static_parser)

    # return parser_main.parse_args()
    # Ignore unrecognized args
    return parser_main.parse_known_args()[0]
