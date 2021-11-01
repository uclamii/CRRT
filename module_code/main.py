import logging
from os.path import join
import sys
import pandas as pd
import time
import mlflow
import mlflow.pytorch

from data.preprocess import preprocess_data
from data.load import merge_features_with_outcome
from exp.cv import run_cv
from exp.ctn_learning import continuous_learning
from utils import get_preprocessed_file_name, load_cli_args, init_cli_args


if __name__ == "__main__":
    load_cli_args()
    args = init_cli_args()
    preprocessed_df_fname = get_preprocessed_file_name(
        args.time_before_start_date,
        args.time_interval,
        args.preprocessed_df_file,
        serialization=args.serialization,
    )
    preprocessed_df_path = join(args.raw_data_dir, preprocessed_df_fname)

    try:
        deserialize_fn = getattr(pd, f"read_{args.serialization}")
        # raise IOError
        df = deserialize_fn(preprocessed_df_path)
    except IOError:
        # Keep a log of how preprocessing went. can call logger anywhere inside of logic from here
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            # print to stdout and log to file
            handlers=[
                logging.FileHandler("dialysis_preproc.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.info("Preprocessed file does not exist! Creating...")
        start_time = time.time()
        df = merge_features_with_outcome(
            args.raw_data_dir,
            args.time_interval,
            args.time_before_start_date,
            args.time_window_end,
        )  # 140s ~2.5 mins, 376.5s ~6mins for daily aggregation
        logging.info(f"Loading took {time.time() - start_time} seconds.")
        serialize_fn = getattr(df, f"to_{args.serialization}")
        serialize_fn(preprocessed_df_path)

    preprocessed_df = preprocess_data(df)
    experiment_name_to_function = {
        "run_cv": {"fn": run_cv, "args": ()},
        "ctn_learning": {"fn": continuous_learning, "args": (args,)},
    }
    experiment_function = experiment_name_to_function[args.experiment]["fn"]
    experiment_args = experiment_name_to_function[args.experiment]["args"]

    if args.experiment_tracking:
        mlflow.set_tracking_uri(f"file://{join(args.local_log_path, 'mlruns')}")
        mlflow.set_experiment(experiment_name=args.experiment)
        # Autologging
        mlflow.pytorch.autolog()

        with mlflow.start_run(run_name=args.run_name):
            # Log all cli args as tags
            mlflow.set_tags(vars(args))
            # run experiment
            results_dict = experiment_function(preprocessed_df, *experiment_args)
            # mlflow.log_metrics(results_dict["mean_scores"])
            # mlflow.log_metrics(results_dict["std_scores"])
    else:
        results_dict = experiment_function(preprocessed_df, *experiment_args)
