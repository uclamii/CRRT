from os.path import join
import mlflow.pytorch

from data.load import load_data
from exp.cv import run_cv
from exp.static_learning import static_learning
from exp.ctn_learning import continuous_learning
from utils import load_cli_args, init_cli_args

if __name__ == "__main__":
    load_cli_args()
    args = init_cli_args()
    preprocessed_df = load_data(args)
    experiment_name_to_function = {
        "run_cv": {"fn": run_cv, "args": ()},
        "static_learning": {"fn": static_learning, "args": (args,)},
        "ctn_learning": {"fn": continuous_learning, "args": (args,)},
    }
    experiment_function = experiment_name_to_function[args.experiment]["fn"]
    experiment_args = experiment_name_to_function[args.experiment]["args"]

    if args.experiment_tracking:
        mlflow.set_tracking_uri(f"file://{join(args.local_log_path, 'mlruns')}")
        mlflow.set_experiment(experiment_name=args.experiment)
        # Autologging
        mlflow.autolog()

        with mlflow.start_run(run_name=args.run_name):
            # Log all cli args as tags
            mlflow.set_tags(vars(args))
            # run experiment
            results_dict = experiment_function(preprocessed_df, *experiment_args)
    else:
        results_dict = experiment_function(preprocessed_df, *experiment_args)
