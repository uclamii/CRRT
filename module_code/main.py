from argparse import Namespace
from os.path import join
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from optuna import create_study, Study
from optuna.samplers import TPESampler
import yaml

from data.load import load_data
from exp.cv import run_cv
from exp.static_learning import static_learning
from exp.ctn_learning import continuous_learning
from exp.utils import get_optuna_grid
from utils import load_cli_args, init_cli_args


def main(args: Namespace, trials=None):
    # Ref: https://github.com/optuna/optuna/issues/862
    # trials will override any other updating of params from CLI or options.yml because it comes last after load/init args.
    if trials is not None:
        params = get_optuna_grid(args.experiment, trials)
        # update args in place
        dargs = vars(args)
        dargs.update(params)

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

        run_name = args.run_name
        if args.tune_n_trials:
            run_name += f" // tune trial: {trials.number}"
        with mlflow.start_run(run_name=run_name):
            # Log all cli args as tags
            mlflow.set_tags(vars(args))
            # run experiment
            results_dict = experiment_function(preprocessed_df, *experiment_args)
    else:
        results_dict = experiment_function(preprocessed_df, *experiment_args)
    # So Optuna can compare and pick best trial
    return results_dict[f"{args.modeln}_val__{args.tune_metric}"]


def store_best_trial_mlflow_run(args: Namespace, study: Study):
    """Store the mlflow run id of the best optuna trial for evaluation later."""
    best_trial = study.best_trial
    client = MlflowClient(join(args.local_log_path, "mlruns"))
    # Get the trial based on the number
    # Get most recent hyperparameter trial for the given run name
    best_run = client.search_runs(
        experiment_ids=client.get_experiment_by_name(args.experiment).experiment_id,
        filter_string=f"tags.mlflow.runName='{args.run_name} // tune trial: {best_trial.number}'",
        order_by=["attributes.start_time"],
    )[0]
    # Add or update mlflow run id to the options.yml file
    with open(join(args.local_log_path, "tune_results.yml"), "w") as tune_results:
        yaml.safe_dump({"best-run-id": best_run.info.run_id}, tune_results)


if __name__ == "__main__":
    load_cli_args()
    args = init_cli_args()

    # Optionally run tuning
    if args.tune_n_trials:
        study = create_study(
            study_name=args.experiment,
            direction=args.tune_direction,
            sampler=TPESampler(),
        )
        # Ref: https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments
        study.optimize(lambda trial: main(args, trial), n_trials=args.tune_n_trials)

        store_best_trial_mlflow_run(args, study)
    else:
        main(args)
