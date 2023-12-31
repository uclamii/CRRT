from argparse import Namespace
from os.path import join
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import Run
from optuna import Study, create_study
from optuna.trial import FrozenTrial
from optuna.samplers import TPESampler, RandomSampler

from exp.cv import run_cv
from exp.static_learning import static_learning
from exp.ctn_learning import continuous_learning
from exp.utils import get_optuna_grid, time_delta_str_to_dict
from models.static_models import ALG_MAP, STATIC_MODEL_FNAME
from cli_utils import load_cli_args, init_cli_args


def run_experiment(args: Namespace, trials=None):
    # Ref: https://github.com/optuna/optuna/issues/862
    # trials will override any other updating of params from CLI or options.yml because it comes last after load/init args.
    if trials is not None:
        params = get_optuna_grid(args.modeln, args.experiment, trials)
        # update args in place
        dargs = vars(args)
        dargs.update(params)

    # some require args and some don't
    experiment_name_to_function = {
        "run_cv": {"fn": run_cv, "args": (args,)},
        "static_learning": {"fn": static_learning, "args": (args,)},
        "ctn_learning": {"fn": continuous_learning, "args": (args,)},
    }
    experiment_function = experiment_name_to_function[args.experiment]["fn"]
    experiment_args = experiment_name_to_function[args.experiment]["args"]

    if args.experiment_tracking:
        mlflow.set_tracking_uri(f"file://{join(args.local_log_path, 'mlruns')}")
        mlflow.set_experiment(experiment_name=args.experiment)
        # Autologging
        mlflow.autolog(log_models=False)

        # Adjusting run name if tuning/evaluating/etc.
        run_name = args.run_name
        # In eval mode after tuning
        if args.tune_n_trials and args.stage == "eval":
            run_name += f" // eval best"
        elif args.tune_n_trials:  # Just tuning
            run_name += f" // tune trial: {trials.number}"
        elif args.new_eval_cohort and args.rolling_evaluation:
            if args.reference_window:
                if args.min_days_on_crrt > 0:
                    run_name += (
                        f" // post_eval_{args.eval_cohort}_{args.min_days_on_crrt} best"
                    )
                else:
                    run_name += f" // post_eval_{args.eval_cohort} best"
            else:
                run_name += f" // post_eval_{args.eval_cohort} rolling window: {args.slide_window_by}"
        elif args.rolling_evaluation and args.stage == "eval":
            run_name += f" // rolling window: {args.slide_window_by}"

        with mlflow.start_run(run_name=run_name):
            # Log all cli args as tags
            mlflow.set_tags(vars(args))
            # run experiment
            results_dict = experiment_function(*experiment_args)
    else:
        results_dict = experiment_function(*experiment_args)
    if (
        args.tune_n_trials and args.stage != "eval"
    ):  # So Optuna can compare and pick best trial based on validation
        return results_dict[f"{args.modeln}_val__{args.tune_metric}"]
    return results_dict


def get_mlflow_model_uri(best_run: Run) -> str:
    return best_run.info.artifact_uri[len("file://") :]


def get_best_trial_mlflow_run(
    args: Namespace, best_trial: FrozenTrial, serialize: bool = False
) -> Run:
    """Get the mlflow run id of the best optuna trial for evaluation."""
    best_args = best_trial.params
    client = MlflowClient(f"file://{join(args.local_log_path, 'mlruns')}")
    # Get the trial based on the number (0 indexed)
    # Get most recent hyperparameter trial for the given run name
    best_run = client.search_runs(
        experiment_ids=client.get_experiment_by_name(args.experiment).experiment_id,
        filter_string=f"tags.mlflow.runName='{args.run_name} // tune trial: {best_trial.number}'",
        order_by=["attributes.start_time DESC"],
    )[0]
    return best_run


def evaluate_post_tuning(args: Namespace, study: Study):
    best_trial = study.best_trial
    best_run = get_best_trial_mlflow_run(args, best_trial)
    # Update the delta since it would be type str
    best_trial.params["pre_start_delta"] = time_delta_str_to_dict(
        best_trial.params["pre_start_delta"]
    )
    # Update with best params and with the run id.
    best_model_path = get_mlflow_model_uri(best_run)
    dargs = vars(args)
    # It's fine to add to args since this is the last run and it won't sully the previous trials (or "coming" runs)
    modeln = best_run.data.tags["modeln"]
    # split the best params into the ones that should be in model_kwargs and not
    top_level_params = {}
    model_kwargs = {}
    for param_name, param_val in best_trial.params.items():
        if param_name.startswith(modeln):
            # exclude the rf_ if modeln is rf
            raw_name = param_name[len(f"{modeln}") + 1 :]
            model_kwargs[raw_name] = param_val
        else:
            top_level_params[param_name] = param_val

    dargs.update(
        {
            **top_level_params,
            # modeln is selected outside of optuna so it wont be in params
            "modeln": modeln,
            # model_kwargs in best_trial.params but flattened out
            "model_kwargs": model_kwargs,
            "best_run_id": best_run.info.run_id,
            "best_model_path": best_model_path,
            "stage": "eval",
            "reference_window": True,
            "rolling_evaluation": True,  # saves to local
        }
    )
    # Run
    run_experiment(args)


def main(args):
    # Optionally run tuning, then evaluate
    if args.tune_n_trials:
        study = create_study(
            study_name=args.experiment,
            direction=args.tune_direction,
            sampler=RandomSampler(seed=args.seed),
        )
        for modeln in ALG_MAP.keys():
            # for modeln in ["lgr"]:
            args.modeln = modeln
            # Ref: https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments
            study.optimize(
                lambda trial: run_experiment(args, trial), n_trials=args.tune_n_trials
            )

        # Evaluate mode
        evaluate_post_tuning(args, study)
    else:
        run_experiment(args)


if __name__ == "__main__":
    load_cli_args()
    args = init_cli_args()
    main(args)
