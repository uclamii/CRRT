from argparse import Namespace
from functools import reduce
import pickle
from typing import Any, Callable, List, Tuple, Union, Dict
import cloudpickle  # Allows serializing functions
import pandas as pd
from posixpath import join
import mlflow

from data.sklearn_loaders import SklearnCRRTDataModule, LOCAL_DATA_DIR
from models.static_models import CRRTStaticPredictor, LOCAL_MODEL_DIR
from data.subpopulation_utils import generate_filters


def load_saved_data(
    data: SklearnCRRTDataModule, args: Namespace, load_local: bool = False
) -> SklearnCRRTDataModule:
    # Load from local directory
    if load_local:
        (
            reference_ids,
            original_columns,
            data_transform,
        ) = data.load_data_params(join(LOCAL_DATA_DIR, args.run_name, "static_data"))
    # Requires a best_model_path
    else:
        (
            reference_ids,
            original_columns,
            data_transform,
        ) = data.load_data_params(join(args.best_model_path, "static_data"))

    if args.new_eval_cohort and args.reference_window:
        reference_ids = None

    data.setup(
        args,
        reference_ids=reference_ids,
        reference_cols=original_columns,
        data_transform=data_transform,
    )
    return data


def load_saved_model(
    model: CRRTStaticPredictor, args: Namespace, load_local: bool = False
) -> CRRTStaticPredictor:
    # Load from local directory
    if load_local:
        model.load_model(join(LOCAL_MODEL_DIR, args.run_name, "static_model"))
    # Requires a best_model_path
    else:
        model.load_model(join(args.best_model_path, "static_model"))

    return model


def load_saved_data_and_model(
    data: SklearnCRRTDataModule,
    model: CRRTStaticPredictor,
    args: Namespace,
    load_local: bool = False,
) -> Tuple[SklearnCRRTDataModule, CRRTStaticPredictor]:
    # Load saved model and data files

    data = load_saved_data(data, args, load_local)
    model = load_saved_model(model, args, load_local)

    return data, model


def static_learning(args: Namespace):
    # Create CRRTDataModule
    filters = generate_filters()
    filters = {
        k: v for groupname, d in filters.items() for k, v in d.items()
    }  # flatten
    data = SklearnCRRTDataModule.from_argparse_args(args, filters=generate_filters())

    # Then create the predictor
    model = CRRTStaticPredictor.from_argparse_args(args)

    if args.stage == "eval":
        # Load up trained portion and hparams - args.best_model_path required if reference_window is True
        # If running all experiments together (train and then immediately eval), then should never have to set reference_window since this is done automatically
        # Cases
        # 1. Tuning & rolling
        #   After training, automatically performs final eval run, sets args.reference_window and args.best_model_path, and saves local
        # 2. Not tuning & rolling
        #   After training, automatically saves locally if rolling_evaluation is set. Don't need reference_window
        data, model = load_saved_data_and_model(
            data,
            model,
            args,
            load_local=(args.rolling_evaluation and not args.reference_window),
        )

        # if tuning the best model will never be dumped, so we dump it on the evaluation of the best model on original reference window
        # also save for non-tuning experiments. redundant if recently trained, as explained in the comment above
        # but useful if wanting to evaluate on any arbitrary experiment.
        #   If wanting to perform post-hoc rolling window evaluation on a previous training run, then should set reference_window on the first evaluation
        if args.rolling_evaluation and args.reference_window:
            model.save_model(join(LOCAL_MODEL_DIR, args.run_name, "static_model"))
            data.dump_data_params(join(LOCAL_DATA_DIR, args.run_name, "static_data"))

        return model.evaluate(data, "test")
    else:  # Training / tuning
        data.setup(args)

        model.fit(data)

        # Some redundancy here for saving models
        # For tuning, mlflow is basically always on so log the model
        if mlflow.active_run() is None:
            model.save_model(join(LOCAL_MODEL_DIR, args.run_name, "static_model"))
            data.dump_data_params(join(LOCAL_DATA_DIR, args.run_name, "static_data"))
        else:
            model.log_model()
            data.dump_data_params()

        # only want to serialize artifacts on training if we're not tuning and we're training the model on the original reference window (not slided yet)
        # functionally slide_window_by = 0 == None, but 0 indicates there will be sliding in the future
        if args.rolling_evaluation:
            if not args.tune_n_trials:
                model.save_model(join(LOCAL_MODEL_DIR, args.run_name, "static_model"))
                data.dump_data_params(
                    join(LOCAL_DATA_DIR, args.run_name, "static_data")
                )

        # informs tuning, different from testing/eval
        return model.evaluate(data, "val")
        # model.evaluate(data, "train")
