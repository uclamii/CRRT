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


def load_data(
    data: SklearnCRRTDataModule,
    model: CRRTStaticPredictor,
    args: Namespace,
    load_local: bool = False,
    save_local: bool = False,
) -> Tuple[SklearnCRRTDataModule, CRRTStaticPredictor]:
    # Load saved model and data files

    # Load from local directory
    if load_local:
        (
            reference_ids,
            original_columns,
            data_transform,
        ) = data.load_data_params(LOCAL_DATA_DIR)
        model.load_model(LOCAL_MODEL_DIR)

    # Requires a best_model_path
    else:
        (
            reference_ids,
            original_columns,
            data_transform,
        ) = data.load_data_params(join(args.best_model_path, "static_data"))
        model.load_model(join(args.best_model_path, "static_model"))

    # Setup the data
    data.setup(
        args,
        reference_ids=reference_ids,
        reference_cols=original_columns,
        data_transform=data_transform,
    )

    # Save locally
    if save_local:
        model.save_model(LOCAL_MODEL_DIR)
        data.dump_data_params(LOCAL_DATA_DIR)

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

        if args.rolling_evaluation:
            # if tuning the best model will never be dumped, so we dump it on the evaluation of the best model on original reference window
            # also save for non-tuning experiments. redundant if recently trained, but useful if wanting to evaluate on any arbitrary experiment

            # If running all experiments together (train and then immediately eval), then should never have to set reference_window since this is done automatically
            # Cases
            # 1. Tuning & rolling
            #   After training, automatically performs final eval run, sets args.reference_window and args.best_model_path, and saves local
            # 2. Not tuning & rolling
            #   After training, automatically saves locally if rolling_evaluation is set. Don't need reference_window
            # If wanting to perform post-hoc rolling window evaluation on a previous training run, then should set reference_window on the first evaluation
            if args.reference_window:
                data, model = load_data(
                    data, model, args, load_local=False, save_local=True
                )

            # Override with already trained model
            else:
                data, model = load_data(
                    data, model, args, load_local=True, save_local=False
                )

        # Override with already trained model
        else:
            data, model = load_data(
                data, model, args, load_local=False, save_local=False
            )
        return model.evaluate(data, "test")
    else:  # Training / tuning
        data.setup(args)

        model.fit(data)

        # Some redundancy here for saving models
        # For tuning, mlflow is basically always on so log the model
        if mlflow.active_run() is None:
            model.save_model(LOCAL_MODEL_DIR)
            data.dump_data_params(LOCAL_DATA_DIR)
        else:
            model.log_model()
            data.dump_data_params()

        # only want to serialize artifacts on training if we're not tuning and we're training the model on the original reference window (not slided yet)
        # functionally slide_window_by = 0 == None, but 0 indicates there will be sliding in the future
        if args.rolling_evaluation:
            if not args.tune_n_trials:
                model.save_model(LOCAL_MODEL_DIR)
                data.dump_data_params(LOCAL_DATA_DIR)

        # informs tuning, different from testing/eval
        return model.evaluate(data, "val")
        # model.evaluate(data, "train")
