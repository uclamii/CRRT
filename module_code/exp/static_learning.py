from argparse import Namespace
import pickle
import pandas as pd
from os.path import join

from data.sklearn_loaders import SklearnCRRTDataModule
from models.static_models import CRRTStaticPredictor

SPLIT_IDS_PATH = join("local_data", "split_ids.pkl")
COLUMNS_PATH = join("local_data", "columns.pkl")
MODEL_DIR = join("local_data", "static_model")


def dump_artifacts_for_rolling_windows(
    data: SklearnCRRTDataModule, model: CRRTStaticPredictor
):
    # this will dump the model D -> D+i, and D+1 retrained and then evaluated
    model.save_model(MODEL_DIR)  # will ensure path exists
    with open(SPLIT_IDS_PATH, "wb") as f:
        pickle.dump(data.split_pt_ids, f)
    with open(COLUMNS_PATH, "wb") as f:  # dump columns after feature selection
        pickle.dump(data.selected_columns_mask, f)


def static_learning(df: pd.DataFrame, args: Namespace):
    # need to update CRRTDataModule
    filters = {
        "heart": lambda df: df["heart_pt_indicator"] == 1,
        "liver": lambda df: df["liver_pt_indicator"] == 1,
        "infection": lambda df: df["infection_pt_indicator"] == 1,
    }
    data = SklearnCRRTDataModule.from_argparse_args(df, args, filters=filters)

    # Pass the original datasets split pt_ids if doing rolling window analysis
    if args.slide_window:
        with open(SPLIT_IDS_PATH, "rb") as f:
            reference_ids = pickle.load(f)
            reference_ids = {
                split: pd.Index(split_ids) for split, split_ids in reference_ids.items()
            }  # enforce Index
        with open(COLUMNS_PATH, "rb") as f:
            reference_cols = pickle.load(f)
    else:
        reference_ids = None
        reference_cols = None
    data.setup(reference_ids=reference_ids, reference_cols_mask=reference_cols)

    # Then need to update the predictor
    model = CRRTStaticPredictor.from_argparse_args(args)
    if args.stage == "eval":
        # Load up trained portion and hparams
        if args.slide_window:  # Override with already trained model
            model.load_model(MODEL_DIR)
        else:
            model.load_model(args.best_model_path)
            # if tuning the best model will never be dumped, so we dump it on the evaluation of the best model on original reference window
            if args.tune_n_trials and args.slide_window == 0:
                dump_artifacts_for_rolling_windows(data, model)
        return model.evaluate(data, "test")
    else:  # Training / tuning
        model.fit(data)

        # only want to serialize artifacts on training if we're not tuning and we're training the model on the original reference window (not slided yet)
        # functionally slide_window = 0 == None, but 0 indicates there will be sliding in the future
        if not args.tune_n_trials and args.slide_window == 0:
            dump_artifacts_for_rolling_windows(data, model)

        # Don't want to run extensive evaluation, just basic metrics for tuning/training
        model.static_model.hparams["curve_names"] = None
        model.static_model.hparams["error_analysis"] = None
        model.static_model.hparams["top_k_feature_importance"] = None
        return model.evaluate(data, "val")
        # model.evaluate(data, "train")
