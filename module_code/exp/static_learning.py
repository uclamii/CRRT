from argparse import Namespace
from functools import reduce
import pickle
from typing import Any, Callable, List, Tuple, Union, Dict
import cloudpickle  # Allows serializing functions
import pandas as pd
from os.path import join

from data.sklearn_loaders import SklearnCRRTDataModule
from models.static_models import CRRTStaticPredictor

SPLIT_IDS_PATH = join("local_data", "split_ids.pkl")
COLUMNS_PATH = join("local_data", "columns.pkl")
DATA_TRANSFORM_PATH = join("local_data", "data_transform.pkl")
MODEL_DIR = join("local_data", "static_model")


def dump_artifacts_for_rolling_windows(
    data: SklearnCRRTDataModule, model: CRRTStaticPredictor
):
    # this will dump the model D -> D+i, and D+1 retrained and then evaluated
    model.save_model(MODEL_DIR)  # will ensure path exists
    with open(SPLIT_IDS_PATH, "wb") as f:
        pickle.dump(data.split_pt_ids, f)
    with open(COLUMNS_PATH, "wb") as f:  # dump  all columns to align
        pickle.dump(data.columns, f)
    with open(DATA_TRANSFORM_PATH, "wb") as f:
        # Ref: https://github.com/scikit-learn/scikit-learn/issues/17390
        cloudpickle.dump(data.data_transform, f)


def combine_filters(
    f1_args: Tuple[Union[str, List[str]], Union[Any, List[Any]]],
    f2_args: Tuple[Union[str, List[str]], Union[Any, List[Any]]],
) -> Tuple[Union[str, List[str]], Union[Any, List[Any]]]:
    f1_cols, f1_vals = f1_args
    f2_cols, f2_vals = f2_args

    def listify(item: Union[Any, List[Any]]) -> List[Any]:
        return item if isinstance(item, list) else [item]

    return (listify(f1_cols) + listify(f2_cols), listify(f1_vals) + listify(f2_vals))


def generate_filters() -> Dict[str, Tuple[str, int]]:
    # these filters will be passed to `SklearnCRRTDataModule.get_filter`
    disease_filter_args = {  # filter based on disease status
        "heart": ("heart_pt_indicator", 1),
        "liver": ("liver_pt_indicator", 1),
        "infection": ("infection_pt_indicator", 1),
        "no_heart_liver_infection": (
            ["infection_pt_indicator", "heart_pt_indicator", "liver_pt_indicator"],
            [0, 0, 0],
        ),
    }
    sex_filter_args = {  # filter based on demographic
        "female": ("SEX", 1),
        "male": ("SEX", 0),
    }
    age_filter_args = {  # age groups every decade until 120 years
        f"age_{range[0]}_to_{range[1]}": ("AGE", range)
        for range in [(n, min(n + 10, 120)) for n in range(0, 120, 10)]
    }
    ethnicity_filter_args = {
        name: ("ETHNICITY", i)
        for i, name in enumerate(["Not Hispanic or Latino", "Hispanic or Latino"])
    }
    race_filter_args = {
        race: (f"RACE_{race}", 1)
        # TODO: pulled from descriptive_report.ipynb, maybe make it auto?
        for race in [
            "American Indian or Alaska Native",
            "Asian",
            "Black or African American",
            "Multiple Races",
            "Native Hawaiian or Other Pacific Islander",
            "Unknown",
            "White or Caucasian",
        ]
    }
    # cartesian_product = {
    #     f"{demo}_{dz}": combine_filters(demo_args, dz_args)
    #     for demo, demo_args in demo_filter_args.items()
    #     for dz, dz_args in disease_filter_args.items()
    # }
    filters = {
        **disease_filter_args,
        **age_filter_args,
        **sex_filter_args,
        **race_filter_args,
        **ethnicity_filter_args,
        # **cartesian_product,
    }
    return filters


def static_learning(args: Namespace):
    # need to update CRRTDataModule
    # TODO: this can be cleaned up /maybe moved and then names passed as flag to select?

    data = SklearnCRRTDataModule.from_argparse_args(args, filters=generate_filters())

    # Pass the original datasets split pt_ids if doing rolling window analysis
    if (not args.tune_n_trials and args.slide_window_by == 0) or args.slide_window_by:
        with open(SPLIT_IDS_PATH, "rb") as f:
            reference_ids = pickle.load(f)
            reference_ids = {
                split: pd.Index(split_ids) for split, split_ids in reference_ids.items()
            }  # enforce Index
        with open(COLUMNS_PATH, "rb") as f:
            original_columns = pickle.load(f)
        with open(DATA_TRANSFORM_PATH, "rb") as f:
            data_transform = cloudpickle.load(f)
    else:
        reference_ids = None
        original_columns = None
        data_transform = None
    data.setup(
        args,
        reference_ids=reference_ids,
        reference_cols=original_columns,
        data_transform=data_transform,
    )

    # Then need to update the predictor
    model = CRRTStaticPredictor.from_argparse_args(args)
    if args.stage == "eval":
        # Load up trained portion and hparams
        if (
            not args.tune_n_trials and args.slide_window_by == 0
        ) or args.slide_window_by:  # Override with already trained model
            model.load_model(MODEL_DIR)
        else:  # executed at the end of tuning/one-off and if slide_window_by is not 0/None
            model.load_model(args.best_model_path)
            # if tuning the best model will never be dumped, so we dump it on the evaluation of the best model on original reference window
            if args.tune_n_trials and args.slide_window_by == 0:
                dump_artifacts_for_rolling_windows(data, model)
        return model.evaluate(data, "test")
    else:  # Training / tuning
        model.fit(data)

        # only want to serialize artifacts on training if we're not tuning and we're training the model on the original reference window (not slided yet)
        # functionally slide_window_by = 0 == None, but 0 indicates there will be sliding in the future
        if not args.tune_n_trials and args.slide_window_by == 0:
            dump_artifacts_for_rolling_windows(data, model)

        # Don't want to run extensive evaluation, just basic metrics for tuning/training
        model.static_model.hparams["curve_names"] = None
        model.static_model.hparams["error_analysis"] = None
        model.static_model.hparams["top_k_feature_importance"] = None
        # informs tuning, different from testing/eval
        return model.evaluate(data, "val")
        # model.evaluate(data, "train")
