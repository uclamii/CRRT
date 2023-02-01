from argparse import Namespace
from functools import reduce
import logging
from os.path import join
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, Optional
from pandas import DataFrame, DatetimeIndex, read_excel, read_csv, merge

# for serialization on the fly
import pandas as pd

serialize_pkg = pd

from data.longitudinal_features import (
    load_diagnoses,
    load_vitals,
    load_labs,
    load_medications,
    load_problems,
    load_procedures,
)
from data.longitudinal_utils import UNIVERSAL_TIME_COL_NAME, get_time_window_mask
from data.utils import (
    loading_message,
    onehot,
    read_files_and_combine,
    get_preprocessed_file_name,
)
from data.preprocess import adhoc_preprocess_data


def get_num_prev_crrt_treatments(df: DataFrame):
    """
    Works on any df as as long as it has pt id and start date.
    Returns number of prev crrt treatments per [pt, start date].
    Good to join with dfs with the same/similar index.
    """
    # get unique start dates per pt (Series[List[Date]])
    # multiindex of pt and then num prev crrt treatments (from reset index)
    treatments_per_pt = df.groupby(["IP_PATIENT_ID"]).apply(
        lambda df: df["Start Date"].reset_index(drop=True)
    )
    # rename the outermost index which should be num prev treatments
    # Changed in pandas v1.3.0
    treatments_per_pt.index.set_names(
        "Num Prev CRRT Treatments", level=-1, inplace=True
    )
    # separate it out and make multindex: [pt, start date]
    num_prev_crrt_treatments = treatments_per_pt.reset_index(level=-1).set_index(
        "Start Date", append=True
    )  # add start date as second index
    return num_prev_crrt_treatments


def load_outcomes(
    raw_data_dir: str,
    group_by: List[str],
    outcome_file: str = "CRRT Deidentified 2015-2021YTD_VF.xlsx",
) -> DataFrame:
    """
    Load outcomes from outcomes file.
    Given higher granularity, so we include a binary outcome (recommend crrt).
    Filters: no pediatric patient outcomes, patients with malformed outcome (i.e. 0 or >1 of the 4 outcomes indicated).
    """

    loading_message("Outcomes")
    outcomes_df = read_excel(
        join(raw_data_dir, outcome_file), sheet_name="2015-2021 YTD"
    )

    positive_outcomes = ["Recov. renal funct.", "Transitioned to HD"]
    negative_outcomes = ["Comfort Care", "Expired "]
    outcome_cols = positive_outcomes + negative_outcomes

    #### Filtering ####

    # For patients with duplicated start dates, take the sample with greatest end date
    outcomes_df = outcomes_df.sort_values(
        by=["IP_PATIENT_ID", "End Date"]
    ).drop_duplicates(subset=["IP_PATIENT_ID", "Start Date"], keep="last")

    # Each row should have exactly 1 1.0 value (one-hot of the 4 cols)
    exactly_one_outcome_mask = outcomes_df[outcome_cols].fillna(0).sum(axis=1) == 1

    outcomes_df = outcomes_df[exactly_one_outcome_mask]

    # Drop missing pt ids
    outcomes_df = outcomes_df.dropna(subset=["IP_PATIENT_ID"])

    # Get rid of Age feature that I constructed since controls don't have outcomes file
    outcomes_df = outcomes_df.drop("Age", axis=1, errors="ignore")

    #### Construct Binary Outcome ####
    # Recommend CRRT if they had a positive outcome.
    recommend_crrt = (outcomes_df[positive_outcomes] == 1).any(axis=1)
    outcomes_df["recommend_crrt"] = recommend_crrt.astype(int)

    #### Construct other features ####
    outcomes_df["CRRT Year"] = DatetimeIndex(outcomes_df["End Date"]).year

    #### Contruct Num Previous Treatments ####
    # patients can have multiple treatments but each (pt, treatment) is 1 sample
    # we dont want to lose info of previous treatments, so we add as feature
    num_prev_crrt_treatments = get_num_prev_crrt_treatments(outcomes_df)

    # join with num prev crrt treatments (set index to [pt, start date])
    return outcomes_df.merge(
        num_prev_crrt_treatments, how="inner", on=group_by
    ).set_index(group_by)


def construct_outcomes(procedures_df: DataFrame, merge_on: List[str]) -> DataFrame:
    """
    For control cohort that doesn't have outcomes.
    Instead of CRRT start date we use a basic heuristic for a "anchoring" date / pseudo start date for time windows
    # In controls this is 4725 patients
    """
    outcomes = (
        procedures_df.groupby("IP_PATIENT_ID")
        # We randomly select the date of a procedure for any patient
        .sample(n=1)[["IP_PATIENT_ID", "PROC_DATE"]]
        # Ensure date is a datetime object
        .astype({"PROC_DATE": "datetime64[ns]"}).assign(
            **{
                # they should not go on crrt and did not
                "recommend_crrt": 0,
                # because they never went on crrt theres 0 days and no prev treatments
                "CRRT Total Days": 0,
                "Num Prev CRRT Treatments": 0,
                # TODO: does it make sense to have this anymore?
                # crrt_year=lambda row: DatetimeIndex(row["PROCEDURE_DATE"]).year,
                "CRRT Year": lambda df: df["PROC_DATE"].map(lambda dt: dt.year),
                # need this for get_time_window_mask, will not be used
                "End Date": lambda df: df["PROC_DATE"],
            }
        )
        # Align column names to a real outcomes file
        .rename(columns={"PROC_DATE": "Start Date"})
    )

    return outcomes.set_index(merge_on)


def load_static_features(
    raw_data_dir: str,
    static_features: List[str] = [
        # "Allergies.txt",  # Very sparse for CRRT patients.
        "Patient_Demographics.txt",
        # "Social_History.txt",
    ],
) -> DataFrame:
    loading_message("Static Features")
    """
    Returns static features dataframe. 1 row per patient.
    only onehot encode multicategorical columns (not binary)
    all binary vars are encoded 0/1 (no/yes)
    """
    # include all patients from all tables, so outer join
    static_df = read_files_and_combine(static_features, raw_data_dir, how="outer")

    cols_to_onehot = []
    if "Allergies.txt" in static_features:
        allergen_code_mapping_fname = join(raw_data_dir, "allergen_code_mapping.csv")
        if not Path(allergen_code_mapping_fname).exists():
            # save description of allergen code as a df mapping
            allergen_code_to_description_mapping = static_df[
                ["ALLERGEN_ID", "DESCRIPTION"]
            ].set_index("ALLERGEN_ID")
            allergen_code_to_description_mapping.to_csv(allergen_code_mapping_fname)
            # drop allergen description since we won't be using it
        static_df.drop("DESCRIPTION", axis=1)

        cols_to_onehot.append("ALLERGEN_ID")

    if "Patient_Demographics.txt" in static_features:
        static_df = static_df.rename(
            {
                "GENDER": "SEX",
                "IP_CURRENT_PCP_ID": "PCP_IP_PROVIDER_ID",
                "VITAL_STATUS": "KNOWN_DECEASED",
            },
            axis=1,
        ).replace({"ETHNICITY": {"Choose Not to Answer": "Patient Refused"}})
        static_df = map_provider_id_to_type(static_df, raw_data_dir)

        # explicitly mapping here instead of numerical encoding automatically so that you know which is which when referencing outputs/data/etc.
        bin_cols_mapping = {
            "SEX": {"Male": 0, "Female": 1},
            "KNOWN_DECEASED": {"Not Known Deceased": 0, "Known Deceased": 1},
        }
        static_df = static_df.replace(bin_cols_mapping)

        cols_to_onehot += [
            "RACE",  # white, other, multiple races, etc.
            "ETHNICITY",  # yes/no hispanic/latino
            "PCP_PROVIDER_TYPE",
        ]

    if "Social_History.txt" in static_features:
        cols_to_onehot += ["TOBACCO_USER", "SMOKING_TOB_STATUS"]

    # will aggregate if there's more than one entry per pateint.
    # this should only affect allergens, the other entries should not be affected
    static_df = onehot(static_df, cols_to_onehot, sum_across_patient=False)

    return static_df


def map_provider_id_to_type(
    static_df: DataFrame,
    raw_data_dir: str,
    provider_mapping_file: str = "Providers.txt",
) -> DataFrame:
    """There are a bunch of IDs but they mostly all map to the same type, so here we'll use the string name instead of code."""
    provider_mapping = read_files_and_combine([provider_mapping_file], raw_data_dir)
    provider_mapping = dict(
        zip(provider_mapping["IP_PROVIDER_ID"], provider_mapping["PROVIDER_TYPE"])
    )
    static_df["PCP_IP_PROVIDER_ID"] = static_df["PCP_IP_PROVIDER_ID"].map(
        provider_mapping
    )
    static_df.rename(columns={"PCP_IP_PROVIDER_ID": "PCP_PROVIDER_TYPE"}, inplace=True)
    return static_df


def merge_features_with_outcome(
    raw_data_dir: str,
    outcomes_df: DataFrame,
    merge_on: List[str] = ["IP_PATIENT_ID", "Start Date"],
    time_interval: Optional[str] = None,
    pre_start_delta: Optional[Dict[str, int]] = None,
    post_start_delta: Optional[Dict[str, int]] = None,
    time_window_end: str = "Start Date",
    slide_window_by: int = 0,
    preloaded_dfs: Dict[str, DataFrame] = {},
) -> DataFrame:
    """
    Loads outcomes and features and then merges them.
    Keeps patients even if they're missing from a data table (Feature).
    Will drop patients who are missing outcomes.
    """
    time_window = get_time_window_mask(
        outcomes_df, pre_start_delta, post_start_delta, time_window_end, slide_window_by
    )

    # this needs to come after load outcomes
    if time_interval:
        merge_on.append(UNIVERSAL_TIME_COL_NAME)

    longitudinal_dfs = [
        preloaded_dfs.get(
            "dx",
            load_diagnoses(
                raw_data_dir,
                time_interval=time_interval,
                time_window=time_window,
            ),
        ),
        preloaded_dfs.get(
            "vitals",
            load_vitals(
                raw_data_dir, time_interval=time_interval, time_window=time_window
            ),
        ),
        preloaded_dfs.get(
            "rx",
            load_medications(
                raw_data_dir, time_interval=time_interval, time_window=time_window
            ),
        ),
        preloaded_dfs.get(
            "labs",
            load_labs(
                raw_data_dir, time_interval=time_interval, time_window=time_window
            ),
        ),
        preloaded_dfs.get(
            "pr",
            load_problems(
                raw_data_dir, time_interval=time_interval, time_window=time_window
            ),
        ),
        preloaded_dfs.get(
            "cpt",
            load_procedures(
                raw_data_dir, time_interval=time_interval, time_window=time_window
            ),
        ),
    ]

    # outer join features with each other so patients who might not have allergies,  for example, are still included
    features = reduce(
        lambda df1, df2: merge(df1, df2, on=merge_on, how="outer"), longitudinal_dfs
    )

    # inner join features with outcomes (only patients with outcomes)
    # merge is incorrect here for the same reason as static
    features_with_outcomes = features.join(outcomes_df, how="inner")

    # patients with multiple treatments will be separated, enforce they're grouped
    # df = features_with_outcomes.groupby(merge_on).last()

    return features_with_outcomes


def process_and_serialize_raw_data(
    args: Namespace, preprocessed_df_path: str, cohort: str
) -> DataFrame:
    logging.info(f"Creating preprocessed file {preprocessed_df_path}!")
    merge_on = ["IP_PATIENT_ID", "Start Date"]
    start = timer()
    # Dynamically get the correct function in this module based on the cohort
    df = globals().get(f"preproc_{cohort}")(args, merge_on)
    logging.info(f"Loading took {timer() - start} seconds.")
    serialize_fn = getattr(df, f"to_{args.serialization}")
    serialize_fn(preprocessed_df_path)
    return df


def get_preprocessed_df_path(args: Namespace, cohort: str) -> str:
    preprocessed_df_fname = get_preprocessed_file_name(
        args.pre_start_delta,
        args.post_start_delta,
        args.time_interval,
        args.time_window_end,
        args.slide_window_by,
        args.preprocessed_df_file,
        args.serialization,
    )
    return join(getattr(args, f"{cohort}_data_dir"), preprocessed_df_fname)


def load_data(args: Namespace, cohort: str) -> DataFrame:
    preprocessed_df_path = get_preprocessed_df_path(args, cohort)
    deserialize_fn = getattr(serialize_pkg, f"read_{args.serialization}")
    try:  # raise IOError
        df = deserialize_fn(preprocessed_df_path)
    except IOError:
        df = process_and_serialize_raw_data(args, preprocessed_df_path, cohort)

    if args.model_type == "static":
        cohort_data_dir = getattr(args, f"{cohort}_data_dir")
        path = join(cohort_data_dir, f"static_data.{args.serialization}")
        try:
            static_features = deserialize_fn(path)
        except IOError:
            static_features = load_static_features(cohort_data_dir).set_index(
                "IP_PATIENT_ID"
            )
            serialize_fn = getattr(static_features, f"to_{args.serialization}")
            serialize_fn(path)

        # Merge would mess it up since static doesn't have UNIVERSAL_TIME_COL_NAME, join will broadcast.
        df = df.join(static_features, how="inner")

    return adhoc_preprocess_data(df, args)


"""
Functions for each cohort/dataset for preprocessing the raw datasets.
Each much be "preproc_" followed by the cohort name.
Cohort names can only have underscores so they're valid var names.
"""


def preproc_ucla_crrt(args: Namespace, merge_on: List[str]) -> DataFrame:
    outcomes_df = load_outcomes(args.ucla_crrt_data_dir, group_by=merge_on)
    return merge_features_with_outcome(
        args.ucla_crrt_data_dir,
        outcomes_df,
        merge_on,
        args.time_interval,
        args.pre_start_delta,
        args.post_start_delta,
        args.time_window_end,
        args.slide_window_by,
    )  # 140s ~2.5 mins, 376.5s ~6mins for daily aggregation


def preproc_ucla_control(args: Namespace, merge_on: List[str]) -> DataFrame:
    procedures_df = load_procedures(args.ucla_control_data_dir, aggregate=False)
    outcomes_df = construct_outcomes(procedures_df, merge_on)
    controls_window_size = args.pre_start_delta
    """
    TODO[High]: this actually can match ucla_crrt by just using everythign from args, but we have to update construct_outcomes to also have "End Date"
        This will be important for the dynamic model
    """
    return merge_features_with_outcome(
        args.ucla_control_data_dir,
        outcomes_df,
        merge_on,
        args.time_interval,
        controls_window_size,
        None,
        "Start Date",
        args.slide_window_by,
    )
