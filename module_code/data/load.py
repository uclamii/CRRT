from functools import reduce
from os.path import join
import pandas as pd
from typing import Dict, List, Optional
from datetime import timedelta

from data.longitudinal_features import (
    load_diagnoses,
    load_vitals,
    load_labs,
    load_medications,
    load_problems,
    load_procedures,
)
from data.longitudinal_utils import TIME_BEFORE_START_DATE, UNIVERSAL_TIME_COL_NAME
from data.utils import (
    loading_message,
    onehot,
    read_files_and_combine,
)


def load_outcomes(
    raw_data_dir: str, outcome_file: str = "CRRT Deidentified 2017-2019.csv",
) -> pd.DataFrame:
    """
    Load outcomes from outcomes file.
    Given higher granularity, so we include a binary outcome (recommend crrt).
    Filters: no pediatric patient outcomes, patients with malformed outcome (i.e. 0 or >1 of the 4 outcomes indicated).
    """

    loading_message("Outcomes")
    outcomes_df = pd.read_csv(join(raw_data_dir, outcome_file))

    positive_outcomes = ["Recov. renal funct.", "Transitioned to HD"]
    negative_outcomes = ["Palliative Care", "Expired "]
    outcome_cols = positive_outcomes + negative_outcomes

    #### Filtering ####
    # Exclude pediatric data
    exclude_peds_mask = (
        outcomes_df["Hospital name"] != "UCLA MEDICAL CENTER- PEDIATRICS"
    )
    # Each row should have exactly 1 1.0 value (one-hot of the 4 cols)
    exactly_one_outcome_mask = outcomes_df[outcome_cols].fillna(0).sum(axis=1) == 1

    # TODO: Should i drop the bad row?
    outcomes_df = outcomes_df[exclude_peds_mask & exactly_one_outcome_mask]

    #### Construct Binary Outcome ####
    # Recommend CRRT if they had a positive outcome.
    recommend_crrt = (outcomes_df[positive_outcomes] == 1).any(axis=1)
    outcomes_df["recommend_crrt"] = recommend_crrt.astype(int)

    #### Construct Start Date ####  -- For convenience of time-windows --
    # Enforce date column to datetime object
    outcomes_df["End Date"] = pd.to_datetime(outcomes_df["End Date"])

    # CRRT Start Date = End Date - (Days on CRRT - 1)
    # e.g. finish on the 10th and 3 days of CRRT: 8th (1), 9th (2), 10th (3)
    offset = outcomes_df["CRRT Total Days"].map(lambda days: timedelta(days=days - 1))
    outcomes_df["Start Date"] = outcomes_df["End Date"] - offset

    return outcomes_df


def load_static_features(
    raw_data_dir: str,
    static_features: List[str] = (
        "Allergies_19-000093_10082020.txt",
        "Patient_Demographics_19-000093_10082020.txt",
        "Social_History_19-000093_10082020.txt",
    ),
) -> pd.DataFrame:
    loading_message("Static Features")
    """Returns static features dataframe. 1 row per patient."""
    # include all patients from all tables, so outer join
    static_df = read_files_and_combine(static_features, raw_data_dir, how="outer")
    static_df = map_provider_id_to_type(static_df, raw_data_dir)

    # TODO: only do this if file doesn't exist
    # save description of allergen code as a df mapping
    allergen_code_to_description_mapping = static_df[
        ["ALLERGEN_ID", "DESCRIPTION"]
    ].set_index("ALLERGEN_ID")
    allergen_code_to_description_mapping.to_csv(
        join(raw_data_dir, "allergen_code_mapping.csv")
    )
    # drop allergen description since we won't be using it
    static_df.drop("DESCRIPTION", axis=1)

    # only onehot encode multicategorical columns (not binary)
    # all binary vars are encoded 0/1 (no/yes)
    cols_to_onehot = [
        "ALLERGEN_ID",
        # TODO: we should pick only one: race or ethnicity, but not both. maybe do this as a precursor step to prediction but not as the preprocess pipeline
        "RACE",
        "ETHNICITY",
        "PCP_PROVIDER_TYPE",
        "TOBACCO_USER",
        "SMOKING_TOB_STATUS",
    ]
    # will aggregate if there's more than one entry per pateint.
    # this should only affect allergens, the other entries should not be affected
    static_df = onehot(static_df, cols_to_onehot, sum_across_patient=True)

    return static_df


def map_provider_id_to_type(
    static_df: pd.DataFrame,
    raw_data_dir: str,
    provider_mapping_file: str = "providers_19-000093_10082020.txt",
) -> pd.DataFrame:
    """There are a bunch of IDs but they mostly all map to the same type, so here we'll use the string name instead of code."""
    provider_mapping = pd.read_csv(join(raw_data_dir, provider_mapping_file))
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
    time_interval: Optional[str] = None,
    time_before_start_date: Dict[str, int] = TIME_BEFORE_START_DATE,
    time_window_end: str = "Start Date",
) -> pd.DataFrame:
    """
    Loads outcomes and features and then merges them.
    Keeps patients even if they're missing from a data table (Feature).
    Will drop patients who are missing outcomes.
    """

    outcomes_df = load_outcomes(raw_data_dir)

    static_df = load_static_features(raw_data_dir)
    longitudinal_dfs = [
        load_diagnoses(
            outcomes_df,
            raw_data_dir,
            time_interval=time_interval,
            time_before_start_date=time_before_start_date,
            time_window_end=time_window_end,
        ),
        load_vitals(
            outcomes_df,
            raw_data_dir,
            time_interval=time_interval,
            time_before_start_date=time_before_start_date,
            time_window_end=time_window_end,
        ),
        load_medications(
            outcomes_df,
            raw_data_dir,
            time_interval=time_interval,
            time_before_start_date=time_before_start_date,
            time_window_end=time_window_end,
        ),
        load_labs(
            outcomes_df,
            raw_data_dir,
            time_interval=time_interval,
            time_before_start_date=time_before_start_date,
            time_window_end=time_window_end,
        ),
        load_problems(
            outcomes_df,
            raw_data_dir,
            time_interval=time_interval,
            time_before_start_date=time_before_start_date,
            time_window_end=time_window_end,
        ),
        load_procedures(
            outcomes_df,
            raw_data_dir,
            time_interval=time_interval,
            time_before_start_date=time_before_start_date,
            time_window_end=time_window_end,
        ),
    ]

    merge_on = (
        ["IP_PATIENT_ID", UNIVERSAL_TIME_COL_NAME] if time_interval else "IP_PATIENT_ID"
    )

    # outer join features with each other so patients who might not have allergies,  for example, are still included
    features = reduce(
        lambda df1, df2: pd.merge(df1, df2, on=merge_on, how="outer"), longitudinal_dfs
    )
    # join with static data, (outer join for same reason)
    # merge would mess it up since static doesn't have UNIVERSAL_TIME_COL_NAME, join will broadcast
    features = features.join(static_df, how="outer")

    # some de-identified IDs missing (NaN), some people with multiple outcomes
    # TODO: decide what to do with multiple outcomes
    working_outcomes_df = (
        outcomes_df.dropna(subset=["IP_PATIENT_ID"])
        .drop_duplicates(subset=["IP_PATIENT_ID"], keep="first")
        .set_index("IP_PATIENT_ID")
    )
    # inner join features with outcomes (only patients with outcomes)
    # merge is incorrect here for the same reason as static
    features_with_outcomes = features.join(working_outcomes_df, how="inner")

    return features_with_outcomes
