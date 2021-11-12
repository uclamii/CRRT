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
from data.longitudinal_utils import UNIVERSAL_TIME_COL_NAME, get_time_window_mask
from data.utils import (
    loading_message,
    onehot,
    read_files_and_combine,
)


def get_num_prev_crrt_treatments(df: pd.DataFrame):
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
    treatments_per_pt.index.rename("Num Prev CRRT Treatments", level=-1, inplace=True)
    # separate it out and make multindex: [pt, start date]
    num_prev_crrt_treatments = treatments_per_pt.reset_index(level=-1).set_index(
        "Start Date", append=True
    )  # add start date as second index
    return num_prev_crrt_treatments


def load_outcomes(
    raw_data_dir: str,
    group_by: List[str],
    outcome_file: str = "CRRT Deidentified 2017-2019.csv",
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

    # Drop missing pt ids
    outcomes_df = outcomes_df.dropna(subset=["IP_PATIENT_ID"])

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

    # patients can have multiple treatments but each (pt, treatment) is 1 sample
    # we dont want to lose info of previous treatments, so we add as feature
    num_prev_crrt_treatments = get_num_prev_crrt_treatments(outcomes_df)

    # join with num prev crrt treatments (set index to [pt, start date])
    return outcomes_df.merge(
        num_prev_crrt_treatments, how="inner", on=group_by
    ).set_index(group_by)


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


def merge_longitudinal_with_static_feaures(
    longitudinal_features: pd.DataFrame,
    static_features: pd.DataFrame,
    how: str = "outer",
) -> pd.DataFrame:
    """
    Outer join: patients with no longitudinal data will stil be included.
    Merge would mess it up since static doesn't have UNIVERSAL_TIME_COL_NAME, join will broadcast.
    """
    return longitudinal_features.join(static_features, how=how)


def merge_features_with_outcome(
    raw_data_dir: str,
    time_interval: Optional[str] = None,
    pre_start_delta: Optional[Dict[str, int]] = None,
    post_start_delta: Optional[Dict[str, int]] = None,
    time_window_end: str = "Start Date",
) -> pd.DataFrame:
    """
    Loads outcomes and features and then merges them.
    Keeps patients even if they're missing from a data table (Feature).
    Will drop patients who are missing outcomes.
    """

    merge_on = ["IP_PATIENT_ID", "Start Date"]
    outcomes_df = load_outcomes(raw_data_dir, group_by=merge_on)
    time_window = get_time_window_mask(
        outcomes_df, pre_start_delta, post_start_delta, time_window_end
    )

    # this needs to come after load outcomes
    if time_interval:
        merge_on.append(UNIVERSAL_TIME_COL_NAME)

    longitudinal_dfs = [
        load_diagnoses(
            raw_data_dir, time_interval=time_interval, time_window=time_window,
        ),
        load_vitals(
            raw_data_dir, time_interval=time_interval, time_window=time_window,
        ),
        load_medications(
            raw_data_dir, time_interval=time_interval, time_window_end=time_window_end,
        ),
        load_labs(raw_data_dir, time_interval=time_interval, time_window=time_window,),
        load_problems(
            raw_data_dir, time_interval=time_interval, time_window=time_window,
        ),
        load_procedures(
            raw_data_dir, time_interval=time_interval, time_window=time_window,
        ),
    ]

    # outer join features with each other so patients who might not have allergies,  for example, are still included
    features = reduce(
        lambda df1, df2: pd.merge(df1, df2, on=merge_on, how="outer"), longitudinal_dfs
    )
    # NOTE: this will be serialized separately instead
    # features = merge_longitudinal_with_static_feaures(features, load_static_features(raw_data_dir), how="outer")

    # inner join features with outcomes (only patients with outcomes)
    # merge is incorrect here for the same reason as static
    features_with_outcomes = features.join(outcomes_df, how="inner")

    # patients with multiple treatments will be separated, enforce they're grouped
    # df = features_with_outcomes.groupby(merge_on).last()

    return features_with_outcomes
