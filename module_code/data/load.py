from functools import reduce
from typing import List
import pandas as pd

from data.longitudinal_features import (
    load_diagnoses,
    load_vitals,
    load_labs,
    load_medications,
    load_problems,
    load_procedures,
)
from data.utils import DATA_DIR, loading_message, read_files_and_combine


def load_outcomes(
    outcome_file: str = "CRRT Deidentified 2017-2019.csv",
) -> pd.DataFrame:
    loading_message("Outcomes")
    outcomes_df = pd.read_csv(f"{DATA_DIR}/{outcome_file}")

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
    # Recommend dialysis if they had a positive outcome.
    recommend_dialysis = (outcomes_df[positive_outcomes] == 1).any(axis=1)
    outcomes_df["recommend_dialysis"] = recommend_dialysis.astype(int)

    return outcomes_df


def load_static_features(
    static_features: List[str] = (
            "Allergies_19-000093_10082020.txt",
            "Patient_Demographics_19-000093_10082020.txt",
            "Social_History_19-000093_10082020.txt",
    ),
    provider_mapping_file: str = "providers_19-000093_10082020.txt",
) -> pd.DataFrame:
    static_df = read_files_and_combine(static_features)

    # map provider id to type
    provider_mapping = pd.read_csv(f"{DATA_DIR}/{provider_mapping_file}")
    provider_mapping = dict(
        zip(provider_mapping["IP_PROVIDER_ID"], provider_mapping["PROVIDER_TYPE"])
    )
    static_df["PCP_IP_PROVIDER_ID"] = static_df["PCP_IP_PROVIDER_ID"].map(
        provider_mapping
    )
    static_df.rename(columns={"PCP_IP_PROVIDER_ID": "PCP_PROVIDER_TYPE"}, inplace=True)

    return static_df


def merge_features_with_outcome() -> pd.DataFrame:
    outcomes_df = load_outcomes()

    static_df = load_static_features()
    longitudinal_dfs = [
        load_diagnoses(outcomes_df=outcomes_df),
        load_vitals(outcomes_df=outcomes_df),
        # load_medications(outcomes_df=outcomes_df), # TODO: still in progress
        load_labs(outcomes_df=outcomes_df),
        load_problems(outcomes_df=outcomes_df),
        load_procedures(outcomes_df=outcomes_df),
    ]
    features_with_outcomes = reduce(
        lambda df1, df2: pd.merge(df1, df2, on="IP_PATIENT_ID", how="inner"),
        longitudinal_dfs + [static_df, outcomes_df],
    )

    return features_with_outcomes
