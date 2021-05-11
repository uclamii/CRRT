import pandas as pd

from data.longitudinal_utils import (
    AGGREGATE_FUNCTIONS,
    hcuppy_map_code,
    time_window_mask,
)
from data.utils import loading_message, read_files_and_combine

from hcuppy.ccs import CCSEngine
from hcuppy.cpt import CPT


def load_diagnoses(
    outcomes_df: pd.DataFrame,
    dx_file: str = "Encounter_Diagnoses_19-000093_10082020.txt",
) -> pd.DataFrame:
    loading_message("Diagnoses")
    dx_df = read_files_and_combine([dx_file])

    # convert icd10 to ccs to reduce number of categories for diagnoses.
    ce = CCSEngine(mode="dx")
    # ICD9 is 2013-15. Outcomes are 2018-19, so we will ignore ICD9 codes for now.
    # NOTE: This needs to change if our time window gets very large and we extend into 2013-15.
    icd10_mask = dx_df["ICD_TYPE"] == 10
    dx_df = hcuppy_map_code(
        dx_df[icd10_mask],
        code_col="ICD_CODE",
        exploded_cols=[
            "CCS_CODE",
            "CCS_DESCRIPTION",
            "CCS_LEVEL1",
            "CCS_LEVEL1_DESCRIPTION",
            "CCS_LEVEL2",
            "CCS_LEVEL2_DESCRIPTION",
        ],
        hcuppy_converter_function=ce.get_ccs,
    )

    dx_feature = aggregate_cat_feature(
        outcomes_df, dx_df, time_col="DIAGNOSIS_DATE", agg_on="CCS_CODE"
    )
    return dx_feature


def load_vitals(
    outcomes_df: pd.DataFrame,
    vitals_file: str = "Flowsheet_Vitals_19-000093_10082020.txt",
) -> pd.DataFrame:
    loading_message("Vitals")
    vitals_df = read_files_and_combine(["Flowsheet_Vitals_19-000093_10082020.txt"])
    vitals_df = split_sbp_and_dbp(vitals_df)

    # drop duplicates for the same patient for the same vital (taken at same time indicates duplicate)
    old_size = vitals_df.shape[0]
    vitals_df = vitals_df.drop_duplicates(
        subset=["IP_PATIENT_ID", "VITAL_SIGN_TYPE", "VITAL_SIGN_TAKEN_TIME"]
    )
    f"Dropped {old_size - vitals_df.shape[0]} rows that were duplicates."

    # these vitals are not float point numbers, we want to ignore them and then convert the vitals to float to aggregate
    ignore_vitals = ["O2 Device"]
    ignore_mask = ~vitals_df["VITAL_SIGN_TYPE"].isin(ignore_vitals)
    vitals_df = vitals_df[ignore_mask]

    # convert to float
    vitals_df["VITAL_SIGN_VALUE"] = vitals_df["VITAL_SIGN_VALUE"].astype(float)

    vitals_feature = aggregate_ctn_feature(
        outcomes_df,
        vitals_df,
        time_col="VITAL_SIGN_TAKEN_TIME",
        agg_on="VITAL_SIGN_TYPE",
        agg_values_col="VITAL_SIGN_VALUE",
    )

    return vitals_feature


def split_sbp_and_dbp(vitals_df: pd.DataFrame) -> pd.DataFrame:
    # Split BP into SBP and DBP
    vitals_df["VITAL_SIGN_TYPE"].replace({"BP": "SBP/DBP"}, inplace=True)
    explode_cols = ["VITAL_SIGN_VALUE", "VITAL_SIGN_TYPE"]

    def try_split_col(col: pd.Series):
        # Split col with "/" in it (only BP values and name) from explode_cols
        try:
            return col.str.split("/").explode()
        except:
            return col

    # Ref: https://stackoverflow.com/a/57122617/1888794
    # don't explode the columsn you set index to, explode the rest via apply, reset everything to normal
    vitals_df = (
        vitals_df.set_index(list(vitals_df.columns.difference(explode_cols)))
        .apply(try_split_col)
        .reset_index()
        .reindex(vitals_df.columns, axis=1)
    )
    return vitals_df


# TODO: get epic drug categories, or figure out how to reduce dimensionality
def load_medications(
    outcomes_df: pd.DataFrame, rx_file: str = "Medications_19-000093_10082020.txt",
) -> pd.DataFrame:
    loading_message("Medications")
    rx_df = read_files_and_combine([rx_file])

    return rx_df


def load_labs(
    outcomes_df: pd.DataFrame, labs_file: str = "Labs_19-000093_10082020.txt"
) -> pd.DataFrame:
    loading_message("Labs")
    labs_df = read_files_and_combine([labs_file])
    # Force numeric, ignore strings
    labs_df["RESULTS"] = pd.to_numeric(labs_df["RESULTS"], errors="coerce")

    labs_feature = aggregate_ctn_feature(
        outcomes_df,
        labs_df,
        time_col="ORDER_TIME",
        agg_on="DESCRIPTION",
        agg_values_col="RESULTS",
    )

    return labs_feature


def load_problems(
    outcomes_df: pd.DataFrame,
    problems_file: str = "Problem_Lists_19-000093_10082020.txt",
    problems_dx_file: str = "problem_list_diagnoses_19-000093_10082020.txt",
) -> pd.DataFrame:
    loading_message("Problems")
    problems_df = read_files_and_combine([problems_dx_file, problems_file])
    problems_df.columns = [col.upper() for col in problems_df.columns]

    # convert icd10 to ccs only to active problems
    ce = CCSEngine(mode="dx")
    active_and_icd10_mask = (problems_df["PROBLEM_STATUS"] == "Active") & (
        problems_df["ICD_TYPE"] == 10
    )
    problems_df = hcuppy_map_code(
        problems_df[active_and_icd10_mask],
        code_col="ICD_CODE",
        exploded_cols=[
            "CCS_CODE",
            "CCS_DESCRIPTION",
            "CCS_LEVEL1",
            "CCS_LEVEL1_DESCRIPTION",
            "CCS_LEVEL2",
            "CCS_LEVEL2_DESCRIPTION",
        ],
        hcuppy_converter_function=ce.get_ccs,
    )

    problems_feature = aggregate_cat_feature(
        outcomes_df, problems_df, time_col="NOTED_DATE", agg_on="CCS_CODE"
    )

    return problems_feature


def load_procedures(
    outcomes_df: pd.DataFrame,
    procedures_file: str = "Procedures_19-000093_10082020.txt",
) -> pd.DataFrame:
    loading_message("Procedures")
    procedures_df = read_files_and_combine([procedures_file])

    # Convert CPT codes to their sections (less granular)
    cpt = CPT()
    procedures_df = hcuppy_map_code(
        procedures_df,
        code_col="PROC_CODE",
        exploded_cols=["CPT_SECTION", "SECTION_DESCRIPTION"],
        hcuppy_converter_function=cpt.get_cpt_section,
    )

    procedures_feature = aggregate_cat_feature(
        outcomes_df, procedures_df, time_col="PROC_DATE", agg_on="CPT_SECTION"
    )

    return procedures_feature


def aggregate_cat_feature(
    outcomes_df: pd.DataFrame, cat_df: pd.DataFrame, time_col: str, agg_on: str
) -> pd.DataFrame:
    # mask for time
    cat_df = time_window_mask(outcomes_df, cat_df, time_col)

    # Get dummies for the categorical column
    cat_feature = pd.get_dummies(cat_df[["IP_PATIENT_ID", agg_on]], columns=[agg_on])
    # Sum across a patient (within a time window)
    cat_feature = cat_feature.groupby("IP_PATIENT_ID").apply(lambda df: df.sum(axis=0))

    # fix indices ruined by groupby
    cat_feature = cat_feature.drop("IP_PATIENT_ID", axis=1).reset_index()

    return cat_feature


def aggregate_ctn_feature(
    outcomes_df: pd.DataFrame,
    ctn_df: pd.DataFrame,
    time_col: str,
    agg_on: str,
    agg_values_col: str,
) -> pd.DataFrame:
    """Aggregate a continuous longitudinal feature (e.g., vitals, labs).
    Filter time window based on a column name provided.
    Aggregate on a column name provided:
        need a column for the name to group by, and the corresponding value column name.
    """
    # filter to window
    ctn_df = time_window_mask(outcomes_df, ctn_df, time_col)

    # Apply aggregate functions (within time window)
    ctn_feature = ctn_df.groupby(["IP_PATIENT_ID", agg_on]).agg(
        {agg_values_col: AGGREGATE_FUNCTIONS}
    )

    # Flatten aggregations from multi_index into a feature vector
    ctn_feature = ctn_feature.unstack()
    ctn_feature.columns = ctn_feature.columns.map("_".join)
    ctn_feature.reset_index(inplace=True)

    return ctn_feature

