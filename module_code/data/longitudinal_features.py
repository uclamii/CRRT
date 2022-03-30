import logging
import pandas as pd
from typing import Optional

from data.longitudinal_utils import (
    aggregate_cat_feature,
    aggregate_ctn_feature,
    hcuppy_map_code,
)
from data.utils import loading_message, read_files_and_combine

from hcuppy.ccs import CCSEngine
from hcuppy.cpt import CPT

"""
Prefix a OR b = (a|b) followed by _ and 1+ characters of any char.
{ diagnoses: dx, meds: PHARM_SUBCLASS, problems: pr, procedures: CPT }
"""
CATEGORICAL_COL_REGEX = r"(dx|PHARM_SUBCLASS|pr|CPT|)_.*"
# CONTINUOUS_COL_REGEX = r"(VITAL_SIGN|RESULT)_.*"


def load_diagnoses(
    raw_data_dir: str,
    dx_file: str = "Encounter_Diagnoses.txt",
    time_interval: Optional[str] = None,
    time_window: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    loading_message("Diagnoses")
    dx_df = read_files_and_combine([dx_file], raw_data_dir)

    # convert icd10 to ccs to reduce number of categories for diagnoses.
    ce = CCSEngine(mode="dx")
    # ICD9 is 2013-15. Outcomes are 2018-19, so we will ignore ICD9 codes for now.
    # NOTE: This needs to change if our time window gets very large and we extend into 2013-15.
    icd10_mask = dx_df["ICD_TYPE"] == 10
    dx_df = hcuppy_map_code(
        dx_df[icd10_mask],
        code_col="ICD_CODE",
        exploded_cols=[
            "dx_CCS_CODE",
            "dx_CCS_DESCRIPTION",
            "dx_CCS_LEVEL1",
            "dx_CCS_LEVEL1_DESCRIPTION",
            "dx_CCS_LEVEL2",
            "dx_CCS_LEVEL2_DESCRIPTION",
        ],
        hcuppy_converter_function=ce.get_ccs,
    )
    dx_feature = aggregate_cat_feature(
        dx_df,
        agg_on="dx_CCS_CODE",
        time_col="DIAGNOSIS_DATE",
        time_interval=time_interval,
        time_window=time_window,
    )
    return dx_feature


def load_vitals(
    raw_data_dir: str,
    vitals_file: str = "Flowsheet_Vitals.txt",
    time_interval: Optional[str] = None,
    time_window: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    loading_message("Vitals")
    vitals_df = read_files_and_combine([vitals_file], raw_data_dir)
    vitals_df = split_sbp_and_dbp(vitals_df)

    # drop duplicates for the same patient for the same vital (taken at same time indicates duplicate)
    old_size = vitals_df.shape[0]
    vitals_df = vitals_df.drop_duplicates(
        subset=["IP_PATIENT_ID", "VITAL_SIGN_TYPE", "VITAL_SIGN_TAKEN_TIME"]
    )
    logging.info(f"Dropped {old_size - vitals_df.shape[0]} rows that were duplicates.")

    # these vitals are not float point numbers, we want to ignore them and then convert the vitals to float to aggregate
    ignore_vitals = ["O2 Device"]
    ignore_mask = ~vitals_df["VITAL_SIGN_TYPE"].isin(ignore_vitals)
    vitals_df = vitals_df[ignore_mask]

    # convert to float
    vitals_df["VITAL_SIGN_VALUE"] = vitals_df["VITAL_SIGN_VALUE"].astype(float)
    vitals_feature = aggregate_ctn_feature(
        vitals_df,
        agg_on="VITAL_SIGN_TYPE",
        agg_values_col="VITAL_SIGN_VALUE",
        time_col="VITAL_SIGN_TAKEN_TIME",
        time_interval=time_interval,
        time_window=time_window,
    )

    return vitals_feature


def split_sbp_and_dbp(vitals_df: pd.DataFrame) -> pd.DataFrame:
    # Split BP into SBP and DBP
    vitals_df["VITAL_SIGN_TYPE"].replace({"BP": "SBP/DBP"}, inplace=True)
    explode_cols = ["VITAL_SIGN_VALUE", "VITAL_SIGN_TYPE"]

    # Ref: https://stackoverflow.com/a/57122617/1888794
    # don't explode the columsn you set index to, explode the rest via apply, reset everything to normal
    vitals_df = (
        vitals_df.set_index(list(vitals_df.columns.difference(explode_cols)))
        .apply(lambda col: col.str.split("/").explode())
        .reset_index()
        .reindex(vitals_df.columns, axis=1)
    )
    return vitals_df


def load_medications(
    raw_data_dir: str,
    rx_file: str = "Medications.txt",
    time_interval: Optional[str] = None,
    time_window: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    NOTE: The medications file originally was Medications_19-000093_10082020.txt
    We needed epic medication classes for less granularity which was processed and ran
    by Javier and returned as meds.txt.

    There are 3 classes: ["MEDISPAN_CLASS_NAME", "THERA_CLASS", "PHARM_SUBCLASS"]
    This would reduce us to: 99, 18, and 459 extra flags for medications respectively.
    """
    loading_message("Medications")
    rx_df = read_files_and_combine([rx_file], raw_data_dir)
    rx_feature = aggregate_cat_feature(
        rx_df,
        agg_on="PHARM_SUBCLASS",
        time_col="ORDER_DATE",
        time_interval=time_interval,
        time_window=time_window,
    )
    return rx_feature


def load_labs(
    raw_data_dir: str,
    labs_file: str = "Labs.txt",
    time_interval: Optional[str] = None,
    time_window: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    loading_message("Labs")
    labs_df = read_files_and_combine([labs_file], raw_data_dir)
    # Force numeric, ignore strings
    labs_df["RESULTS"] = pd.to_numeric(labs_df["RESULTS"], errors="coerce")

    labs_feature = aggregate_ctn_feature(
        labs_df,
        # DESCRIPTION will give "Basic Metabolic Panel" even if internally it's "Sodium"
        # TODO: Sodium(LDQ) vs Sodium under description=sodium
        agg_on="COMPONENT_NAME",
        agg_values_col="RESULTS",
        time_col="ORDER_TIME",
        time_interval=time_interval,
        time_window=time_window,
    )

    return labs_feature


def load_problems(
    raw_data_dir: str,
    problems_file: str = "Problem_Lists.txt",
    problems_dx_file: str = "Problem_List_Diagnoses.txt",
    time_interval: Optional[str] = None,
    time_window: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    loading_message("Problems")
    problems_df = read_files_and_combine(
        [problems_dx_file, problems_file], raw_data_dir
    )
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
            "pr_CCS_CODE",
            "pr_CCS_DESCRIPTION",
            "pr_CCS_LEVEL1",
            "pr_CCS_LEVEL1_DESCRIPTION",
            "pr_CCS_LEVEL2",
            "pr_CCS_LEVEL2_DESCRIPTION",
        ],
        hcuppy_converter_function=ce.get_ccs,
    )

    problems_feature = aggregate_cat_feature(
        problems_df,
        agg_on="pr_CCS_CODE",
        time_col="NOTED_DATE",
        time_interval=time_interval,
        time_window=time_window,
    )

    return problems_feature


def load_procedures(
    raw_data_dir: str,
    procedures_file: str = "Procedures.txt",
    time_interval: Optional[str] = None,
    time_window: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    loading_message("Procedures")
    procedures_df = read_files_and_combine([procedures_file], raw_data_dir)

    # Convert CPT codes to their sections (less granular)
    cpt = CPT()
    procedures_df = hcuppy_map_code(
        procedures_df,
        code_col="PROC_CODE",
        exploded_cols=["CPT_SECTION", "SECTION_DESCRIPTION"],
        hcuppy_converter_function=cpt.get_cpt_section,
    )

    procedures_feature = aggregate_cat_feature(
        procedures_df,
        agg_on="CPT_SECTION",
        time_col="PROC_DATE",
        time_interval=time_interval,
        time_window=time_window,
    )

    # Any indication of inpatient surgery before crrt start
    surgery_indicator = "CPT_SECTION_CPT1-C"
    # TODO: filter to the past week regardless of time window. or just check the codes directly?
    procedures_feature["Surgery in Past Week"] = (
        procedures_feature[surgery_indicator] > 0
    )

    return procedures_feature
