import logging
from os.path import isfile, join
from typing import Optional, Union
from pandas import DataFrame, to_numeric
from hcuppy.ccs import CCSEngine
from hcuppy.cpt import CPT


from data.longitudinal_utils import (
    aggregate_cat_feature,
    aggregate_ctn_feature,
    hcuppy_map_code,
)
from data.utils import FILE_NAMES, loading_message, read_files_and_combine

"""
Prefix a OR b = (a|b) followed by _ and 1+ characters of any char.
{ diagnoses: dx, meds: PHARM_SUBCLASS, problems: pr, procedures: CPT }
"""
CATEGORICAL_COL_REGEX = r"(dx|PHARM_SUBCLASS|pr|CPT|)_.*"
# CONTINUOUS_COL_REGEX = r"(VITAL_SIGN|RESULT)_.*"


def load_diagnoses(
    raw_data_dir: str,
    dx_file: str = FILE_NAMES["dx"],
    time_interval: Optional[str] = None,
    time_window: Optional[Union[DataFrame, str]] = None,
) -> DataFrame:
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
    vitals_file: str = FILE_NAMES["vitals"],
    time_interval: Optional[str] = None,
    time_window: Optional[Union[DataFrame, str]] = None,
) -> DataFrame:
    loading_message("Vitals")
    vitals_df = read_files_and_combine([vitals_file], raw_data_dir)
    vitals_df = unify_vital_names(vitals_df)
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


def unify_vital_names(vitals_df: DataFrame) -> DataFrame:
    """Refer to `notebooks/align_crrt_and_ctrl.ipynb`"""
    # TODO: match case sensitivity of vital names between controls and crrt? all to caps? all to lower?
    mapping = {
        "Temp": "Temperature",
        "BMI (Calculated)": "BMI",
        "R BMI": "BMI",
        "WEIGHT/SCALE": "Weight",
        "BP": "SBP/DBP",
        "BLOOD PRESSURE": "SBP/DBP",
        "Resp": "Respirations",
        "PULSE OXIMETRY": "SpO2",
    }
    return vitals_df.replace({"VITAL_SIGN_TYPE": mapping})


def split_sbp_and_dbp(vitals_df: DataFrame) -> DataFrame:
    # Split BP into SBP and DBP
    explode_cols = ["VITAL_SIGN_VALUE", "VITAL_SIGN_TYPE"]

    # Ref: https://stackoverflow.com/a/57122617/1888794
    vitals_df = vitals_df.apply(
        lambda col: col.str.split("/") if col.name in explode_cols else col
    ).explode(explode_cols)
    return vitals_df


def load_medications(
    raw_data_dir: str,
    rx_file: str = FILE_NAMES["rx"],
    time_interval: Optional[str] = None,
    time_window: Optional[Union[DataFrame, str]] = None,
) -> DataFrame:
    """
    NOTE: The medications file originally was Medications_19-000093_10082020.txt
    We needed epic medication classes for less granularity which was processed and ran
    by Javier and returned as meds.txt.

    There are 3 classes: ["MEDISPAN_CLASS_NAME", "THERA_CLASS", "PHARM_SUBCLASS"]
    This would reduce us to: 99, 18, and 459 extra flags for medications respectively.
    """
    loading_message("Medications")
    rx_df = read_files_and_combine([rx_file], raw_data_dir)
    rx_df = rx_df.rename({"MEDISPAN_SUBCLASS_NAME": "PHARM_SUBCLASS"}, axis=1)
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
    labs_file: str = FILE_NAMES["labs"],
    time_interval: Optional[str] = None,
    time_window: Optional[Union[DataFrame, str]] = None,
) -> DataFrame:
    loading_message("Labs")
    labs_df = read_files_and_combine([labs_file], raw_data_dir)
    labs_df = labs_df.rename({"RESULT": "RESULTS"}, axis=1)
    # Force numeric, ignore strings
    labs_df["RESULTS"] = to_numeric(labs_df["RESULTS"], errors="coerce")

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
    problems_file: str = FILE_NAMES["pr"],
    problems_dx_file: str = FILE_NAMES["pr_dx"],
    time_interval: Optional[str] = None,
    time_window: Optional[Union[DataFrame, str]] = None,
) -> DataFrame:
    loading_message("Problems")
    # Control data doesn't have the dx_file. Only keep the files that exist.
    files = [
        file
        for file in [problems_dx_file, problems_file]
        if isfile(join(raw_data_dir, file))
    ]  # But we need to be sure that at least one exists
    assert len(files) > 0, "No problems files found."
    problems_df = read_files_and_combine(files, raw_data_dir)
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


# TODO: make all other functions look like this one? (code_col, time_col, aggregate added as parameters)
def load_procedures(
    raw_data_dir: str,
    procedures_file: str = FILE_NAMES["cpt"],
    code_col="PROC_CODE",
    time_col="PROC_DATE",
    time_interval: Optional[str] = None,
    time_window: Optional[Union[DataFrame, str]] = None,
    aggregate: bool = True,
) -> DataFrame:
    loading_message("Procedures")
    procedures_df = read_files_and_combine([procedures_file], raw_data_dir)
    procedures_df = procedures_df.rename(
        {"PROCEDURE_CODE": "PROC_CODE", "PROCEDURE_DATE": "PROC_DATE"}, axis=1
    )

    # Convert CPT codes to their sections (less granular)
    cpt = CPT()
    procedures_df = hcuppy_map_code(
        procedures_df,
        code_col=code_col,
        exploded_cols=["CPT_SECTION", "SECTION_DESCRIPTION"],
        hcuppy_converter_function=cpt.get_cpt_section,
    )

    if not aggregate:
        return procedures_df

    procedures_feature = aggregate_cat_feature(
        procedures_df,
        agg_on="CPT_SECTION",
        time_col=time_col,
        time_interval=time_interval,
        time_window=time_window,
    )

    # Any indication of inpatient surgery before crrt start
    surgery_indicator = "CPT_SECTION_CPT1-C"
    # TODO[HIGH]: filter to the past week regardless of time window. or just check the codes directly?
    procedures_feature["Surgery in Past Week"] = (
        procedures_feature[surgery_indicator] > 0
    ).astype(int)

    return procedures_feature
