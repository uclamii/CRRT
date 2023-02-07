import logging
from pickle import load
from os.path import isfile, join
from typing import Optional, Union
from pandas import DataFrame, to_numeric, concat
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

    # Cedars alignment. Assume contact date is diagnosis date since that's the date we have
    dx_df = dx_df.rename(
        {"CURRENT_ICD10_LIST": "ICD_CODE", "CONTACT_DATE": "DIAGNOSIS_DATE"}, axis=1
    )

    # convert icd10 to ccs to reduce number of categories for diagnoses.
    ce = CCSEngine(mode="dx")
    # ICD9 is 2013-15. Outcomes are 2018-19, so we will ignore ICD9 codes for now.
    # They will just show as NaNs.
    # NOTE: This needs to change if our time window gets very large and we extend into 2013-15.
    # icd10_mask = dx_df["ICD_TYPE"] == 10
    dx_df = hcuppy_map_code(
        dx_df,
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

    # Cedars alignment.
    vitals_df = vitals_df.rename(
        {
            "MEAS_NAME": "VITAL_SIGN_TYPE",
            "RECORDED_TIME": "VITAL_SIGN_TAKEN_TIME",
            "MEAS_VALUE": "VITAL_SIGN_VALUE",
        },
        axis=1,
    )

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

    # calculate BMI if missing. Should only be the case for Cedars
    vitals_df = calculate_bmi(vitals_df)

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
        # UCLA
        "Temp": "Temperature",
        "BMI (Calculated)": "BMI",
        "R BMI": "BMI",
        "WEIGHT/SCALE": "Weight",
        "BP": "SBP/DBP",
        "BLOOD PRESSURE": "SBP/DBP",
        "Resp": "Respirations",
        "PULSE OXIMETRY": "SpO2",
        # Cedars
        "HEIGHT_IN": "Height",
        "TEMP": "Temperature",
        "O2_SATURATION": "SpO2",
        "RESP_RATE": "Respirations",
        "HEART_RATE": "Pulse",
        "WEIGHT_OZ": "Weight",
    }
    return vitals_df.replace({"VITAL_SIGN_TYPE": mapping})


def split_sbp_and_dbp(vitals_df: DataFrame) -> DataFrame:
    # Split BP into SBP and DBP
    explode_cols = ["VITAL_SIGN_VALUE", "VITAL_SIGN_TYPE"]

    # Cedars has some na which fails on the split below
    old_size = vitals_df.shape[0]
    vitals_df = vitals_df.dropna(subset=["VITAL_SIGN_VALUE"])
    logging.info(f"Dropped {old_size - vitals_df.shape[0]} rows that were na.")

    # Ref: https://stackoverflow.com/a/57122617/1888794
    vitals_df = vitals_df.apply(
        lambda col: col.str.split("/") if col.name in explode_cols else col
    ).explode(explode_cols)

    return vitals_df


def calculate_bmi(vitals_df: DataFrame) -> DataFrame:
    """
    UCLA has BMI as a function of height and weight. Cedars does not explicitly have this but can calculate
    Rule: for each patient, if they had weight and height measured at the same time, calculate BMI for that time.
        strict with time because measurements can vary if they are taken multiple times a dat
    """

    if "BMI" in vitals_df["VITAL_SIGN_TYPE"].unique():
        return vitals_df

    # Get rows that document weight
    weight = vitals_df.loc[vitals_df["VITAL_SIGN_TYPE"] == "Weight"].copy()

    # Get rows that document height
    height = vitals_df.loc[vitals_df["VITAL_SIGN_TYPE"] == "Height"].copy()

    # Merge weight and height only for measurements that occured for the same patients at the same time
    # Since weight and height are located the same column (VITAL_SIGN_VALUE), we must remember weight as VITAL_SIGN_VALUE_x and height as VITAL_SIGN_VALUE_y
    bmi = weight.merge(
        height, on=["IP_PATIENT_ID", "VITAL_SIGN_TAKEN_TIME"], how="inner"
    )

    # Calculate BMI as 703*weight_in_lb/height_in_inch^2
    bmi["BMI"] = 703 / 16 * bmi["VITAL_SIGN_VALUE_x"] / bmi["VITAL_SIGN_VALUE_y"] ** 2

    # Drop unecessary columns due to the merge. Can probably be done in a cleaner way
    bmi = bmi.drop(
        [
            "INPATIENT_DATA_ID_y",
            "VITAL_SIGN_TYPE_y",
            "VITAL_SIGN_TYPE_x",
            "VITAL_SIGN_VALUE_y",
            "VITAL_SIGN_VALUE_x",
        ],
        axis=1,
    )

    # Rename to align with the original dataframe
    bmi = bmi.rename(
        columns={"INPATIENT_DATA_ID_x": "INPATIENT_DATA_ID", "BMI": "VITAL_SIGN_VALUE"}
    )

    # Fill out a the VITAL_SIGN_TYPE column with 'BMI' since all values are BMI
    bmi["VITAL_SIGN_TYPE"] = "BMI"

    # Return concatenation
    return concat([vitals_df, bmi])


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


    Cedars has: ["PHARM_SUBCLASS, "THERA_CLASS", "PHARM_CLASS"]
    """
    loading_message("Medications")
    rx_df = read_files_and_combine([rx_file], raw_data_dir)

    rx_df = rx_df.rename(
        {
            "MEDISPAN_SUBCLASS_NAME": "PHARM_SUBCLASS",
            "ORDERING_DATE": "ORDER_DATE",
            "NAME": "MEDICATION_NAME",
        },
        axis=1,
    )

    # Additional cleanup
    rx_df["PHARM_SUBCLASS"] = rx_df["PHARM_SUBCLASS"].str.upper()

    rx_df = map_medications(rx_df, raw_data_dir)

    rx_feature = aggregate_cat_feature(
        rx_df,
        agg_on="PHARM_SUBCLASS",
        time_col="ORDER_DATE",
        time_interval=time_interval,
        time_window=time_window,
    )
    return rx_feature


def map_medications(
    rx_df: DataFrame,
    raw_data_dir: str,
    medication_mapping_file: str = "Medications_Mapping.pkl",
) -> DataFrame:

    if not isfile(join(raw_data_dir, medication_mapping_file)):
        return rx_df

    with open(join(raw_data_dir, medication_mapping_file), "rb") as f:
        loaded_dict = load(f)

    rx_df["PHARM_SUBCLASS"] = rx_df["PHARM_SUBCLASS"].replace(loaded_dict)

    return rx_df


def load_labs(
    raw_data_dir: str,
    labs_file: str = FILE_NAMES["labs"],
    time_interval: Optional[str] = None,
    time_window: Optional[Union[DataFrame, str]] = None,
) -> DataFrame:
    loading_message("Labs")
    labs_df = read_files_and_combine([labs_file], raw_data_dir)

    labs_df = map_encounter_to_patient(raw_data_dir, labs_df)

    labs_df = labs_df.rename({"RESULT": "RESULTS", "NAME": "COMPONENT_NAME"}, axis=1)

    labs_df = map_labs(labs_df, raw_data_dir)

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


def map_encounter_to_patient(
    raw_data_dir: str, df: DataFrame, encounter_file: str = FILE_NAMES["enc"]
):

    loading_message("Encounters")
    enc_df = read_files_and_combine([encounter_file], raw_data_dir)

    # Left merge adds a new IP_PATIENT_ID_y column for IP_ENCOUNTER_ID in enc_df that exist in df
    # The original IP_PATIENT_ID is saved as IP_PATIENT_ID_x
    df = df.merge(
        enc_df[["IP_ENCOUNTER_ID", "IP_PATIENT_ID"]], on="IP_ENCOUNTER_ID", how="left"
    )

    # The combine_first column fills ONLY the nan rows in IP_PATIENT_ID_x with values from IP_PATIENT_ID_y
    # In essence, keep original IP_PATIENT_ID if it existed, else fill with the new from the encounters file
    df["IP_PATIENT_ID"] = df["IP_PATIENT_ID_x"].combine_first(df["IP_PATIENT_ID_y"])

    # Remove the created columns
    df = df.drop(["IP_PATIENT_ID_x", "IP_PATIENT_ID_y"], 1)

    return df


# TODO: Refactor with the medications mapping
def map_labs(
    static_df: DataFrame,
    raw_data_dir: str,
    proc_mapping_file: str = "Labs_Mapping.pkl",
) -> DataFrame:

    # Should only do for Cedars
    if not isfile(join(raw_data_dir, proc_mapping_file)):
        return static_df

    with open(join(raw_data_dir, proc_mapping_file), "rb") as f:
        loaded_dict = load(f)

    static_df["COMPONENT_NAME"] = static_df["COMPONENT_NAME"].replace(loaded_dict)

    return static_df


def load_problems(
    raw_data_dir: str,
    problems_file: str = FILE_NAMES["pr"],
    problems_dx_file: str = FILE_NAMES["pr_dx"],
    time_interval: Optional[str] = None,
    time_window: Optional[Union[DataFrame, str]] = None,
) -> DataFrame:
    loading_message("Problems")
    # Control and Cedars data doesn't have the dx_file. Only keep the files that exist.
    files = [
        file
        for file in [problems_dx_file, problems_file]
        if isfile(join(raw_data_dir, file))
    ]  # But we need to be sure that at least one exists
    assert len(files) > 0, "No problems files found."
    problems_df = read_files_and_combine(files, raw_data_dir)
    problems_df.columns = [col.upper() for col in problems_df.columns]

    # Cedars alignment
    problems_df = problems_df.rename(
        {"STATUS": "PROBLEM_STATUS", "CURRENT_ICD10_LIST": "ICD_CODE"}, axis=1
    )

    # convert icd10 to ccs only to active problems
    ce = CCSEngine(mode="dx")

    # Cedars does not have ICD_TYPE (all ICD10)
    if "ICD_TYPE" in problems_df.columns:
        active_and_icd10_mask = (problems_df["PROBLEM_STATUS"] == "Active") & (
            problems_df["ICD_TYPE"] == 10
        )
    else:
        active_and_icd10_mask = problems_df["PROBLEM_STATUS"] == "ACTIVE"

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
        {
            # Control
            "PROCEDURE_CODE": "PROC_CODE",
            "PROCEDURE_DATE": "PROC_DATE",
            # Cedars
            "PROC_START_TIME": "PROC_DATE",
        },
        axis=1,
    )

    procedures_df = map_proc_code_to_cpt(procedures_df, raw_data_dir)

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

    if "CPT_SECTION_CPT1-C" in procedures_feature.columns:
        procedures_feature["surgery_indicator"] = (
            procedures_feature[surgery_indicator] > 0
        ).astype(int)

    return procedures_feature


# TODO: Refactor with the medications mapping
def map_proc_code_to_cpt(
    static_df: DataFrame,
    raw_data_dir: str,
    proc_mapping_file: str = "Procedures_Code_Mapping.pkl",
) -> DataFrame:

    # Should only do for Cedars
    if not isfile(join(raw_data_dir, proc_mapping_file)):
        return static_df

    with open(join(raw_data_dir, proc_mapping_file), "rb") as f:
        loaded_dict = load(f)

    static_df["PROC_CODE"] = static_df["PROC_CODE"].astype(str)

    static_df["PROC_CODE"] = static_df["PROC_CODE"].replace(loaded_dict)

    return static_df
