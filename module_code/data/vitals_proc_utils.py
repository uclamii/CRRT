import logging
from pandas import DataFrame, concat, to_datetime


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
    Rule: for each patient, if they had weight and height measured, for each weight, calculate BMI based on the height
            measured at the nearest time.

    Note this doesn't have any optimization and iterates through all patients - might be able to make it faster
    """

    if "BMI" in vitals_df["VITAL_SIGN_TYPE"].unique():
        return vitals_df

    # New DataFrame for BMI
    bmi_df = DataFrame({column: {} for column in vitals_df.columns})

    # Get rows that document weight
    weights = vitals_df.loc[vitals_df["VITAL_SIGN_TYPE"] == "Weight"].copy()
    weights["VITAL_SIGN_TAKEN_TIME"] = to_datetime(weights["VITAL_SIGN_TAKEN_TIME"])

    # Get rows that document height
    heights = vitals_df.loc[vitals_df["VITAL_SIGN_TYPE"] == "Height"].copy()
    heights["VITAL_SIGN_TAKEN_TIME"] = to_datetime(heights["VITAL_SIGN_TAKEN_TIME"])

    # Iterate through all unique patients that have a height measurement
    for patient in heights["IP_PATIENT_ID"].unique():
        # Get the height and weight measurements for that patient
        patient_heights = heights[heights["IP_PATIENT_ID"] == patient].copy()
        patient_weights = weights[weights["IP_PATIENT_ID"] == patient].copy()

        # Iterate through all weights for that unique patient
        for j, weight in patient_weights.iterrows():
            # Get the height measurement from the closest day to the weight measurement
            patient_heights["TIME_DIFF"] = (
                patient_heights["VITAL_SIGN_TAKEN_TIME"]
                - weight["VITAL_SIGN_TAKEN_TIME"]
            )
            selected_height = patient_heights[
                patient_heights["TIME_DIFF"] == patient_heights["TIME_DIFF"].min()
            ]

            # Calculate BMI as 703*weight_in_lb/height_in_inch^2
            bmi = (
                703
                / 16
                * weight["VITAL_SIGN_VALUE"]
                / selected_height["VITAL_SIGN_VALUE"] ** 2
            )

            new_row = {
                "IP_PATIENT_ID": weight["IP_PATIENT_ID"],
                "INPATIENT_DATA_ID": weight["INPATIENT_DATA_ID"],
                "VITAL_SIGN_TAKEN_TIME": weight["VITAL_SIGN_TAKEN_TIME"],
                "VITAL_SIGN_TYPE": "BMI",
                "VITAL_SIGN_VALUE": bmi,
            }
            new_row = DataFrame(new_row)

            bmi_df = concat([bmi_df, new_row])

    # Return concatenation
    return concat([vitals_df, bmi_df])
