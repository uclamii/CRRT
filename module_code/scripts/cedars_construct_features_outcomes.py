"""
Conversion of raw files from Cedars to similar format as UCLA
"""

from argparse import Namespace
import pandas as pd
from os.path import join
from os import getcwd
import sys

sys.path.insert(0, join(getcwd(), "module_code"))

from cli_utils import load_cli_args, init_cli_args

INPUT_FILE = "Cedars CRRT Outcomes Jul'16-Dec'21_anonymized_AG_11.07.22.xlsx"
INPUT_SHEET = "PAT_ID ONLY"
OUTPUT_FILE = "CRRT Deidentified 2015-2021YTD_VF.xlsx"
OUTPUT_SHEET = "2015-2021 YTD"

mapping = {
    "karumanchi_00001867_adt - anonymized.xlsx": "Hospital_Unit_Transfers.txt",
    "karumanchi_00001867_allergy - anonymized.xlsx": "Allergies.txt",
    "karumanchi_00001867_demographics - anonymized.xlsx": "Patient_Demographics.txt",
    "karumanchi_00001867_enc_dx - anonymized.xlsx": "Encounter_Diagnoses.txt",
    "karumanchi_00001867_encounters - anonymized.xlsx": "Encounters.txt",
    "karumanchi_00001867_family_hx - anonymized.xlsx": "Family_History.txt",
    "karumanchi_00001867_flowsheet_daily_io - anonymized.xlsx": None,
    "karumanchi_00001867_flowsheet_vitals - anonymized.xlsx": "Flowsheet_Vitals.txt",
    "karumanchi_00001867_labs - anonymized.xlsx": "Labs.txt",
    "karumanchi_00001867_medications - anonymized.xlsx": "Medications.txt",
    "karumanchi_00001867_problem_list - anonymized - updated.xlsx": "Problem_Lists.txt",  # 'Problem_List_Diagnoses.txt'
    "karumanchi_00001867_procedures - anonymized.xlsx": "Procedures.txt",
    "karumanchi_00001867_social_hx - anonymized (1).xlsx": "Social_History.txt",
}
# Missing from cedars: Providers


def align_outcomes(args: Namespace):
    """
    Align Cedars outcomes with UCLA. Consists of:
        * Renaming the patient ID column
        * Creating a start date column
        * Enforcing end date to be a datetime object
        * Reformatting the Month column to match UCLA
    """

    df = pd.read_excel(join(args.cedars_crrt_data_dir, INPUT_FILE), "PAT_ID ONLY")

    df.rename(columns={"PAT_ID": "IP_PATIENT_ID"}, inplace=True)

    #### Construct Start Date ####  -- For convenience of time-windows --

    # CRRT Start Date = End Date - (Days on CRRT - 1)
    # e.g. finish on the 10th and 3 days of CRRT: 8th (1), 9th (2), 10th (3)
    df["Start Date"] = pd.DatetimeIndex(df["End Date"]) - pd.to_timedelta(
        df["CRRT Total Days"] - 1, unit="d"
    )

    # Enforce date column to datetime object
    df["End Date"] = pd.to_datetime(df["End Date"])

    # UCLA Month format is <Abbrev_month>-<Abbrev_year>. Adjust Cedars to match
    df["Month"] = df["Month"].apply(lambda x: x.strftime("%b-%y"))

    with pd.ExcelWriter(join(args.cedars_crrt_data_dir, OUTPUT_FILE)) as writer:
        df.to_excel(writer, index=False, sheet_name=OUTPUT_SHEET)


def rename_features(args: Namespace):
    """
    Align Cedars features with UCLA. Consists of:
        * Renaming the patient ID column
        * Renaming the files to match FILE_NAMES from UCLA
    """

    for input_feature, output_feature in mapping.items():
        if output_feature is not None:
            df = pd.read_excel(join(args.cedars_crrt_data_dir, input_feature))
            df.rename(
                columns={
                    "PAT_ID": "IP_PATIENT_ID",
                    "PAT_ENC_CSN_ID": "IP_ENCOUNTER_ID",
                },
                inplace=True,
            )
            df.to_csv(join(args.cedars_crrt_data_dir, output_feature), index=False)


def main():
    load_cli_args()
    args = init_cli_args()
    align_outcomes(args)
    rename_features(args)


if __name__ == "__main__":
    main()
