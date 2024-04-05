"""
Conversion of raw files from Controls to similar format as UCLA
"""

from argparse import Namespace
import pandas as pd
from os.path import join
from os import getcwd
import sys
import csv

sys.path.insert(0, join(getcwd(), "module_code"))

from cli_utils import load_cli_args, init_cli_args

mapping = {
    "Hospital_Unit_Transfers.csv": "Hospital_Unit_Transfers.txt",
    "Allergies.csv": "Allergies.txt",
    "Patient_Demographics.csv": "Patient_Demographics.txt",
    "Encounter_Diagnoses.csv": "Encounter_Diagnoses.txt",
    "Encounters.csv": "Encounters.txt",
    "Family_History.csv": "Family_History.txt",
    "Flowsheet_Vitals.csv": "Flowsheet_Vitals.txt",
    "Labs.csv": "Labs.txt",
    "Medications.csv": "Medications.txt",
    "Problem_Lists.csv": "Problem_Lists.txt",  # 'Problem_List_Diagnoses.txt'
    "Procedures.csv": "Procedures.txt",
    "Providers.csv": "Providers.txt",
    # Extra
    "Patient_Identifiers.csv": "Patient_Identifiers.txt",
    "Case_to_Control_Cross_Reference.csv": "Case_to_Control_Cross_Reference.txt",
    "Charlson.csv": "Charlson.txt",
}
# Missing from controls: Social History


def rename_features(args: Namespace):
    """
    Align Control features with UCLA. Consists of:
        * Renaming the files to match FILE_NAMES from UCLA
        * Removing columns from specific files to make them readable by pandas
    """

    for input_feature, output_feature in mapping.items():
        if output_feature is not None:
            # TODO: review these columns. These files have columns with sparse values but no column name. Removing them for now
            # Allergies needs deleting 2 columns
            # Medications needs deleting 1 column
            # Problem_Lists needs deleting 6 columns
            # Procedures needs deleting 1 columns
            if input_feature in [
                "Medications.csv",
                "Allergies.csv",
                "Problem_Lists.csv",
                "Procedures.csv",
            ]:
                # Reads just the columns names
                with open(join(args.ucla_control_data_dir, input_feature), "r") as f:
                    reader = csv.reader(f)
                    cols = next(reader)
                    print(len(cols), cols)

                # From the csv, only read the columns with column names
                df = pd.read_csv(
                    join(args.ucla_control_data_dir, input_feature),
                    names=cols,  # pass the column names
                    usecols=list(range(len(cols))),  # limit the width of the csv
                    skiprows=1,  # start reading on the second row (ignore the first row with col names)
                    header=None,  # don't automatically set the column names - use the names passed in "names" argument
                )
            else:
                # Read as normal
                df = pd.read_csv(join(args.ucla_control_data_dir, input_feature))

            # Convert to upper
            df.columns = [x.upper() for x in df.columns]

            # These do not have 'IP_PATIENT_ID'
            if input_feature not in [
                "Providers.csv",
                "Case_to_Control_Cross_Reference.csv",
            ]:
                assert "IP_PATIENT_ID" in df.columns

            # Save
            df.to_csv(join(args.ucla_control_data_dir, output_feature), index=False)


def main():
    load_cli_args()
    args = init_cli_args()
    rename_features(args)


if __name__ == "__main__":
    main()
