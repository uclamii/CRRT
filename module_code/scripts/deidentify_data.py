from argparse import ArgumentParser, Namespace
import pandas as pd
from os.path import join

from utils import load_cli_args

# will be deleted once mapping is complete
MAPPING_FILE = "Patient_Identifiers.txt"


def main(args: Namespace):
    mapping = pd.read_csv(join(args.raw_data_dir, MAPPING_FILE))
    mapping = dict(zip(mapping["MRN"], mapping["IP_PATIENT_ID"]))

    unmapped = pd.read_excel(join(args.raw_data_dir, args.file_to_map), sheet_name=None)
    for sheetname, df in unmapped.items():
        mrn_col = "Medical record number"
        if mrn_col not in df.columns:
            mrn_col = "Medical Record Number"
        if mrn_col not in df.columns:
            continue
        # map values
        df[mrn_col] = df[mrn_col].map(mapping)
        # Rename column from MRN to deidendified patient ID
        df.rename(columns={mrn_col: "IP_PATIENT_ID"}, inplace=True)
        #     df.drop(mrn_col, axis=1, inplace=True)

    with pd.ExcelWriter(join(args.raw_data_dir, args.deidentified_file)) as writer:
        for sheetname, df in unmapped.items():
            df.to_excel(writer, index=False, sheet_name=sheetname)


if __name__ == "__main__":
    load_cli_args()
    p = ArgumentParser()
    p.add_argument("--file-to-map", type=str, help="Excel file to deidentify.")
    p.add_argument("--deidentified-file", type=str, help="Output file name.")
    p.add_argument("--raw-data-dir", type=str, help="Root directory of data files.")
    args = p.parse_known_args()[0]
    main(args)
