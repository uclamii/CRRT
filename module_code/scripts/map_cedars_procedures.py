from argparse import Namespace
from typing import List
import pickle
import sys
from os import getcwd
from os.path import join
from pandas import DataFrame, read_excel
from hcuppy.cpt import CPT

sys.path.insert(0, join(getcwd(), "module_code"))

from cli_utils import load_cli_args, init_cli_args
from data.utils import FILE_NAMES, read_files_and_combine


def trace_code_connections(starting_code: str, mapping_df: DataFrame) -> List[str]:
    """
    Find all the different codes that a starting code is related to, as documented in the mapping_df
    Each PROC_CODE is associated with an OT_PROC_CODE, but mapping is not one-to-one nor one-way
    Perform a depth first search to find all codes that a given code is 'connected' to
    """

    visited_codes = []
    codes_to_visit = [starting_code]

    while len(codes_to_visit) > 0:

        current_code = codes_to_visit.pop()

        if current_code not in visited_codes:

            # get related codes from OT_PROC_CODE when the current code is PROC_CODE
            # AND get related codes from PROC_CODE when the current code is OT_PROC_CODE
            neighbours_forwards = mapping_df[mapping_df["PROC_CODE"] == current_code][
                "OT_PROC_CODE"
            ].to_list()
            neighbours_backwards = mapping_df[
                mapping_df["OT_CODE_TYPE"] == current_code
            ]["PROC_CODE"].to_list()

            codes_to_visit = codes_to_visit + neighbours_forwards + neighbours_backwards

            visited_codes.append(current_code)
    return visited_codes


def create_and_serialize_cpt_mapping_dict(
    args: Namespace, procedures: DataFrame, mapping_df: DataFrame
) -> None:
    cpt = CPT()

    original_codes = []
    cpt_codes = []

    for i, code in enumerate(procedures["PROC_CODE"].unique()):

        # Trace all connected codes
        neighbours = trace_code_connections(code, mapping_df)
        found_code = "na"
        # Look through all neighbors
        for neighbour in neighbours:

            # check if it is a cpt code. if it is not, this will return 'na'
            cpt_code = cpt.get_cpt_section(neighbour)

            # If there is one cpt code, store it
            if cpt_code["sect"] != "na":
                found_code = neighbour
                break

        original_codes.append(code)
        cpt_codes.append(found_code)

    mapping_dict = dict(zip(original_codes, cpt_codes))

    with open(
        join(args.cedars_crrt_data_dir, "Procedures_Code_Mapping.pkl"), "wb"
    ) as f:
        pickle.dump(mapping_dict, f)


def main():
    load_cli_args()
    args = init_cli_args()

    cedars_proc = read_files_and_combine([FILE_NAMES["cpt"]], args.cedars_crrt_data_dir)
    proc_mapping = read_excel(
        join(
            args.cedars_crrt_data_dir, "karumanchi_00001867_proc_code_bridge_table.xlsx"
        )
    )

    cedars_proc["PROC_CODE"] = cedars_proc["PROC_CODE"].astype(
        str
    )  # Already CPT: {99195, 93925, 92950}
    proc_mapping["PROC_CODE"] = proc_mapping["PROC_CODE"].astype(str)
    proc_mapping["OT_PROC_CODE"] = proc_mapping["OT_PROC_CODE"].astype(str)

    create_and_serialize_cpt_mapping_dict(args, cedars_proc, proc_mapping)


if __name__ == "__main__":
    main()
