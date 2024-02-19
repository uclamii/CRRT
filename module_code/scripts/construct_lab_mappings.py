from argparse import Namespace
import pickle
import sys
from os import getcwd
from os.path import join, isfile
from pandas import DataFrame, read_excel
import re
from fuzzywuzzy import fuzz
import jellyfish
import itertools

sys.path.insert(0, join(getcwd(), "module_code"))

from cli_utils import load_cli_args, init_cli_args
from data.utils import FILE_NAMES, read_files_and_combine
from data.longitudinal_features import map_encounter_to_patient


def jelly_pairwise_diff(left: list, right: list) -> DataFrame:
    results = {
        combo: jellyfish.levenshtein_distance(*combo)
        for combo in itertools.product(left, right)
    }
    distances = DataFrame(
        {"Distance": results.values()}, index=results.keys()
    ).sort_values("Distance")
    return distances


def fuzzy_pairwise_diff(left: list, right: list, mode: str = None) -> DataFrame:
    if mode == "sort":
        comparison_function = fuzz.token_sort_ratio
    elif mode == "partial":
        comparison_function = fuzz.partial_ratio
    else:
        comparison_function = fuzz.ratio

    results = {
        combo: comparison_function(*combo) for combo in itertools.product(left, right)
    }
    distances = DataFrame(
        {"Distance": results.values()}, index=results.keys()
    ).sort_values("Distance", ascending=False)
    return distances


def create_labs_mapping_dict(args: Namespace, cedars_labs: set, ucla_labs: set) -> None:
    matching_cedars_lab_names = []
    matching_ucla_lab_names = []

    ucla_labs["REGEX_COMPONENT_NAME"] = ucla_labs["COMPONENT_NAME"].str.replace(
        "[^a-zA-Z0-9]", " ", regex=True
    )
    ucla_labs["REGEX_COMPONENT_NAME"] = ucla_labs["REGEX_COMPONENT_NAME"].apply(
        lambda x: re.sub(" +", " ", x).strip()
    )

    cedars_labs["REGEX_COMPONENT_NAME"] = cedars_labs["COMPONENT_NAME"].str.replace(
        "[^a-zA-Z0-9]", " ", regex=True
    )
    cedars_labs["REGEX_COMPONENT_NAME"] = cedars_labs["REGEX_COMPONENT_NAME"].apply(
        lambda x: re.sub(" +", " ", x).strip()
    )

    only_cedars = (set(cedars_labs["REGEX_COMPONENT_NAME"].unique())).difference(
        set(ucla_labs["REGEX_COMPONENT_NAME"].unique())
    )
    all_ucla = set(ucla_labs["REGEX_COMPONENT_NAME"].unique())
    distances = jelly_pairwise_diff(only_cedars, all_ucla)
    pairs_with_one_char_diff = distances[distances["Distance"] == 1]

    # Single character differences. Find differences that are either a space or an 'S' (indicates plurality)
    for i, row in pairs_with_one_char_diff.iterrows():
        cedars_characters = dict.fromkeys(row.name[0], 0)
        ucla_characters = dict.fromkeys(row.name[1], 0)

        diff_chars = set(cedars_characters.keys()).symmetric_difference(
            set(ucla_characters.keys())
        )

        if len(diff_chars) > 0:
            if "S" in diff_chars or " " in diff_chars:
                matching_cedars_lab_names.append(row.name[0])
                matching_ucla_lab_names.append(row.name[1])
        else:
            for character in cedars_characters.keys():
                cedars_characters[character] = row.name[0].count(character)
                ucla_characters[character] = row.name[1].count(character)

                if cedars_characters[character] != ucla_characters[character]:
                    if character == "S" or character == " ":
                        matching_cedars_lab_names.append(row.name[0])
                        matching_ucla_lab_names.append(row.name[1])

    # Same code just reorganized string
    dist_sorted = fuzzy_pairwise_diff(only_cedars, all_ucla, mode="sort")

    # Highly similar. Manually inspected that >= 97 is the same except for one case
    similar_strings = dist_sorted[dist_sorted["Distance"] >= 97]
    for i, row in similar_strings.iterrows():
        if "FACTOR" not in row.name[1]:  # Exclude Factor VII mapping to Factor VIII
            matching_cedars_lab_names.append(row.name[0])
            matching_ucla_lab_names.append(row.name[1])

    # One exception from 96
    similar_strings = dist_sorted[dist_sorted["Distance"] == 96]
    for i, row in similar_strings.iterrows():
        if "ESTIMATE" in row.name[1]:
            matching_cedars_lab_names.append(row.name[0])
            matching_ucla_lab_names.append(row.name[1])

    # Get overlap
    same = list(
        set(cedars_labs["REGEX_COMPONENT_NAME"]).intersection(
            ucla_labs["REGEX_COMPONENT_NAME"]
        )
    )

    # Aggregate
    matching_cedars_lab_names = matching_cedars_lab_names + same
    matching_ucla_lab_names = matching_ucla_lab_names + same

    # Get rows with mappinngs
    cedars_mapping_df = cedars_labs[
        cedars_labs["REGEX_COMPONENT_NAME"].isin(matching_cedars_lab_names)
    ]
    ucla_mapping_df = ucla_labs[
        ucla_labs["REGEX_COMPONENT_NAME"].isin(matching_ucla_lab_names)
    ]

    # Map cedars to adjusted labs
    cedars_lab_mapping = dict(
        zip(
            cedars_mapping_df["COMPONENT_NAME"],
            cedars_mapping_df["REGEX_COMPONENT_NAME"],
        )
    )

    # Map cedars adjusted labs to ucla adjusted labs
    REGEX_lab_mapping = dict(zip(matching_cedars_lab_names, matching_ucla_lab_names))

    # Map ucla adjusted labs to ucla labs
    ucla_lab_mapping = dict(
        zip(ucla_mapping_df["REGEX_COMPONENT_NAME"], ucla_mapping_df["COMPONENT_NAME"])
    )

    # Create a final direct mapping
    final_mapping = {}
    for key in cedars_lab_mapping:
        final_mapping[key] = ucla_lab_mapping[
            REGEX_lab_mapping[cedars_lab_mapping[key]]
        ]

    print(len(final_mapping))
    return final_mapping


def load_labs_from_scratch(args):
    cedars_labs = read_files_and_combine(
        [FILE_NAMES["labs"]], args.cedars_crrt_data_dir
    )
    ucla_labs = read_files_and_combine([FILE_NAMES["labs"]], args.ucla_crrt_data_dir)

    cedars_labs = map_encounter_to_patient(args.cedars_crrt_data_dir, cedars_labs)

    # Cedars alignment
    cedars_labs = cedars_labs.rename(
        {"RESULT": "RESULTS", "NAME": "COMPONENT_NAME"}, axis=1
    )
    ucla_labs = ucla_labs.rename(
        {"RESULT": "RESULTS", "NAME": "COMPONENT_NAME"}, axis=1
    )

    final_mapping = create_labs_mapping_dict(args, cedars_labs, ucla_labs)
    return final_mapping


def use_manual_file(cohort):
    df = read_excel("../Data/manual_labs_mapping.xlsx")
    mapping = {}
    for i, row in df.iterrows():
        # nan
        if row[cohort] != row[cohort]:
            continue
        if row[f"{cohort} Map"] != row[f"{cohort} Map"]:
            continue

        if row[cohort] != row[f"{cohort} Map"]:
            if row[cohort] in mapping.keys():
                assert mapping[row[cohort]] == row[f"{cohort} Map"], print(
                    cohort, mapping[row[cohort]], row[f"{cohort} Map"]
                )
            mapping[row[cohort]] = row[f"{cohort} Map"]
    return mapping


def main():
    load_cli_args()
    args = init_cli_args()

    if isfile("../Data/manual_labs_mapping.xlsx"):
        cohorts = ["UCLA CRRT", "UCLA CRRT", "Cedars CRRT"]
        for cohort, raw_data_dir in zip(
            cohorts,
            [
                args.ucla_crrt_data_dir,
                args.ucla_control_data_dir,
                args.cedars_crrt_data_dir,
            ],
        ):
            mapping = use_manual_file(cohort)
            with open(join(raw_data_dir, "Labs_Mapping.pkl"), "wb") as f:
                pickle.dump(mapping, f)
    else:
        mapping = load_labs_from_scratch(args)
        for raw_data_dir in [
            args.ucla_crrt_data_dir,
            args.ucla_control_data_dir,
            args.cedars_crrt_data_dir,
        ]:
            # Save mapping
            with open(join(raw_data_dir, "Labs_Mapping.pkl"), "wb") as f:
                pickle.dump(mapping, f)


if __name__ == "__main__":
    main()
