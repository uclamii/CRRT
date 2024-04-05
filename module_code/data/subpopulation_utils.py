"""
maps indication group name to a dictionary of the category name within that group, and a tuple of its column name and its value or range of values.
"""

from typing import Dict, Tuple, Union, List, Any


def generate_filters() -> Dict[str, Dict[str, Tuple[str, Union[int, Tuple[int, int]]]]]:
    # these filters will be passed to `SklearnCRRTDataModule.get_filter`
    disease_filter_args = {  # filter based on disease status
        "heart": ("heart_pt_indicator", 1),
        "liver": ("liver_pt_indicator", 1),
        "infection": ("infection_pt_indicator", 1),
        "no_heart_liver_infection": (
            ["infection_pt_indicator", "heart_pt_indicator", "liver_pt_indicator"],
            [0, 0, 0],
        ),
    }
    sex_filter_args = {  # filter based on demographic
        "female": ("SEX", 1),
        "male": ("SEX", 0),
    }
    age_filter_args = {  # age groups every decade until 100 years (we filter over 100)
        f"age_{range[0]}_to_{range[1]}": ("AGE", range)
        for range in [(n, min(n + 10, 100)) for n in range(20, 100, 10)]
    }
    ethnicity_filter_args = {
        name: ("ETHNICITY", i)
        for i, name in enumerate(["Not Hispanic or Latino", "Hispanic or Latino"])
    }
    race_filter_args = {
        race: (f"RACE_{race}", 1)
        # TODO: pulled from descriptive_report.ipynb, maybe make it auto?
        for race in [
            "American Indian or Alaska Native",
            "Asian",
            "Black or African American",
            "Multiple Races",
            "Native Hawaiian or Other Pacific Islander",
            "Unknown",
            "White or Caucasian",
        ]
    }

    filters = {
        "disease_indicator": disease_filter_args,
        "age_groups": age_filter_args,
        "sex": sex_filter_args,
        "race": race_filter_args,
        "ethnicity": ethnicity_filter_args,
        # **cartesian_product,
    }
    return filters


def combine_filters(
    f1: Dict[str, Tuple[str, Union[int, Tuple[int, int]]]],
    f2: Dict[str, Tuple[str, Union[int, Tuple[int, int]]]],
) -> Dict[str, Tuple[str, Union[int, Tuple[int, int]]]]:
    def listify(item: Union[Any, List[Any]]) -> List[Any]:
        return item if isinstance(item, list) else [item]

    combined = {}
    for f1_category_name, f1_args in f1.items():
        for f2_category_name, f2_args in f2.items():
            f1_cols, f1_vals = f1_args
            f2_cols, f2_vals = f2_args
            combined[f"{f1_category_name}_{f2_category_name}"] = (
                listify(f1_cols) + listify(f2_cols),
                listify(f1_vals) + listify(f2_vals),
            )
    return combined
