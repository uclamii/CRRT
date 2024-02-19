from typing import Callable, Dict, Tuple, Union
from pandas import DataFrame, Series, to_numeric
from numpy import nan
from pickle import load
from os.path import isfile, join
from pint import UnitRegistry, Quantity, UndefinedUnitError, DimensionalityError

from data.utils import FILE_NAMES, loading_message, read_files_and_combine

MANUAL_EQUIVALENT_UNITS = [
    {"/hpf", "per hpf"},
    {"ml", "ml/d"},  # 24 HR. URINE VOLUME
    {"cells/ul", "/ul"},  # WBCS and others
    {"%", "% normal"},  # VWF: RISTOCETIN CO-FACTOR
    {"%", "% binding inhibition"},
    {"%", "% baseline"},
    {"%", "% of total"},
    {"%", "% activity"},
    {"%", "g/dl"},
    {"cpy/ml", "copiesml"},
    {"uge/ml", "mcg eq/ml", "ug eq/ml"},
    {"units/ml", "% normal"},
    {"mg/dl", "mg/dl adult"},
    {"au/ml", "u/ml", "titer"},
    {"ng/ml feu", "ng/mlfeu"},
    {"ml/min/1.73m2", "ml/min/bsa", "ml/min/1.73", "ml/min/1.73m", "ml/min"},
    {"mg/dl", "md/dl", "mg/di"},  # typo
    {"ng/ml", "ng/ml rbc"},
    {"au/ml", "ai"},
    {"u/g hgb", "u/g hb"},
    {"liv", "index"},  # LYME DISEASE AB,TOTAL
    {"ng/ml", "eia unit"},
    # ("fl = femtoliter = cu mic = cubic micron")
    {"fl", "cu mic"},
    {"nmol bce/mmol", "nm bce/mm"},
    {"m2", "meters s"},
    {"pg/mlcdf", "pg/ml"},
    {"m/ul", "mi/mm3", "m/mm3", "x10e6/ul"},
    {"iv", "IV", "index", "INDEX"},
]
# https://www.cdc.gov/cliac/docs/addenda/cliac0313/13A_CLIAC_2013March_UnitsOfMeasure.pdf
NUMERICAL_COUNT_UNIT_VALUES = {
    "x10e3": 1e3,
    "thousand": 1e3,
    "thous": 1e3,
    "1000": 1e3,
    "10e3": 1e3,
    "k": 1e3,
    "x10e6": 1e6,
    "10e6": 1e6,
    "mill": 1e6,
    "million": 1e6,
    "X10*9": 1e9,
    "x10*9": 1e9,
    "x10*12": 1e12,
    "cells": 1,
}
MOL_TO_EQ = {  # meq/mol
    "ANION GAP": 1,
    "BASE EXCESS": 1,
    "BICARBONATE": 1,
    "CARBON DIOXIDE": 1,
    "CHLORIDE": 1,
    "MAGNESIUM": 2,
    "PHOSPHORUS": 3,  # 3 or 5
    "POTASSIUM": 1,
    "SODIUM": 1,
}
G_TO_MOL = {  # inverse of molecular weight, so mol/g
    "AMMONIA": 0.058719906,
    "BETA-HYDROXYBUTYRATE": 0.009605,
    "CREATININE": 0.008842,
    "CREATININE, RANDOM URINE": 0.008842,
    "CREATININE,RANDOM URINE": 0.008842,
    "LIPOPROTEIN (A)": 0.0028011204,
    "MAGNESIUM": 0.0411438,
    "METANEPHRINE": 0.0050702226,
    "METANEPHRINE, FREE PLASMA": 0.0050702226,
    "NORMETANEPHRINE, FREE PLASMA": 0.0050702226,
    "PTH-RELATED PROTEIN (PTH-RP)": 0.0001061008,
    "VITAMIN B6": 0.0059101655,
    "VITAMIN C": 0.0056818182,
}


def specific_lab_preproc(df, lab_name):
    """Method to modify a specific lab"""

    # check if lab exists
    # only run on cohorts that contain <lab_name>
    if lab_name in df["COMPONENT_NAME"].unique():
        lab_result = (
            df[df["COMPONENT_NAME"] == lab_name]["RESULTS"]
            .str.extract(r"(\d+(\.\d+)?)%")[0]  # regex to extract percentages
            .astype(float)
        )
        lab_result[lab_result > 100] = nan
        df.loc[df["COMPONENT_NAME"] == lab_name, "RESULTS"] = lab_result
    return df


def force_boolean(s):
    """If the lab result is a string 'negative', assingn to zero"""
    negatives = s.str.contains("(?i)neg|negative|false", na=False)
    s[negatives] = "0"
    return s


def force_to_upper_lower_bound(s):
    """Method to push number to bounds from a string
    # setting > or < to float
    # using regex replace strings with > or < sign that contain a number, keep existing
    # numeric values as well
    """
    return s.str.replace(r"[<>]\s*(\d+)", r"\1")  # regex


def force_lab_numeric(results: Series) -> Series:
    # upper/lower bound by stripping >/<[=] and words
    bounded = results.str.replace("+|<|>|=|[\D]", "")
    # keep numbers from original (strip will be NaN), and make any strings that aren't numeric (e.g. Test Not Performed) NaNs.
    return to_numeric(bounded.where(bounded.notna(), results).replace("", nan))


def map_encounter_to_patient(
    raw_data_dir: str, df: DataFrame, encounter_file: str = FILE_NAMES["enc"]
):
    # skip if all patient ids exist
    if not df["IP_PATIENT_ID"].isnull().values.any():
        return df

    loading_message("Encounters")
    enc_df = read_files_and_combine([encounter_file], raw_data_dir)

    # for controls
    enc_df = enc_df.rename({"IP_ENC_ID": "IP_ENCOUNTER_ID"}, axis=1)

    # Left merge adds a new IP_PATIENT_ID_y column for IP_ENCOUNTER_ID in enc_df that exist in df
    # The original IP_PATIENT_ID is saved as IP_PATIENT_ID_x
    df = df.merge(
        enc_df[["IP_ENCOUNTER_ID", "IP_PATIENT_ID"]], on="IP_ENCOUNTER_ID", how="left"
    )

    # The combine_first column fills ONLY the nan rows in IP_PATIENT_ID_x with values from IP_PATIENT_ID_y
    # In essence, keep original IP_PATIENT_ID if it existed, else fill with the new from the encounters file
    df["IP_PATIENT_ID"] = df["IP_PATIENT_ID_x"].combine_first(df["IP_PATIENT_ID_y"])

    # Remove the created columns
    df = df.drop(["IP_PATIENT_ID_x", "IP_PATIENT_ID_y"], axis=1)

    return df


# TODO: Refactor with the medications mapping
def map_labs(
    static_df: DataFrame,
    raw_data_dir: str,
    lab_mapping_file: str = "Labs_Mapping.pkl",
) -> DataFrame:
    # Should only do for Cedars
    if not isfile(join(raw_data_dir, lab_mapping_file)):
        return static_df

    with open(join(raw_data_dir, lab_mapping_file), "rb") as f:
        loaded_dict = load(f)

    # This is a expensive replace (millions of rows)
    # Note, if using .replace(), a huge amount of overhead is observed
    # .map(dict.get) is SIGNIFICANTLY faster. The caveat is that replace can
    #   handle when a code is not in the mapping. map requires that all codes be in
    #   the mapping dictionary. Use the mask to get around this issue
    mask = static_df["COMPONENT_NAME"].isin(loaded_dict.keys())
    static_df.loc[mask, "COMPONENT_NAME"] = static_df.loc[mask, "COMPONENT_NAME"].map(
        loaded_dict.get
    )
    # static_df["COMPONENT_NAME"] = static_df["COMPONENT_NAME"].replace(loaded_dict)

    # pretty crude, some labs should be split into venous/arterial. right now, encode possible mappings with $, with the convention of arterial being first. Iterate and find these
    def venous_arterial_split(description, component):
        if "ven" in description.lower():
            return component.split("$")[-1]
        elif "art" in description.lower():
            return component.split("$")[0]
        else:
            return component.split("$")[0]

    POSSIBLE_VEN_ART = [
        "BICARBONATE, ARTERIAL$BICARBONATE, VENOUS",
        "BASE EXCESS, ARTERIAL$BASE EXCESS, VENOUS",
        "O2 SATURATION-ARTERIAL$O2 SATURATION-VENOUS",
        "PH, ARTERIAL$PH,VENOUS",
        "PCO2, ARTERIAL$PCO2,VENOUS",
        "PO2, ARTERIAL$PO2,VENOUS",
        "FIO2, ARTERIAL$FIO2, VENOUS",
    ]
    mask = static_df["COMPONENT_NAME"].isin(POSSIBLE_VEN_ART)
    static_df.loc[mask, "COMPONENT_NAME"] = static_df.loc[
        mask, ["COMPONENT_NAME", "DESCRIPTION"]
    ].apply(
        lambda x: venous_arterial_split(x["DESCRIPTION"], x["COMPONENT_NAME"]), axis=1
    )
    # for i, row in static_df.iterrows():
    #     if "$" not in row["COMPONENT_NAME"]:
    #         continue

    #     if "ven" in row["DESCRIPTION"].lower():
    #         static_df.loc[i, "COMPONENT_NAME"] = row["COMPONENT_NAME"].split("$")[-1]
    #     elif "art" in row["DESCRIPTION"].lower():
    #         static_df.loc[i, "COMPONENT_NAME"] = row["COMPONENT_NAME"].split("$")[0]
    #     else:
    #         static_df.loc[i, "COMPONENT_NAME"] = row["COMPONENT_NAME"].split("$")[0]

    # TODO: flag to filter by mapping on UCLA-CEDARS or UCLA-CEDARS-COntrol_UCLA or no filtering

    return static_df


def align_units(
    labs_df: DataFrame,
    raw_data_dir: str,
    unit_mapping_file: str = "unit_mappings.pkl",
) -> DataFrame:
    # Assumes lab results are numeric
    unit_converter = UnitRegistry()
    unit_converter.define("micro- = 1e-6 = u = mc")
    unit_converter.define("iu = u")
    unit_converter.define("mm3 = mm**3")
    unit_converter.define("eq = equivalent")

    if not isfile(join(raw_data_dir, unit_mapping_file)):
        raise Exception(
            "Unit mapping file is missing. Please run the `construct_lab_unit_mappings.py` script."
        )

    with open(join(raw_data_dir, unit_mapping_file), "rb") as f:
        mapping = load(f)  # map lab -> mode unit + count

    mask = labs_df["COMPONENT_NAME"].isin(mapping.keys())
    labs_df.loc[mask, "RESULTS"] = labs_df.loc[
        mask, ["RESULTS", "COMPONENT_NAME", "REFERENCE_UNIT"]
    ].apply(convert(mapping, unit_converter), axis=1)
    # Drop all rows with NaN results
    return labs_df[labs_df["RESULTS"].notna()]


def convert(
    mapping: Dict[str, Dict[str, Union[str, int]]], unit_converter: UnitRegistry
) -> Callable:
    def convert_sample(sample: Series) -> float:
        # mode unit name and count of that lab
        lab_name = sample["COMPONENT_NAME"]
        result = sample["RESULTS"]

        if lab_name not in mapping:  # no conversion
            return result

        lab_unit_info = mapping[lab_name]
        # this is hard to do for each separate or combined cohorts
        # ignore labs with fewer than 50 samples (we will drop nans)
        # if lab_unit_info["count"] < 50:
        # return nan

        unit = cleanup_unit(sample["REFERENCE_UNIT"])
        mode = cleanup_unit(lab_unit_info["mode_unit"])

        # no conversion if units are equivalent
        units_are_equivalent = any(
            mode in equiv_set and unit in equiv_set
            for equiv_set in MANUAL_EQUIVALENT_UNITS
        )

        if not units_are_equivalent:
            try:
                result, unit = convert_between_g_mol_eq(lab_name, result, unit, mode)
                result, unit, mode = convert_numerical_count_unit_values(
                    result, unit, mode
                )
                result *= unit_converter(unit).to(mode).magnitude
            except:  # anything we can't figure out how to convert will be dropped.
                return nan
        return result

    return convert_sample


def cleanup_unit(unit: str) -> str:
    # drop (calc) and creat
    unit = unit.replace("(calc)", "").replace("creat", "").replace("crt", "")
    # 24hrs -> day
    if "/24" in unit:
        return unit.split("/")[0] + "/d"
    return unit


def convert_between_g_mol_eq(
    lab_name: str, result: float, unit: str, mode: str
) -> Tuple[float, str]:
    # g <-> mol <-> eq
    need_valence = [
        ("eq", "mol") if ("mol" in mode or "g" in mode) and "eq" in unit else None,
        ("mol", "eq") if "eq" in mode and ("g" in unit or "mol" in unit) else None,
    ]
    need_molecular_weight = [
        ("mol", "g") if "g" in mode and ("mol" in unit or "eq" in unit) else None,
        ("g", "mol") if ("mol" in mode or "eq" in mode) and "g" in unit else None,
    ]
    # order matters for g <-> eq
    conversions = [(need_valence, MOL_TO_EQ)]
    if (any(need_valence) and any(need_molecular_weight)) and "g" in unit:
        # first convert g -> mol, then mol -> eq
        conversions.insert(0, (need_molecular_weight, G_TO_MOL))
    else:  # first convert eq -> mol, then mol -> g
        conversions.append((need_molecular_weight, G_TO_MOL))

    for directional_conversion, lookup_table in conversions:
        if any(directional_conversion):
            scale = next(
                iter([v for k, v in lookup_table.items() if k in lab_name]), None
            )
            if scale:
                # opposite direciton requires division
                if directional_conversion[0]:
                    result /= scale
                else:
                    result *= scale
                unit = unit.replace(*directional_conversion)  # e.g. mol -> g
    return (result, unit)


def convert_numerical_count_unit_values(
    result: float, unit: str, mode: str
) -> Tuple[float, str, str]:
    # pick longest match (e.g. I want thousand if both thous and thousand match)
    count_unit_convert_mode = sorted(
        [count for count in NUMERICAL_COUNT_UNIT_VALUES if count in mode],
        key=len,
        reverse=True,
    )
    count_unit_convert_other = sorted(
        [count for count in NUMERICAL_COUNT_UNIT_VALUES if count in unit],
        key=len,
        reverse=True,
    )
    to_convert_mode = next(iter(count_unit_convert_mode), "")
    to_convert_other = next(iter(count_unit_convert_other), "")
    unit = unit.replace(to_convert_other, "u") if to_convert_other else unit
    mode = mode.replace(to_convert_mode, "u") if to_convert_mode else mode
    # TODO:PP check division
    result *= NUMERICAL_COUNT_UNIT_VALUES.get(
        to_convert_mode, 1
    ) / NUMERICAL_COUNT_UNIT_VALUES.get(to_convert_other, 1)

    return (result, unit, mode)
