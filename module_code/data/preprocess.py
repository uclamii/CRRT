from argparse import Namespace
import pandas as pd

from data.utils import get_pt_type_indicators


# TODO: this probably shouldn't be its own separate module. Maybe move it into sklearn_loaders?
def adhoc_preprocess_data(df: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    """Pre-processes the data for use by ML model (Adhoc)."""

    # for sliding window analysis, we want to only keep patients with <= max_slide number of days on CRRT
    if args.max_days_on_crrt is not None:
        df = df[
            (df["CRRT Total Days"] <= args.max_days_on_crrt)
            & (df["CRRT Total Days"] >= args.min_days_on_crrt)
        ]

    drop_columns = [
        "Month",
        "Hospital name",
        "CRRT Total Days",  # this will not exist for incoming new patients.
        "End Date",
        "Machine",
        "ICU",
        "Recov. renal funct.",
        "Transitioned to HD",
        "Comfort Care",
        "Expired ",
        "KNOWN_DECEASED",
        "CRRT Year",
    ]
    # Ignore errors: it's ok if these don't exist (e.g. in ucla: control)
    df = df.drop(drop_columns, axis=1, errors="ignore")
    # Get rid of "Unnamed" Column
    df = df.drop(df.columns[df.columns.str.contains("^Unnamed")], axis=1)
    # drop columns with all nan values
    df = df[df.columns[~df.isna().all()]]
    # TODO: move this to get baked into the data?
    df = get_pt_type_indicators(df)

    # Exclude pediatric data, adults considered 21+
    is_adult_mask = df["AGE"] >= 21
    df = df[~is_adult_mask] if args.patient_age == "peds" else df[is_adult_mask]

    return df
