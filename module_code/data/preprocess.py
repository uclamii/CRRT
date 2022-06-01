from argparse import Namespace
import pandas as pd

from data.utils import get_pt_type_indicators


def preprocess_data(df: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    """Pre-processes the data for use by ML model."""

    drop_columns = [
        "Month",
        "Hospital name",
        "CRRT Total Days",
        "End Date",
        "Machine",
        "ICU",
        "Recov. renal funct.",
        "Transitioned to HD",
        "Comfort Care",
        "Expired ",
    ]
    df = df.drop(drop_columns, axis=1)
    # Get rid of "Unnamed" Column
    df = df.drop(df.columns[df.columns.str.contains("^Unnamed")], axis=1)
    # drop columns with all nan values
    df = df[df.columns[~df.isna().all()]]
    # TODO: move this to get baked into the data?
    df = get_pt_type_indicators(df)

    # Exclude pediatric data, adults considered 21+
    is_adult_mask = df["Age"] >= 21
    df = df[~is_adult_mask] if args.patient_age == "peds" else df[is_adult_mask]

    return df
