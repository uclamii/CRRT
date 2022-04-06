import pandas as pd

from data.load import get_pt_type_indicators


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
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
    # drop columns with all nan values
    df = df[df.columns[~df.isna().all()]]
    # TODO: move this to get baked into the data?
    df = get_pt_type_indicators(df)

    return df
