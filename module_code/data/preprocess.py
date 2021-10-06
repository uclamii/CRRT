import pandas as pd


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
        "Palliative Care",
        "Expired ",
    ]
    df = df.drop(drop_columns, axis=1)
    # drop columns with all nan values
    df = df[df.columns[~df.isna().all()]]

    return df
