"""
Any final preprocessing of the already processed dataframe
Used in load.py
"""

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
        "CPT_SECTION_na",
        "pr_CCS_CODE_na",
        "dx_CCS_CODE_na",
        # Unwanted labs
        "ISSUE DATE_min",
        "ISSUE DATE_max",
        "ISSUE DATE_mean",
        "ISSUE DATE_std",
        "ISSUE DATE_skew",
        "ISSUE DATE_len",
        "UNITS ORDERED_min",
        "UNITS ORDERED_max",
        "UNITS ORDERED_mean",
        "UNITS ORDERED_std",
        "UNITS ORDERED_skew",
        "UNITS ORDERED_len",
        "BLOOD PRODUCT EXPIRATION_min",
        "BLOOD PRODUCT EXPIRATION_max",
        "BLOOD PRODUCT EXPIRATION_mean",
        "BLOOD PRODUCT EXPIRATION_std",
        "BLOOD PRODUCT EXPIRATION_skew",
        "BLOOD PRODUCT EXPIRATION_len",
        "ISSUING PHYSICIAN_min",
        "ISSUING PHYSICIAN_max",
        "ISSUING PHYSICIAN_mean",
        "ISSUING PHYSICIAN_std",
        "ISSUING PHYSICIAN_skew",
        "ISSUING PHYSICIAN_len",
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

    # drop high missingness
    if args.drop_percent is not None:
        orig_columns = df.columns

        # For recordkeeping
        before_filter_unique_features = (
            df.columns.str.replace(".*_indicator", "indicator", regex=True)
            .str.replace("RACE.*", "RACE", regex=True)
            # tobacco/smoking/allergen aren't currenly in the intersection of features for all 3 cohorts, i've included it here just in case
            .str.replace(".*TOBACCO_USER.*", "TOBACCO_USER", regex=True)
            .str.replace(".*SMOKING_TOB_STATUS.*", "SMOKING_TOB_STATUS", regex=True)
            .str.replace(".*ALLERGEN_ID.*", "ALLERGEN_ID", regex=True)
            .str.replace("_(mean|min|max|std|skew|len)", "", regex=True)
            .unique()
        )

        nulls = df.isnull().sum().to_frame("count_missing")
        nulls["percent_missing"] = nulls["count_missing"] / len(df) * 100
        keep_list = nulls[nulls["percent_missing"] < args.drop_percent].index
        print(f"Keeping {len(keep_list)} columns from {len(nulls)}")
        df = df.loc[:, keep_list]

        # For recordkeeping
        after_filter_unique_features = (
            df.columns.str.replace(".*_indicator", "indicator", regex=True)
            .str.replace("RACE.*", "RACE", regex=True)
            # tobacco/smoking/allergen aren't currenly in the intersection of features for all 3 cohorts, i've included it here just in case
            .str.replace(".*TOBACCO_USER.*", "TOBACCO_USER", regex=True)
            .str.replace(".*SMOKING_TOB_STATUS.*", "SMOKING_TOB_STATUS", regex=True)
            .str.replace(".*ALLERGEN_ID.*", "ALLERGEN_ID", regex=True)
            .str.replace("_(mean|min|max|std|skew|len)", "", regex=True)
            .unique()
        )
        print(
            f"Keeping {len(after_filter_unique_features)} raw features from {len(before_filter_unique_features)}"
        )

    return df, orig_columns
