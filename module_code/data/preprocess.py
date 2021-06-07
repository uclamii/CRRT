import pandas as pd


def preprocess_data(df):
    """
    Pre-processes the data for use by ML model

    Parameters
    ----------
    df: pandas.DataFrame
        input DataFrame

    Returns
    -------
        pandas.DataFrame

    """
    # features not currently included: Month, Hospital name, CRRT Total Days, End Date, Machine, ICU
    columns = list(df.columns)
    id_col = "IP_PATIENT_ID"
    cat_features = [
        "GENDER",
        "RACE",
        "ETHNICITY",
        "PCP_PROVIDER_TYPE",
        "TOBACCO_USER",
        "CIGARETTES_YN",
        "SMOKING_TOB_STATUS",
        "ALCOHOL_USER",
        "IV_DRUG_USER_YN",
        "ALLERGEN_ID",
    ]
    real_features = [
        "AGE",
        "TOBACCO_PAK_PER_DY",
        "TOBACCO_USED_YEARS",
        "ALCOHOL_OZ_PER_WK",
        "ILLICIT_DRUG_FREQ",
    ] + [col for col in columns if ("VITAL_SIGN" in col) or ("RESULT" in col)]
    counts_features = [
        col for col in columns if ("CCS_CODE" in col) or ("CPT_SECTION" in col)
    ]
    target = ["recommend_crrt"]
    select_cols = [id_col] + cat_features + real_features + counts_features + target
    df = df[select_cols]

    df = fill_missing_values(df, cat_features, real_features, counts_features)
    # needs to happen after filling missing or else you'll get invalid entry nan
    df = convert_to_numeric(df, real_features)

    # drop columns with all nan values
    df = df[df.columns[~df.isna().all()]]

    df = one_hot_encode(df, cat_features)

    return df


def convert_to_numeric(df, real_features):
    """

    Parameters
    ----------
    df
    real_features: list(str)

    Returns
    -------
        pandas.DataFrame
    """
    if "ALCOHOL_OZ_PER_WK" in real_features:
        df["ALCOHOL_OZ_PER_WK"] = df["ALCOHOL_OZ_PER_WK"].apply(alc_freq_to_numeric)
    return df


def alc_freq_to_numeric(x):
    if x == "0":
        return 0
    if x == "3.6 - 4.2":
        return 3.9
    if x == ".6":
        return 0.6
    if x == "3.6":
        return 3.6
    if x == "1.8 - 3":
        return 2.4
    if x == "1.8":
        return 1.8
    if x == "2.4":
        return 2.4
    if x == "6":
        return 6
    if x == "8.4":
        return 8.4
    if x == ".6 - 1.2":
        return 0.8
    if x == "12.6":
        return 0.0
    if x is 0:
        return 0
    if x is None:
        return 0
    else:
        raise ValueError("Invalid entry: {}".format(x))


def fill_missing_values(df, cat_features, real_features, counts_features):
    """

    Parameters
    ----------
    df
    cat_features: list(str)
    real_features: list(str)
    counts_features: list(str)

    Returns
    -------
        pandas.DataFrame
    """

    # if CCS code, CPT_SECTION, or any counts feature and missing, put 0
    # if real features (like results), (generally) convert missing to nan.
    # in the training vector, convert to mean of train column
    # special static features included

    if "TOBACCO_USER" in cat_features:
        df["TOBACCO_USER"].fillna("Never", inplace=True)
    if "ALCOHOL_USER" in cat_features:
        df["ALCOHOL_USER"].fillna("No", inplace=True)
    if "PCP_PROVIDER_TYPE" in cat_features:
        df["PCP_PROVIDER_TYPE"].fillna("Physician", inplace=True)

    if "TOBACCO_PAK_PER_DY" in real_features:
        df["TOBACCO_PAK_PER_DY"].fillna(0, inplace=True)
    if "TOBACCO_USED_YEARS" in real_features:
        df["TOBACCO_USED_YEARS"].fillna(0, inplace=True)
    if "ALCOHOL_OZ_PER_WK" in real_features:
        df["ALCOHOL_OZ_PER_WK"].fillna(0, inplace=True)
    if "ILLICIT_DRUG_FREQ" in real_features:
        df["ILLICIT_DRUG_FREQ"].fillna(0, inplace=True)

    for ft in counts_features:
        df[ft].fillna(0, inplace=True)

    return df


def one_hot_encode(df, cat_features):
    """

    Parameters
    ----------
    df
    cat_features: list(str)

    Returns
    -------
        pandas.DataFrame
    """
    for column in cat_features:
        temp_df = pd.get_dummies(df[column], prefix=column)
        df = pd.merge(left=df, right=temp_df, left_index=True, right_index=True,)
        df = df.drop(columns=column)
    return df

