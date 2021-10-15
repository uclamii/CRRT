import pandas as pd
from sktime import ROCKETClassifier, HIVECOTEV1


def continuous_learning(
    df: pd.DataFrame, modeln: str, seed: int, outcome_coln: str = "recommend_crrt"
):
    # https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.classification.hybrid.HIVECOTEV1.html
    if modeln == "hivecote":
        model = HIVECOTEV1(n_jobs=-1, random_state=seed)
    # https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.classification.kernel_based.ROCKETClassifier.html
    elif modeln == "rocket":
        model = ROCKETClassifier(n_jobs=-1, random_state=seed)
    elif modln == "lstm":
        model = LSTM()
