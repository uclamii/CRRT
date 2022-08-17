from typing import Union
from numpy import std, mean, sqrt, arcsin, ndarray
from pandas import DataFrame, Series
from scipy.stats import chi2_contingency


def hedges_g(
    dist_error: Union[ndarray, Series], dist_true: Union[ndarray, Series]
) -> float:
    """
    Corrected cohen's d if groups are unequal size, assumes population standard deviation is same for both groups.
    Ref: https://stackoverflow.com/a/33002123/1888794
    """
    n1 = len(dist_error)
    n2 = len(dist_true)
    dof = n1 + n2 - 2  # degrees of freedom
    return abs(
        (mean(dist_error) - mean(dist_true))
        / sqrt(
            (
                (n1 - 1) * std(dist_error, ddof=1) ** 2
                + (n2 - 1) * std(dist_true, ddof=1) ** 2
            )
            / dof
        )
    )


def cramers_corrected_stat(contingency: DataFrame) -> float:
    """
    Calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    Ref: https://stackoverflow.com/a/39266194/1888794
    """
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.to_numpy().sum()
    phi2 = chi2 / n
    r, k = contingency.shape

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    normalize = min((kcorr - 1), (rcorr - 1))
    # if denominator is 0 it's invalid, just return 0 (prevent warning)
    return sqrt(phi2corr / normalize) if normalize else 0


def cohens_h(contingency: DataFrame) -> float:
    """
    Assumes 2x2 confusion matrix/contingency table, (non-directional).
    Ref: https://en.wikipedia.org/wiki/Cohen%27s_h"""
    proportions = [  # Row = dist_error, col = dist_true. we compare prop of 0 class.
        contingency.apply(lambda row: row / row.sum(), axis=1).iloc[0, 0],
        contingency.apply(lambda col: col / col.sum(), axis=0).iloc[0, 0],
    ]
    transformed_proportions = [
        2 * arcsin(sqrt(proportion)) for proportion in proportions
    ]
    return abs(transformed_proportions[0] - transformed_proportions[1])
