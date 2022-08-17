from typing import List, Tuple, Union
from mlflow import log_text
from numpy import ndarray, unique
from pandas import Series, DataFrame, crosstab
from sklearn.base import ClassifierMixin
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, fisher_exact, shapiro

from evaluate.effect_size_metrics import cohens_h, cramers_corrected_stat, hedges_g

filter_fns = {
    "tp": lambda data, labels: (data == labels) & (labels == 1),
    "tn": lambda data, labels: (data == labels) & (labels == 0),
    "fp": lambda data, labels: (data != labels) & (data == 1),
    "fn": lambda data, labels: (data != labels) & (data == 0),
}


def is_ctn(
    feature_values: ndarray, threshold: float = 0.05, long_tailed: bool = False
) -> bool:
    # Ref: https://stackoverflow.com/a/35827646/1888794
    feature_values = Series(feature_values)
    if long_tailed:
        return (
            1.0 * feature_values.value_counts(normalize=True).head(5).sum()
            < 1 - threshold
        )
    return 1.0 * feature_values.nunique() / feature_values.count() > 0.05


def test_normality(metric: List[float], confidence_level: float = 0.95) -> bool:
    """
    Tests for normality using the shapiro-wilk test.
    Outputs W,p. We want p > (alpha = 1 - CI = .05 usually)
    Ref: https://machinelearningmastery.com/
        a-gentle-introduction-to-normality-tests-in-python/
    p > alpha: Sample looks Gaussian (fail to reject H0)
         else: Sample does not look Gaussian (reject H0)
    """
    return shapiro(metric)[1] > (1 - confidence_level)


def test_reject(p: float, confidence_level: float = 0.95) -> bool:
    """
    Tests if a and b come from the same distribution.
    p > alpha: a and b come from the same distribution (fail to reject H0)
        else: a and b come from different distributions (reject H0)
    """
    return p <= (1 - confidence_level)


def get_contingency(
    dist_error: Union[ndarray, Series],
    dist_true: Union[ndarray, Series],
    error_type: str,
    true_type: str,
) -> DataFrame:
    if len(dist_error) == len(dist_true):
        return crosstab(dist_error, dist_true)
    # gets contingency if uneven
    # rows: error vs true, cols: counts of occurrences for each category
    categories, counts = unique(dist_error, return_counts=True)
    error_counts = DataFrame(counts, index=categories, columns=[error_type])
    categories, counts = unique(dist_true, return_counts=True)
    true_counts = DataFrame(counts, index=categories, columns=[true_type])
    return error_counts.join(true_counts, how="outer").fillna(0)


def get_pvalue_and_effect_size(
    dist_error: Union[ndarray, Series],
    dist_true: Union[ndarray, Series],
    error_type: str,
    true_type: str,
    seed: int,
) -> Tuple[str, float, str, float]:
    # if there is any errors, skips anything less than 3 samples
    if len(dist_error) > 2:
        if is_ctn(dist_error):  # continuous / quantitative
            effect_size = hedges_g(dist_error, dist_true)
            effect_size_name = "hedges_g"
            # check normal (test rquires at least 3 data points)
            if test_normality(dist_error):
                # Ref: https://www.statology.org/two-sample-t-test-python/
                return (
                    "t_ind",
                    test_reject(ttest_ind(dist_error, dist_true, random_state=seed)[1]),
                    effect_size_name,
                    effect_size,
                )
            # NOT NORMAL
            # Ref: https://www.statology.org/mann-whitney-u-test-python/
            return (
                "mannwhitney_u",
                test_reject(mannwhitneyu(dist_error, dist_true)[1]),
                effect_size_name,
                effect_size,
            )
        # categorical / qualitative
        contingency = get_contingency(dist_error, dist_true, error_type, true_type)
        if contingency.shape == (2, 2):  # BINARY: expect 2x2
            # Ref: https://www.statology.org/fishers-exact-test-python/
            return (
                "fisher_exact",
                test_reject(fisher_exact(contingency)[1]),
                "cohens_h",
                cohens_h(contingency),
            )
        # MULTICATEGORICAL
        # although technically frequencies/counts must be > 5 there's no other python alternatives for multicategorical
        return (
            "chi2",
            test_reject(chi2_contingency(contingency)[1]),
            "cramers_v",
            cramers_corrected_stat(contingency),
        )
    # cannot do the test
    return (None, None, None, None)


def model_randomness(
    data: ndarray,
    labels: ndarray,
    prefix: str,
    model: ClassifierMixin,
    columns: List[str],
    categorical_columns: List[str],
    seed: int,
):
    """Tests for model randomness by comparing
    ("fn", "tp"), ("fn", "tn"), ("fp", "tn"), ("fp", "tp")
    If the FN and TP are statistically significantly different, it means FN looks like negatives even though they’re really positive, which means the model was following the stats and learned something reasonable because it doesn’t look like the positive class.
    That means maybe we need other variables or there’s noise in the feature data.
    But, if the FN looks like TP then the model is failing to learn something meaningful, and suggests that the model is more random.
    However, just statistical tests are not enough, we also need to report effect sizes.
    """
    preds = model.predict(data)
    subsets = {k: data[v(preds, labels)] for k, v in filter_fns.items()}
    # we can assume independence between these because
    # A) mutually exclusive groups
    # B) sampling without replacement
    comparisons = [("fn", "tp"), ("fn", "tn"), ("fp", "tn"), ("fp", "tp")]
    table = {}
    for comparison in comparisons:
        error_type, true_type = comparison  # unpack
        # for each feature
        comparison_name = f"{comparison[0]}_vs_{comparison[1]}"
        table = {
            "Test Statistic": {},
            "Reject H0": {},
            "Measure Name": {},
            "Effect Size": {},
        }
        for colidx, coln in enumerate(columns):
            # e.g. fn_vs_tp -> SBP (all rows)
            dist_error = subsets[error_type][:, colidx]
            dist_true = subsets[true_type][:, colidx]
            (
                stat_name,
                reject_h0,
                effect_size_name,
                effect_size,
            ) = get_pvalue_and_effect_size(
                dist_error, dist_true, error_type, true_type, seed
            )
            table["Test Statistic"][(comparison_name, coln)] = stat_name
            table["Reject H0"][(comparison_name, coln)] = reject_h0
            table["Measure Name"][(comparison_name, coln)] = effect_size_name
            table["Effect Size"][(comparison_name, coln)] = effect_size
    # To read: read_csv("tmp.txt", index_col=[0,1])
    log_text(
        DataFrame(table).sort_values("Effect Size", ascending=False).to_csv(),
        f"{prefix}_dist_comparison_table.csv",
    )
