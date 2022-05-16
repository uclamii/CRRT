from typing import List
from mlflow import log_text, log_metrics
from numpy import ndarray, unique, vstack
from pandas import Series, DataFrame
from sklearn.base import ClassifierMixin
import scipy.stats as st

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
    return st.shapiro(metric)[1] > (1 - confidence_level)


def test_fail_to_reject(p: float, confidence_level: float = 0.95) -> bool:
    """
    Tests if a and b come from the same distribution.
    p > alpha: a and b come from the same distribution (fail to reject H0)
        else: a and b come from different distributions (reject H0)
    """
    return p > (1 - confidence_level)


def model_randomness(
    data: ndarray,
    labels: ndarray,
    prefix: str,
    model: ClassifierMixin,
    columns: str,
    seed: int,
):
    """Tests for model randomness by comparing
    ("fn", "tp"), ("fn", "tn"), ("fp", "tn"), ("fp", "tp")
    If the FN and TP are statistically significantly different, it means FN looks like negatives even though they’re really positive, which means the model was following the stats and learned something reasonable because it doesn’t look like the positive class.
    That means maybe we need other variables or there’s noise in the feature data.
    But, if the FN looks like TP then the model is failing to learn something meaningful, and suggests that the model is more random.
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
        table[f"{comparison[0]}_vs_{comparison[1]}"] = {}
        for colidx, coln in enumerate(columns):
            table[coln] = {}
            table[coln]["test_used"] = []
            table[coln]["test_result"] = []
            # e.g. fn -> SBP (all rows)
            dist_error = subsets[error_type][:, colidx]
            dist_true = subsets[true_type][:, colidx]
            if len(dist_error) > 0:  # if there is any errors
                if is_ctn(dist_error):  # continuous / quantitative
                    if test_normality(dist_error):  # check normal
                        # Ref: https://www.statology.org/two-sample-t-test-python/
                        table[coln]["test_used"].append("t_ind")
                        table[coln]["test_result"].append(
                            test_fail_to_reject(
                                st.ttest_ind(dist_error, dist_true, random_state=seed)[1]
                            )
                        )
                    else:  # not normal
                        # Ref: https://www.statology.org/mann-whitney-u-test-python/
                        table[coln]["test_used"].append("mannwhitney_u")
                        table[coln]["test_result"].append(
                            test_fail_to_reject(st.mannwhitneyu(dist_error, dist_true)[1])
                        )
                else:  # categorical / qualitative
                    # rows: error vs true, cols: counts of occurrences for each category
                    categories, counts = unique(dist_error, return_counts=True)
                    error_counts = DataFrame(counts, index=categories, columns=[error_type])
                    categories, counts = unique(dist_true, return_counts=True)
                    true_counts = DataFrame(counts, index=categories, columns=[true_type])
                    contingency = error_counts.join(true_counts, how="outer").fillna(0).values
                    if contingency.shape[1] == 2:  # binary
                        # Ref: https://www.statology.org/fishers-exact-test-python/
                        table[coln]["test_used"].append("fisher_exact")
                        table[coln]["test_result"].append(
                            test_fail_to_reject(st.fisher_exact(contingency)[1])
                        )
                    else:
                        # although technically frequencies/counts must be > 5 there's no other python alternatives for multicategorical
                        table[coln]["test_used"].append("chi2")
                        table[coln]["test_result"].append(
                            test_fail_to_reject(st.chi2_contingency(contingency)[1])
                        )
    log_text(DataFrame(table).to_string(), f"{prefix}_dist_comparison_table.txt")
    # TODO: log the metrics separately?
        