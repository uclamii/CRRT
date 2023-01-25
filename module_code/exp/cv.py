from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    average_precision_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from scipy.stats import pearsonr
import numpy as np

from data.load import load_data

metrics = {
    "auroc": lambda gt, pred_probs, decision_thresh: roc_auc_score(gt, pred_probs),
    "ap": lambda gt, pred_probs, decision_thresh: average_precision_score(
        gt, pred_probs
    ),
    "brier": lambda gt, pred_probs, decision_thresh: brier_score_loss(gt, pred_probs),
    "accuracy": lambda gt, pred_probs, decision_thresh: accuracy_score(
        gt, (pred_probs >= decision_thresh).astype(int)
    ),
    "f1": lambda gt, pred_probs, decision_thresh: f1_score(
        gt, (pred_probs >= decision_thresh).astype(int)
    ),
    "recall": lambda gt, pred_probs, decision_thresh: recall_score(
        gt, (pred_probs >= decision_thresh).astype(int)
    ),
    "specificity": lambda gt, pred_probs, decision_thresh: recall_score(
        gt, (pred_probs >= decision_thresh).astype(int), pos_label=0
    ),
    "precision": lambda gt, pred_probs, decision_thresh: precision_score(
        gt, (pred_probs >= decision_thresh).astype(int)
    ),
    "conf_matrix": lambda gt, pred_probs, decision_thresh: confusion_matrix(
        gt, (pred_probs >= decision_thresh).astype(int)
    ),
    "TN": lambda gt, pred_probs, decision_thresh: confusion_matrix(
        gt, (pred_probs >= decision_thresh).astype(int)
    )[0, 0],
    "FN": lambda gt, pred_probs, decision_thresh: confusion_matrix(
        gt, (pred_probs >= decision_thresh).astype(int)
    )[1, 0],
    "TP": lambda gt, pred_probs, decision_thresh: confusion_matrix(
        gt, (pred_probs >= decision_thresh).astype(int)
    )[1, 1],
    "FP": lambda gt, pred_probs, decision_thresh: confusion_matrix(
        gt, (pred_probs >= decision_thresh).astype(int)
    )[0, 1],
}


algs = {
    "logistic": LogisticRegression,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "naive_bayes": MultinomialNB,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "lgb": LGBMClassifier,
    "xgb": XGBClassifier,
}


def run_cv(
    args,
    alg="logistic",
    corr_thresh=None,
    top_n_fts=None,
    alg_kwargs=None,
    decision_thresh=0.5,
    patient_id_col="IP_PATIENT_ID",
    target_col="recommend_crrt",
    eval_metrics=("ap", "auroc", "accuracy", "f1", "conf_matrix"),
    n_splits=10,
    random_state=0,
):
    """
    Runs cross-validation based on user-defined parameters.

    Parameters
    ----------
    input_df: pandas.DataFrame
    alg: str
    corr_thresh: float
    top_n_fts: int
    alg_kwargs: dict
    decision_thresh: float
    patient_id_col: str
    target_col: str
    eval_metrics: tuple
    n_splits: int
    random_state: int

    Returns
    -------
    tuple

    """
    input_df = load_data(args, args.train_val_cohort)

    # shuffle df
    input_df = shuffle(input_df, random_state=random_state)

    # convert features/targets to arrays and collect array indices for real features'
    cols = list(input_df.columns)
    patient_ids = input_df[patient_id_col].unique()

    # for patients with multiple targets, pick min target for stratified splitting
    patient_targets = [
        min(list(input_df.loc[input_df[patient_id_col] == pid][target_col]))
        for pid in patient_ids
    ]

    # keep track of feature columns and real/cat feature columns (for filling in missing values based on train data)
    feature_cols = [col for col in cols if col not in [target_col, patient_id_col]]
    real_cols = [
        col
        for col in cols
        if ("VITAL_SIGN" in col)
        or ("RESULT" in col)
        or ("CPT_SECTION" in col)
        or ("pr_CCS_COD" in col)
    ]
    real_cols_indices = [i for i, col in enumerate(feature_cols) if col in real_cols]
    cat_cols_indices = [i for i, col in enumerate(feature_cols) if col not in real_cols]

    # cross-validation script
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metric_scores = {metric: [] for metric in eval_metrics}
    fold_models = []
    used_features = []
    for i, (train_index, test_index) in enumerate(
        skf.split(patient_ids, patient_targets)
    ):
        print("Start fold: {}".format(i))

        # get corresponding df/array from patient ids
        train_patient_ids = patient_ids[train_index]
        test_patient_ids = patient_ids[test_index]
        train_df = input_df.loc[input_df[patient_id_col].isin(train_patient_ids)]
        test_df = input_df.loc[input_df[patient_id_col].isin(test_patient_ids)]

        X_train, X_test = (
            train_df[feature_cols].to_numpy(),
            test_df[feature_cols].to_numpy(),
        )
        y_train, y_test = (
            train_df[target_col].to_numpy(),
            test_df[target_col].to_numpy(),
        )
        # fill missing values as 0 for cat features
        X_train_cat = X_train[:, cat_cols_indices]
        nan_train_inds = np.where(np.isnan(X_train_cat))
        X_train_cat[nan_train_inds] = 0
        # any columns with all zero are set to 0
        nan_train_inds_ = np.where(np.isnan(X_train_cat))
        X_train_cat[nan_train_inds_] = 0
        X_train[:, cat_cols_indices] = X_train_cat

        # do the same for test
        X_test_cat = X_test[:, cat_cols_indices]
        nan_test_inds = np.where(np.isnan(X_test_cat))
        X_test_cat[nan_test_inds] = 0
        # any columns with all zero from train are set to 0 in test
        nan_test_inds_ = np.where(np.isnan(X_test_cat))
        X_test_cat[nan_test_inds_] = 0
        X_test[:, cat_cols_indices] = X_test_cat

        # fill missing values with mean for real features. get fill values from train, fill train/test
        # from https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
        X_train_real = X_train[:, real_cols_indices]
        real_fill_values = np.nanmean(X_train_real, axis=0)
        nan_train_inds = np.where(np.isnan(X_train_real))
        X_train_real[nan_train_inds] = np.take(real_fill_values, nan_train_inds[1])
        # any columns with all zero are set to 0
        nan_train_inds_ = np.where(np.isnan(X_train_real))
        X_train_real[nan_train_inds_] = 0
        X_train[:, real_cols_indices] = X_train_real

        # do the same for test
        X_test_real = X_test[:, real_cols_indices]
        nan_test_inds = np.where(np.isnan(X_test_real))
        X_test_real[nan_test_inds] = np.take(real_fill_values, nan_test_inds[1])
        # any columns with all zero from train are set to 0 in test
        nan_test_inds_ = np.where(np.isnan(X_test_real))
        X_test_real[nan_test_inds_] = 0
        X_test[:, real_cols_indices] = X_test_real

        print(
            "Array shapes - X_train: {}, X_test: {}, y_train: {}, y_test: {}".format(
                X_train.shape, X_test.shape, y_train.shape, y_test.shape
            )
        )

        # get feature mask  for features with target correlation above threshold
        feature_corrs = np.array(
            [pearsonr(X_train[:, col], y_train)[0] for col in range(X_train.shape[1])]
        )
        # any features with constant input are defined as 0 correlation
        nan_features = np.where(np.isnan(feature_corrs))
        feature_corrs[nan_features] = 0
        # get absolute value of correlations and get feature mask
        feature_corrs = np.abs(feature_corrs)
        if isinstance(corr_thresh, float) and (0 <= corr_thresh < 1):
            feature_mask = feature_corrs >= corr_thresh
            feature_names = [
                feature_name
                for feature_mask_val, feature_name in zip(feature_mask, feature_cols)
                if feature_mask_val
            ]
            used_features.append(feature_names)
        elif isinstance(top_n_fts, int) and (top_n_fts > 0):
            feature_mask = feature_corrs.argsort()[-1 * top_n_fts :]
            feature_names = [feature_cols[i] for i in feature_mask]
            used_features.append(feature_names)
        else:
            raise ValueError(
                "You need to define a correlation threshold or number of features!"
            )

        X_train = X_train[:, feature_mask]
        X_test = X_test[:, feature_mask]
        print(
            "(After filtering features) Array shapes - X_train: {}, X_test: {}".format(
                X_train.shape, X_test.shape
            )
        )
        print("Fitting model")
        if alg_kwargs:
            clf = algs[alg](**alg_kwargs)
        else:
            clf = algs[alg]()
        clf.fit(X_train, y_train)
        print("Predicting on val set")
        pred_probs = clf.predict_proba(X_test)[:, 1]

        print("Calculating Metrics")
        for metric in metric_scores.keys():
            metric_fn = metrics[metric]
            metric_score = metric_fn(y_test, pred_probs, decision_thresh)
            metric_scores[metric].append(metric_score)

        fold_models.append(clf)

    print("Done!")
    mean_metric_scores = {}
    std_metric_scores = {}
    for metric in eval_metrics:
        scores = np.array(metric_scores[metric])
        print(
            "{}: mean: {}+/-{}".format(
                metric,
                np.round(np.mean(scores, axis=0), 4),
                np.round(np.std(scores, axis=0), 4),
            )
        )
        mean_metric_scores[metric + "_mean"] = np.round(np.mean(scores, axis=0), 4)
        std_metric_scores[metric + "_std"] = np.round(np.std(scores, axis=0), 4)

    return {
        "fold_scores": metric_scores,
        "mean_scores": mean_metric_scores,
        "std_scores": std_metric_scores,
        "fold_models": fold_models,
        "fold_features": used_features,
    }
