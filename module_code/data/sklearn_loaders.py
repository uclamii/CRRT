from argparse import ArgumentParser, Namespace
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# from sktime.transformations.series.impute import Imputer
# explicitly require this experimental feature
# from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler

# Local
from data.longitudinal_features import CATEGORICAL_COL_REGEX
from data.base_loaders import AbstractCRRTDataModule, DataLabelTuple
from data.utils import Preselected, SelectThreshold, f_pearsonr
from data.load import load_data

ADDITIONAL_CATEGORICAL_COLS = [
    "SEX",
    "ETHNICITY",
    "surgery_indicator",
    "liver_pt_indicator",
    "heart_pt_indicator",
    "infection_pt_indicator",
]


class SklearnCRRTDataModule(AbstractCRRTDataModule):
    def __init__(
        self,
        seed: int,
        outcome_col_name: str,
        train_val_cohort: str,
        eval_cohort: str,
        # val comes from train := (1 - test_split_size) * val_split_size
        val_split_size: float,
        # not necessary if eval_cohort != train_val_cohort.
        test_split_size: float = None,
        kbest: int = None,
        corr_thresh: float = None,
        impute_method: str = "simple",
        filters: Dict[str, Callable] = None,
    ):
        super().__init__()
        self.seed = seed
        self.train_val_cohort = train_val_cohort
        self.eval_cohort = eval_cohort
        self.outcome_col_name = outcome_col_name
        if test_split_size is None:
            assert (
                self.eval_cohort != self.train_val_cohort
            ), "You must specify a test_split_size is if the eval_cohort is the same as train_val_cohort."
        self.test_split_size = test_split_size
        self.val_split_size = val_split_size
        self.kbest = kbest
        self.corr_thresh = corr_thresh
        self.filters = filters
        self.impute_method = impute_method
        if self.impute_method == "simple":
            # the empty features will be dropped in feature selection.
            self.imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        # TODO: allow different neighbors to be passed in to be tuned
        elif self.impute_method == "knn":
            self.imputer = KNNImputer(keep_empty_features=True)

    def setup(
        self,
        args: Namespace,
        stage: Optional[str] = None,
        reference_ids: Optional[Dict[str, pd.Index]] = None,
        reference_cols: Optional[Union[List[str], pd.Index]] = None,
        data_transform: Optional[Callable[..., np.ndarray]] = None,
    ):
        """
        Ops performed across GPUs. e.g. splits, transforms, etc.
        """
        X, y = self.load_data_and_additional_preproc(
            args, self.train_val_cohort, reference_cols
        )

        # its the same for all the sequences, just take one
        # y = y.groupby("IP_PATIENT_ID").last()

        # Load other dataset if applicable and align columns
        if self.train_val_cohort != self.eval_cohort:
            X_eval, y_eval = self.load_data_and_additional_preproc(
                args, self.eval_cohort, reference_cols
            )
            # combine columns (outer join)
            all_columns = X.columns.union(X_eval.columns)
            # make sure columns from both are all available (but they're all missing)
            # the missing values will be simple imputed (ctn) and 0 imputed (nan)
            # NOTE: if keep_empty_features for our imputers in the post split transform
            #   pipeline is False, then the fully empty continuous features will be dropped.
            # by the serialized transform function
            # ref: https://stackoverflow.com/a/30943503/1888794
            X = X.reindex(columns=all_columns)
            X_eval = X_eval.reindex(columns=all_columns)

        #### Columns ####
        """
        Needs to come after:
          reference cols are potentially set
          potentially aligning columns with the eval dataset
        Needs to come before:
          feature selection for sliding window analysis
        """
        self.columns = X.columns
        self.categorical_columns = X.filter(
            regex=CATEGORICAL_COL_REGEX, axis=1
        ).columns.union(ADDITIONAL_CATEGORICAL_COLS)
        # set this here instead of init so that outcome col isn't included
        self.ctn_columns = X.columns.difference(self.categorical_columns)

        split_args = [X, y, reference_ids]
        if self.train_val_cohort != self.eval_cohort:
            split_args += [X_eval, y_eval]
        X_y_tuples = self.split_dataset(*split_args)
        split_names = list(X_y_tuples.keys())

        # Apply filters for subpopulation analysis later
        # MUST OCCUR BEFORE TRANSFORM (before feature selection)
        if self.filters:
            # We set up filters ahead of time so that we don't have to worry about feature selection
            for split in split_names:
                X = X_y_tuples[split][0]
                filters = {
                    k: self.get_filter(X, *args)
                    for groupname, filter in self.filters.items()
                    for k, args in filter.items()
                }
                setattr(self, f"{split}_filters", filters)

        # fit pipeline on train, call transform in get_item of dataset
        self.data_transform = (
            data_transform
            if data_transform is not None
            else self.get_post_split_transform(X_y_tuples["train"])
        )

        # set self.train, self.val, self.test
        for split in split_names:
            X, y = X_y_tuples[split]
            setattr(self, split, (self.data_transform(X), y))

    def load_data_and_additional_preproc(
        self, args: Namespace, cohort: str, reference_cols=None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Wrapper function for loading multiple cohorts as one dataset"""

        # convention for multiple cohorts to be split by a '+' symbol
        cohort = cohort.split("+")
        all_columns = pd.Index([])

        X_all_cohorts = []
        y_all_cohorts = []

        for single_cohort in cohort:
            X, y = self.single_cohort_load_data_and_additional_preproc(
                args, single_cohort, reference_cols
            )

            # combine columns (outer join)
            all_columns = all_columns.union(X.columns)

            X_all_cohorts.append(X)
            y_all_cohorts.append(y)

        X_all_cohorts = [X.reindex(columns=all_columns) for X in X_all_cohorts]
        X_all_cohorts = pd.concat(X_all_cohorts)
        y_all_cohorts = pd.concat(y_all_cohorts)
        return (X_all_cohorts, y_all_cohorts)

    def single_cohort_load_data_and_additional_preproc(
        self, args: Namespace, cohort: str, reference_cols=None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        preprocessed_df = load_data(args, cohort)
        X, y = (
            preprocessed_df.drop(self.outcome_col_name, axis=1),
            preprocessed_df[self.outcome_col_name],
        )

        # remove unwanted columns, esp non-numeric ones, before pad and pack
        X = X.select_dtypes(["number"])

        # add reference cols if not none
        if reference_cols is not None:
            # make sure columns in original exist here (but they're all missing)
            # the missing values will be simple imputed (ctn) and 0 imputed (nan)
            # by the serialized transform function
            # ref: https://stackoverflow.com/a/30943503/1888794
            X = X.reindex(columns=reference_cols)
            # drop cols in X but not in reference
            X = X.drop(X.columns.difference(reference_cols), axis=1)
        return (X, y)

    @classmethod
    @staticmethod
    def get_filter(
        df: pd.DataFrame,
        cols: Union[str, List[str]],
        vals: Union[Any, List[Union[int, str, Tuple[int, int]]]],
        combine: str = "AND",
    ) -> pd.Series:
        def apply_check(col: str, val: Union[int, str, Tuple[int, int]]) -> pd.Series:
            if isinstance(val, tuple):  # assumes [a, b)
                assert len(val) == 2, "Tuple passed isn't length two [a, b)."
                return (df[cols] >= val[0]) & (df[cols] < val[1])
            return df[col] == val

        if isinstance(cols, list):
            assert isinstance(vals, list) and len(vals) == len(
                cols
            ), "cols don't match vals for getting filters."
            lambda_fns = map(lambda col_val: apply_check(*col_val), zip(cols, vals))
            # roll up the functions
            if combine == "AND":
                return reduce(lambda f1, f2: f1 & f2, lambda_fns)
            else:  # OR
                return reduce(lambda f1, f2: f1 | f2, lambda_fns)

        return apply_check(cols, vals)

    def get_post_split_transform(
        self, train: DataLabelTuple, reference_cols_mask: Optional[List[bool]] = None
    ) -> Callable:
        """
        The serialized preprocessed df should alreayd have dealth with categorical variables and aggregated them as counts, so we only deal with numeric / continuous variables.
        """
        pipeline = Pipeline(
            [
                # impute for continuous columns
                (
                    "ctn-fillna",
                    ColumnTransformer(
                        [  # (name, transformer, columns) tuples
                            (
                                f"{self.impute_method}-impute",
                                self.imputer,
                                # TODO convert to indices
                                self.ctn_columns,
                            )
                        ],
                        remainder="passthrough",
                    ),
                ),
                # zero out everything else
                (f"0-fill-cat", SimpleImputer(strategy="constant", fill_value=0)),
                # standardize data (need to do it in a way that either preserves pandas or doesn't need it for ctn-fillna.)
                ("scale", MinMaxScaler()),
                # feature-selection doesn't allow NaNs in the data, make sure to impute first.
                ("feature-selection", self.get_feature_selection(reference_cols_mask)),
            ]
        )

        pipeline.fit(*train)
        return pipeline.transform

    def get_feature_selection(
        self, reference_cols_mask: Optional[List[bool]] = None
    ) -> Callable:
        """
        Fit the feature selection transform based on either the k best features or the number of features
        above a correlation threshold. Passthrough option also available.
        """
        # TODO: Test these work as intended
        if reference_cols_mask is not None:
            return Preselected(support_mask=reference_cols_mask)
        if self.kbest and self.corr_thresh:
            raise ValueError("Both kbest and corr_thresh are not None")
        if self.kbest:
            # TODO: maybe want to update sklearn and use https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.r_regression.html#sklearn.feature_selection.r_regression
            return SelectKBest(f_pearsonr, k=self.kbest)
        elif self.corr_thresh:
            return SelectThreshold(f_pearsonr, threshold=self.corr_thresh)
        # passthrough transform
        return SelectKBest(lambda X, y: np.zeros(X.shape[1]), k="all")

    def stratify_groupby_split(self, X, y, n_splits):
        # TODO TEST
        groups = X["IP_PATIENT_ID"]
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=self.seed
        )
        for _, (train_idxs, test_idxs) in enumerate(cv.split(X, y, groups)):
            # We only need one split
            break
        return train_idxs, test_idxs
        """
        # Use:
        train_val_idxs, test_idxs = stratify_groupby_split(X, y, n_splits=5)  # 1/5 = 20%
        train_val_X = X.loc[train_val_idxs].reset_index(drop=True).copy()
        train_val_y = y.loc[train_val_idxs].reset_index(drop=True).copy()
        train_idxs, val_idxs = stratify_groupby_split(train_val_X, train_val_y, n_splits=5)
        """

    def split_dataset(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        reference_ids: Optional[Dict[str, pd.Index]] = None,
        X_eval: Optional[pd.DataFrame] = None,
        y_eval: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Dict[str, DataLabelTuple]:
        """
        Splitting with stratification using sklearn.
        We then convert to Dataset so the Dataloaders can use that.

        If train_val_cohort != eval_cohort then the whole test split will be from the test_cohort loaded data.
        Reference IDs will be from there as well.
        """
        separate_eval_dataset = X_eval is not None and y_eval is not None

        # sample = [pt, treatment]
        # ensure data is split by patient
        def pick_unique_pt_ids(data: pd.DataFrame) -> np.ndarray:
            # If aggregating over a time-interval, a new column "DATE" is introduced
            to_drop = ["Start Date"]
            if "DATE" in data.index.names:
                to_drop.append("DATE")
            return data.index.droplevel(to_drop).unique().values

        sample_ids = {"train_val": pick_unique_pt_ids(X)}
        if separate_eval_dataset:
            sample_ids["eval"] = pick_unique_pt_ids(X_eval)

        if reference_ids is not None:  # filter id to the serialized ones
            # There are patients we don't want include:
            # there may be fewer patients in this dataset (D_{+i}) than original because they don't have enough data after the sliding window
            # or there may be extraneous ones that wouldn't have enough data for the window without a slide, but now do, that we don't want to include
            # Solve this by inner joining
            train_ids = reference_ids["train"].join(
                sample_ids["train_val"], how="inner"
            )
            val_ids = reference_ids["val"].join(sample_ids["train_val"], how="inner")
            # test_ids will separately come from eval_cohort if it was different
            test_ids_key = "train_val" if not separate_eval_dataset else "eval"
            test_ids = reference_ids["test"].join(sample_ids[test_ids_key], how="inner")
        else:
            # patient_ids = X.index.unique("IP_PATIENT_ID").values
            train_val_labels = y.groupby("IP_PATIENT_ID").first()
            if not separate_eval_dataset:  # need to split twice
                train_val_ids, test_ids = train_test_split(
                    sample_ids["train_val"],
                    test_size=self.test_split_size,
                    stratify=train_val_labels,
                    random_state=self.seed,
                )
            else:
                # split train_val into train and val, and take all of eval to be test
                train_val_ids = sample_ids["train_val"]
                test_ids = sample_ids["eval"]

            train_ids, val_ids = train_test_split(
                train_val_ids,
                test_size=self.val_split_size,
                stratify=train_val_labels[train_val_ids],
                random_state=self.seed,
            )

        # In order to serialize splits for rolling window analysis
        self.split_pt_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

        # return (X,y) pair, where X is a List of pd dataframes for each pt
        # this is so the dimensions match when we zip them into a pytorch dataset
        # Note that like this, we will end up with a set size larger than we expect
        # Because we stratify by ID and certain patients may have more treatments than others.
        if not separate_eval_dataset:  # pull all data from the same cohort
            return {
                split_name: (X.loc[ids], y[ids])
                for split_name, ids in self.split_pt_ids.items()
            }
        # otherwise pull test split from the eval cohort separately
        return {
            "train": (X.loc[train_ids], y[train_ids]),
            "val": (X.loc[val_ids], y[val_ids]),
            "test": (X_eval.loc[test_ids], y_eval[test_ids]),
        }

    @staticmethod
    # def add_data_args(parent_parsers: List[ArgumentParser]) -> ArgumentParser:
    def add_data_args(p: ArgumentParser) -> ArgumentParser:
        # TODO: Add required when using ctn learning or somethign
        # p = ArgumentParser(parents=parent_parsers, add_help=False)
        p.add_argument(
            "--outcome-col-name",
            type=str,
            default="recommend_crrt",
            help="Name of outcome column in outcomes table or preprocessed df.",
        )
        p.add_argument(
            "--test-split-size",
            type=float,
            help="Percent of whole dataset to use for training.",
        )
        p.add_argument(
            "--val-split-size",
            type=float,
            help="Percent of train_val dataset to use for validation. Equivalent / real value = (1 - test-split-size) * val-split-size.",
        )
        p.add_argument(
            "--kbest",
            type=int,
            default=None,
            help="Name of outcome column in outcomes table or preprocessed df.",
        )
        p.add_argument(
            "--corr_thresh",
            type=float,
            default=None,
            help="Name of outcome column in outcomes table or preprocessed df.",
        )
        p.add_argument(
            "--impute-method",
            type=str,
            default="simple",
            choices=["simple", "knn"],
            help="Which impute_method method is desired.",
        )
        p.add_argument(
            "--train-val-cohort",
            type=str,
            choices=[
                "ucla_crrt",
                "ucla_control",
                "ucla_crrt+control",
                "cedars_crrt",
                "ucla_crrt+cedars_crrt",
            ],
            help="Name of cohort/dataset to use for training and validation.",
        )
        p.add_argument(
            "--eval-cohort",
            type=str,
            choices=[
                "ucla_crrt",
                "ucla_control",
                "ucla_crrt+control",
                "cedars_crrt",
                "ucla_crrt+cedars_crrt",
            ],
            help="Name of cohort/dataset to use for evaluation.",
        )
        return p
