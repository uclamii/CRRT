from argparse import ArgumentParser, Namespace
from lib2to3.pgen2.token import OP
from typing import Callable, Dict, List, Optional, Tuple, Union
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

# Local
from data.longitudinal_features import CATEGORICAL_COL_REGEX
from data.base_loaders import AbstractCRRTDataModule, DataLabelTuple
from data.utils import Preselected, SelectThreshold, f_pearsonr
from data.load import load_data

ADDITIONAL_CATEGORICAL_COLS = [
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
            self.imputer = SimpleImputer(strategy="mean")
        # TODO: allow different neighbors to be passed in to be tuned
        elif self.impute_method == "knn":
            self.imputer = KNNImputer()

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

        #### Columns ####
        # Needs to come after reference cols are potentially set
        # need to save this before feature selection for sliding window analysis
        self.columns = X.columns
        self.categorical_columns = X.filter(
            regex=CATEGORICAL_COL_REGEX, axis=1
        ).columns.union(ADDITIONAL_CATEGORICAL_COLS)
        # set this here instead of init so that outcome col isn't included
        self.ctn_columns = X.columns.difference(self.categorical_columns)

        # its the same for all the sequences, just take one
        # y = y.groupby("IP_PATIENT_ID").last()

        split_args = [X, y, reference_ids]
        if self.train_val_cohort != self.eval_cohort:
            X_eval, y_eval = self.load_data_and_additional_preproc(
                args, self.eval_cohort, reference_cols
            )
            split_args += [X_eval, y_eval]
        train_tuple, val_tuple, test_tuple = self.split_dataset(*split_args)

        # Apply filters for subpopulation analysis later
        # MUST OCCUR BEFORE TRANSFORM (before feature selection)
        if self.filters:
            # We set up filters ahead of time so that we don't have to worry about feature selection
            self.train_filters = {k: v(train_tuple[0]) for k, v in self.filters.items()}
            self.val_filters = {k: v(val_tuple[0]) for k, v in self.filters.items()}
            self.test_filters = {k: v(test_tuple[0]) for k, v in self.filters.items()}

        # fit pipeline on train, call transform in get_item of dataset
        if data_transform is not None:
            self.data_transform = data_transform
        else:
            self.data_transform = self.get_post_split_transform(train_tuple)

        # set self.train, self.val, self.test
        # self.train = CRRTDataset(train_tuple, self.data_transform)
        # self.val = CRRTDataset(val_tuple, self.data_transform)
        # self.test = CRRTDataset(test_tuple, self.data_transform)
        self.train = (self.data_transform(train_tuple[0]), train_tuple[1])
        self.val = (self.data_transform(val_tuple[0]), val_tuple[1])
        self.test = (self.data_transform(test_tuple[0]), test_tuple[1])

    def load_data_and_additional_preproc(
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
                # feature-selection doesn't allow NaNs in the data, make sure to impute first.
                ("feature-selection", self.get_feature_selection(reference_cols_mask)),
            ]
        )

        data, labels = train
        pipeline.fit(data, labels)

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
    ) -> Tuple[DataLabelTuple, DataLabelTuple, DataLabelTuple]:
        """
        Splitting with stratification using sklearn.
        We then convert to Dataset so the Dataloaders can use that.

        If train_val_cohort != eval_cohort then the whole test split will be from the test_cohort loaded data.
        Reference IDs will be from there as well.
        """
        separate_eval_dataset = X_eval is not None and y_eval is not None
        # sample = [pt, treatment]
        # TODO: ensure patient is in same split
        # ensure data is split by patient
        sample_ids = {"train_val": X.index.droplevel(["Start Date"]).unique().values}
        train_val_labels = y.groupby("IP_PATIENT_ID").first()
        if separate_eval_dataset:
            sample_ids["eval"] = X_eval.index.droplevel(["Start Date"]).unique().values

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
            return ((X.loc[ids], y[ids]) for ids in (train_ids, val_ids, test_ids))
        # otherwise pull test split from the eval cohort separately
        return (
            (X.loc[train_ids], y[train_ids]),
            (X.loc[val_ids], y[val_ids]),
            (X_eval.loc[test_ids], y_eval[test_ids]),
        )

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
            choices=["ucla_crrt", "ucla_control", "ucla_crrt+control", "cedars_crrt"],
            help="Name of cohort/dataset to use for training and validation.",
        )
        p.add_argument(
            "--eval-cohort",
            type=str,
            choices=["ucla_crrt", "ucla_control", "ucla_crrt+control", "cedars_crrt"],
            help="Name of cohort/dataset to use for evaluation.",
        )
        return p
