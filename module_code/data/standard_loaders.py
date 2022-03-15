from argparse import ArgumentParser
from typing import Callable, Optional, Tuple, Union
import pandas as pd
import numpy as np

# from sktime.transformations.series.impute import Imputer
# explicitly require this experimental feature
# from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

# Local
from data.longitudinal_features import CATEGORICAL_COL_REGEX
from data.base_loaders import AbstractCRRTDataModule, CRRTDataset, DataLabelTuple
from module_code.data.utils import SelectThreshold, f_pearsonr


class SklearnCRRTDataModule(AbstractCRRTDataModule):
    def __init__(
        self,
        preprocessed_df: pd.DataFrame,
        seed: int,
        outcome_col_name: str,
        test_split_size: float,
        # val comes from train := (1 - test_split_size) * val_split_size
        val_split_size: float,
        kbest=None,
        corr_thresh=None,
    ):
        super().__init__()
        self.seed = seed
        self.preprocessed_df = preprocessed_df
        self.outcome_col_name = outcome_col_name
        self.test_split_size = test_split_size
        self.val_split_size = val_split_size
        self.categorical_columns = preprocessed_df.filter(
            regex=CATEGORICAL_COL_REGEX, axis=1
        ).columns
        self.ctn_columns = preprocessed_df.columns.difference(self.categorical_columns)
        self.kbest = kbest
        self.corr_thresh = corr_thresh

    def setup(self, stage: Optional[str] = None):
        """
        Ops performed across GPUs. e.g. splits, transforms, etc.
        """
        X, y = (
            self.preprocessed_df.drop(self.outcome_col_name, axis=1),
            self.preprocessed_df[self.outcome_col_name],
        )
        # its the same for all the sequences, just take one
        # y = y.groupby("IP_PATIENT_ID").last()

        # remove unwanted columns, esp non-numeric ones, before pad and pack
        X = X.select_dtypes(["number"])

        self.nfeatures = X.shape[1]

        train_tuple, val_tuple, test_tuple = self.split_dataset(X, y)

        # fit pipeline on train, call transform in get_item of dataset
        self.data_transform = self.get_post_split_transform(train_tuple)

        # feature selection
        self.ft_select_transform = self.get_post_split_features(train_tuple)

        # set self.train, self.val, self.test
        self.train = CRRTDataset(
            train_tuple, self.data_transform, self.ft_select_transform
        )
        self.val = CRRTDataset(val_tuple, self.data_transform, self.ft_select_transform)
        self.test = CRRTDataset(
            test_tuple, self.data_transform, self.ft_select_transform
        )

    def get_post_split_transform(self, train: DataLabelTuple) -> Callable:
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
                                "fillna",
                                SimpleImputer(strategy="mean"),
                                # TODO convert to indices
                                self.ctn_columns,
                            )
                        ],
                        remainder="passthrough",
                    ),
                ),
                # zero out everything else
                ("simple-impute", SimpleImputer(strategy="constant", fill_value=0)),
                ("feature-selection", self.get_feature_selection()),
            ]
        )

        data, labels = train
        pipeline.fit(data)

        return pipeline.transform

    def get_feature_selection(self) -> Callable:
        """
        Fit the feature selection transform based on either the k best features or the number of features
        above a correlation threshold. Passthrough option also available.
        """
        # TODO: Test these work as intended
        if self.kbest and self.corr_thresh:
            raise ValueError("Both kbest and corr_thresh are not None")
        if self.kbest:
            # TODO: maybe want to update sklearn and use https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.r_regression.html#sklearn.feature_selection.r_regression
            return SelectKBest(f_pearsonr, k=self.kbest)
        elif self.corr_thresh:
            return SelectThreshold(f_pearsonr, threshold=self.corr_thresh)
        # passthrough transform
        return SelectKBest(lambda X, y: np.zeros(X.shape[1]), k="all")

    def split_dataset(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
    ) -> Tuple[DataLabelTuple, DataLabelTuple, DataLabelTuple]:
        """
        Splitting with stratification using sklearn.
        We then convert to Dataset so the Dataloaders can use that.
        """
        # sample = [pt, treatment]
        # TODO: ensure patient is in same split
        # ensure data is split by patient
        sample_ids = X.index.droplevel(["Start Date", "DATE"]).unique().values
        labels = y.groupby("IP_PATIENT_ID").first()
        # patient_ids = X.index.unique("IP_PATIENT_ID").values
        train_val_ids, test_ids = train_test_split(
            sample_ids,
            test_size=self.test_split_size,
            stratify=labels,
            random_state=self.seed,
        )
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=self.val_split_size,
            stratify=labels[train_val_ids],
            random_state=self.seed,
        )

        # return (X,y) pair, where X is a List of pd dataframes for each pt
        # this is so the dimensions match when we zip them into a pytorch dataset
        return (
            (
                X.reset_index(level=["Start Date", "DATE"])
                .loc[ids]
                .set_index(["Start Date", "DATE"], append=True),
                labels[ids],
            )
            # (X.loc[ids], y[ids])
            for ids in (train_ids, val_ids, test_ids)
        )

    @staticmethod
    def add_data_args(parent_parser: ArgumentParser) -> ArgumentParser:
        # TODO: Add required when using ctn learning or somethign
        p = ArgumentParser(parents=[parent_parser], add_help=False)
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
        return p
