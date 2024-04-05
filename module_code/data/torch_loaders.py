"""
Module for processing data tables for modelling
TODO: this has not been maintained. Will update as we move towards dynamic models
"""

from argparse import ArgumentParser
from typing import Callable, Optional, Tuple, Union
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# from sktime.transformations.series.impute import Imputer
# explicitly require this experimental feature
# from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer

# from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data.longitudinal_features import CATEGORICAL_COL_REGEX
from data.base_loaders import AbstractCRRTDataModule, CRRTDataset, DataLabelTuple


class TorchCRRTDataModule(pl.LightningDataModule, AbstractCRRTDataModule):
    PAD_VALUE = -1

    def __init__(
        self,
        preprocessed_df: pd.DataFrame,
        seed: int,
        batch_size: int,
        num_gpus: int,
        outcome_col_name: str,
        test_split_size: float,
        # val comes from train := (1 - test_split_size) * val_split_size
        val_split_size: float,
    ):
        super().__init__()
        self.seed = seed
        self.preprocessed_df = preprocessed_df
        self.save_hyperparameters(ignore=["seed", "preprocessed_df"])
        self.categorical_columns = preprocessed_df.filter(
            regex=CATEGORICAL_COL_REGEX, axis=1
        ).columns

    def setup(self, stage: Optional[str] = None):
        """
        Ops performed across GPUs. e.g. splits, transforms, etc.
        """
        X, y = (
            self.preprocessed_df.drop(self.hparams.outcome_col_name, axis=1),
            self.preprocessed_df[self.hparams.outcome_col_name],
        )
        # its the same for all the sequences, just take one
        # y = y.groupby("IP_PATIENT_ID").last()

        # remove unwanted columns, esp non-numeric ones, before pad and pack
        X = X.select_dtypes(["number"])

        self.nfeatures = X.shape[1]

        train_tuple, val_tuple, test_tuple = self.split_dataset(X, y)

        # fit pipeline on train, call transform in get_item of dataset
        transform = self.get_post_split_transform(train_tuple)

        # set self.train, self.val, self.test
        self.train = TorchCRRTDataset(train_tuple, transform)
        self.val = TorchCRRTDataset(val_tuple, transform)
        self.test = TorchCRRTDataset(test_tuple, transform)

    def train_dataloader(self):
        return self.get_dataloader(self.train)

    def val_dataloader(self):
        return self.get_dataloader(self.val)

    def test_dataloader(self):
        return self.get_dataloader(self.test)

    #############
    #  HELPERS  #
    #############
    def get_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            collate_fn=self.batch_collate,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_gpus * 4,
        )

    def batch_collate(
        self, batch: Tuple[np.ndarray, int], transform: Callable = None
    ) -> DataLabelTuple:
        """
        Batch is a list of tuples with (example, label).
        Pad the variable length sequences, add seq lens, and enforce tensor.
        """
        X, y = zip(*batch)
        y = Tensor(y)
        seq_lens = Tensor([len(pt_seq) for pt_seq in X])
        X = pad_sequence(X, batch_first=True, padding_value=self.PAD_VALUE)

        return (X, y, seq_lens)

    def get_post_split_transform(self, train: DataLabelTuple) -> Callable:
        """
        The serialized preprocessed df should alreayd have dealth with categorical variables and aggregated them as counts, so we only deal with numeric / continuous variables.
        """
        # TODO: write tests
        pipeline = Pipeline(
            [
                (
                    "categorical-fillna",
                    ColumnTransformer(
                        [  # (name, transformer, columns) tuples
                            (
                                "fillna",
                                SimpleImputer(strategy="constant", fill_value=0),
                                # TODO convert to indices
                                self.categorical_columns,
                            )
                        ],
                        remainder="passthrough",
                    ),
                ),
                # (
                #     "interpolate",
                #     FunctionTransformer(
                #         func=lambda nparr: pd.DataFrame(nparr)
                #         .interpolate(method="linear")
                #         .values
                #     ),
                # )
                # ("scale", StandardScaler()),
                # ("iteraitve-impute", IterativeImputer(max_iter=10, random_state=self.seed)),
                # TODO: this might explode with more patients (since features will increase)
                # TODO: alternate: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
                # ("knn-impute", KNNImputer()),
                ("simple-impute", SimpleImputer(strategy="constant", fill_value=0)),
            ]
        )

        data, labels = train
        # flattens each patient,treatment squence into one long df
        # TODO: This is just a hack until our imputation is more developed
        pipeline.fit(pd.concat(data))

        return pipeline.data_transform

    def split_dataset(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> Tuple[DataLabelTuple, DataLabelTuple, DataLabelTuple]:
        """
        Splitting with stratification using sklearn.
        We then convert to Dataset so the Dataloaders can use that.
        """
        # sample = [pt, treatment]
        # TODO: ensure patient is in same split
        # do not separate separate dates per a patient
        sample_ids = X.index.droplevel("DATE").unique().values
        labels = y.groupby(["IP_PATIENT_ID", "Start Date"]).first()
        # patient_ids = X.index.unique("IP_PATIENT_ID").values
        train_val_ids, test_ids = train_test_split(
            sample_ids,
            test_size=self.hparams.test_split_size,
            stratify=labels,
            random_state=self.seed,
        )
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=self.hparams.val_split_size,
            stratify=labels[train_val_ids],
            random_state=self.seed,
        )

        # return (X,y) pair, where X is a List of pd dataframes for each pt
        # this is so the dimensions match when we zip them into a pytorch dataset
        return (
            ([X.loc[id] for id in ids], labels[ids])
            # (X.loc[ids], y[ids])
            for ids in (train_ids, val_ids, test_ids)
        )

    @staticmethod
    def add_data_args(p: ArgumentParser) -> ArgumentParser:
        p.add_argument(
            "--batch-size",
            type=int,
            help="Batch size to use when training.",
        )
        p.add_argument(
            "--num-gpus",
            type=int,
            help="Number of GPUs to use for data loading and other data and training processes. Note this is used for both data loading and pl.Trainer.",
        )
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
        return p


class TorchCRRTDataset(CRRTDataset, Dataset):
    def __getitem__(self, index: int):
        X, y = super().__getitem__(index)
        return Tensor(X), y
