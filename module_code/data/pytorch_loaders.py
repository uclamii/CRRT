from argparse import ArgumentParser, Namespace
import inspect
from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sktime.transformations.series.impute import Imputer

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa

# from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer  # , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SplitDataTuple = Tuple[pd.DataFrame, pd.Series, List[int]]


class CRRTDataModule(pl.LightningDataModule):
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

    def setup(self, stage: Optional[str] = None):
        """
        Ops performed across GPUs. e.g. splits, transforms, etc.
        """
        # remove unwanted columns, esp non-numeric ones, before pad and pack
        df = self.preprocessed_df.select_dtypes(["number"])
        # target is separate from the original outcomes, don't include them
        # sets dims, note this will pad target column too,
        # padded_array = self.pad_sequences(df)
        padded_array = df

        # split into data and labels
        outcome_col_index = self.preprocessed_df.columns.get_loc(
            self.hparams.outcome_col_name
        )
        X, y = (
            # ignore outcome col
            np.delete(padded_array, outcome_col_index, axis=-1),
            # keep only outcome col
            padded_array[:, :, outcome_col_index],
        )
        # remove 1 from nfeatures since outcome col got removed
        self.dims = (self.dims[0], self.dims[1], self.dims[2] - 1)

        train_tuple, val_tuple, test_tuple = self.split_dataset(X, y)
        train_tuple, val_tuple, test_tuple = self.process_dataset(
            train_tuple, val_tuple, test_tuple
        )

        # set self.train, self.val, self.test
        self.train = self.get_dataset_from_df(train_tuple)
        self.val = self.get_dataset_from_df(val_tuple)
        self.test = self.get_dataset_from_df(test_tuple)

    def train_dataloader(self):
        return self.get_dataloader(self.train)

    def val_dataloader(self):
        return self.get_dataloader(self.val)

    def test_dataloader(self):
        return self.get_dataloader(self.test)

    #############
    #  HELPERS  #
    #############
    # def get_dataset_from_df(
    # self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray],
    # ) -> Dataset:
    def get_dataset_from_df(self, *args) -> Dataset:
        """
        Pytorch modules require Datasets for train/val/test,
        but we start with a df or ndarray, especially after splitting the data.
        """
        return TensorDataset(*(Tensor(arg) for arg in args))

    def get_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_gpus * 4,
        )

    def process_dataset(
        self, train: SplitDataTuple, val: SplitDataTuple, test: SplitDataTuple
    ) -> Tuple[SplitDataTuple, SplitDataTuple, SplitDataTuple]:
        """
        The serialized preprocessed df should alreayd have dealth with categorical variables and aggregated them as counts, so we only deal with numeric / continuous variables.
        """
        # TODO: write tests
        pipeline = Pipeline(
            [
                # ("scale", StandardScaler()),
                # ("iteraitve-impute", IterativeImputer(max_iter=10, random_state=self.seed)),
                ("knn-impute", KNNImputer()),
            ]
        )

        def transform_nonpadded(
            split_data: Tuple[SplitDataTuple, SplitDataTuple, SplitDataTuple],
            fit: bool = False,
        ) -> np.ndarray:
            """Transform only nonpadded portion of data."""
            X, _, seq_lens = split_data
            batch_size, max_seq_len, nfeatures = X.shape
            # flatten to 2d
            flattened = X.reshape(batch_size * max_seq_len, nfeatures)
            mask_out_padded = (flattened != self.PAD_VALUE).all(axis=1)
            flattened_no_padding = flattened[mask_out_padded]
            # impute on 2d
            if fit:
                print("Fitting transformer...")
                pipeline.fit(flattened_no_padding)
            print("Running transform..")
            imputed = pipeline.transform(flattened_no_padding)

            # put imputed values back in
            start = 0
            for i, seq_len in enumerate(seq_lens):
                X[i][:seq_len] = imputed[start : start + seq_len]
                start += seq_len

            return X

        # Replace with transformed version
        print("Transforming data...")
        train_tuple = transform_nonpadded(train, fit=True) + train[1:]
        val_tuple = transform_nonpadded(val) + val[1:]
        test_tuple = transform_nonpadded(test) + test[1:]

        return (train_tuple, val_tuple, test_tuple)

    def split_dataset(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[SplitDataTuple, SplitDataTuple, SplitDataTuple]:
        """
        Splitting with stratification using sklearn, needs nparray.
        We then convert to Dataset so the Dataloaders can use that.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values

        # its the same for all the sequences, just take the first one
        y = y[:, 0]

        (
            X_train_val,
            X_test,
            y_train_val,
            y_test,
            seq_lens_train_val,
            seq_lens_test,
        ) = train_test_split(
            X,
            y,
            self.seq_lengths,
            test_size=self.hparams.test_split_size,
            stratify=y,
            random_state=self.seed,
        )
        X_train, X_val, y_train, y_val, seq_lens_train, seq_lens_val = train_test_split(
            X_train_val,
            y_train_val,
            seq_lens_train_val,
            test_size=self.hparams.val_split_size,
            stratify=y_train_val,
            random_state=self.seed,
        )

        train_tuple = (X_train, y_train, seq_lens_train)
        val_tuple = (X_val, y_val, seq_lens_val)
        test_tuple = (X_test, y_test, seq_lens_test)

        return (train_tuple, val_tuple, test_tuple)

        """
        ##### This ways uses random_split from pytorch itself, cannot stratify
        # Assign train/val/test datasets for use in dataloaders
        # Reproducibility
        generator = Generator().manual_seed(self.seed)
        # Get literal sizes for random_split
        test_size = int(self.hparams.test_split_size * len(dataset))
        train_val_size = len(dataset) - test_size
        # split trainval / test
        train_val, self.test = random_split(
            dataset, [train_val_size, test_size], generator=generator
        )
        # split train and val from trainval
        val_size = int(self.hparams.val_split_size * len(train_val))
        train_size = len(train_val) - val_size
        self.train, self.val = random_split(
            train_val, [train_size, val_size], generator=generator
        )
        """

    def pad_sequences(self, X: pd.DataFrame) -> np.ndarray:
        """
        Sets dimensions of dataset.
        https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        """
        # TODO: write tests
        grouped_sample_dfs = X.groupby(level="IP_PATIENT_ID")
        n_samples = grouped_sample_dfs.ngroups  # number of patients
        # Save this for packing/unpacking later
        self.seq_lengths = grouped_sample_dfs.size().values  # seq len per patient
        n_features = X.shape[1]  # number of features per pt per seq entry
        longest_seq = max(self.seq_lengths)

        # Create empty matrix with padding tokens
        self.dims = (n_samples, longest_seq, n_features)
        padded = np.ones(self.dims) * self.PAD_VALUE
        # copy over the actual sequences
        start = 0
        for i, seq_len in enumerate(self.seq_lengths):
            # original sequence values for sample i (.iloc gets row by row, not by outer index (pt))
            sequence = X.iloc[start : start + seq_len].values
            # fill sample i from beginning to seq_len with the original values
            padded[i, 0:seq_len] = sequence
            start += seq_len

        return padded

    @staticmethod
    def add_data_args(parent_parser: ArgumentParser) -> ArgumentParser:
        # TODO: Add required when using ctn learning or somethign
        p = ArgumentParser(parents=[parent_parser], add_help=False)
        p.add_argument(
            "--batch-size", type=int, help="Batch size to use when training.",
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

    @classmethod
    def from_argparse_args(
        cls,
        preprocessed_df: np.ndarray,
        args: Union[Namespace, ArgumentParser],
        **kwargs
    ) -> "CRRTDataModule":
        """
        Create an instance from CLI arguments.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
        # Ref: https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.8.3/pytorch_lightning/trainer/trainer.py#L750
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        data_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        data_kwargs.update(**kwargs)

        return cls(preprocessed_df, **data_kwargs)
