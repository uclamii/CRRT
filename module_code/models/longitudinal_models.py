from argparse import ArgumentParser, Namespace
import inspect
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

# ML modules
import torch
from torch import Tensor, load
from torch.nn import LSTM, Linear, Module, ModuleDict, Sigmoid
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics


from sktime.classification.base import BaseClassifier
from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.kernel_based import ROCKETClassifier

from data.pytorch_loaders import CRRTDataModule
from data.argparse_utils import YAMLStringListToList


class LongitudinalModel(pl.LightningModule):
    def __init__(
        self,
        seed: int,
        # Model params
        modeln: str,
        nfeatures: int,
        hidden_layers: List[int],
        # Training Params
        metrics: List[str],
        loss_name: str,
        optim_name: str = "Adam",
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,  # l2_penalty
    ):
        super().__init__()
        self.seed = seed
        # using this a little bit as a cheat so i dont have to do self.x = x
        self.save_hyperparameters(ignore=["seed", "metrics"])

        self.model = self.build_model()
        self.metrics = self.configure_metrics(metrics)
        self.loss = self.configure_loss(loss_name)

    ####################################
    #  Pt Lightning Overriden Methods  #
    ####################################
    # Order: https://pytorch-lightning.readthedocs.io/en/latest/starter/style_guide.html#method-order
    def forward(self, X: Tensor, orig_seq_lens: Tensor) -> Tensor:
        """
        What happens when you pass data through an object of this class.
        Pack > pass through model > unpack
        """
        batch_size, seq_len, nfeatures = X.shape
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        # lengths required to be on cpu
        X_packed = pack_padded_sequence(
            X, orig_seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        # pass data to the lstm
        lstm_output_packed, (hidden, cell) = self.model["lstm"](X_packed)
        # undo the packing operation
        # lstm_output, lengths = pad_packed_sequence(lstm_output_packed, batch_first=True)
        # classification: grab last hidden state (h[-1])
        last_outputs = hidden[-1]

        # Flatten to vector of features
        fc_output = self.model["fc"](last_outputs)
        # output activation
        output = self.model["output_activation"](fc_output)

        # enforce dimensions are the same at the end
        return output

    def training_step(self, batch: Tensor, batch_idx) -> torch.float:
        loss, outputs = self.shared_step(batch)
        self.shared_logging_step_end(outputs, "train")
        return loss

    def validation_step(self, batch, batch_idx) -> torch.float:
        loss, outputs = self.shared_step(batch)
        self.shared_logging_step_end(outputs, "val")
        return loss

    def test_step(self, batch, batch_idx) -> torch.float:
        loss, outputs = self.shared_step(batch)
        self.shared_logging_step_end(outputs, "test")
        return loss

    ############################
    #  Training Logic Helpers  #
    ############################
    def shared_step(self, batch) -> Dict[str, float]:
        data, ground_truth, orig_seq_lens = batch
        output = self(data, orig_seq_lens).squeeze()
        # loss = self.compute_fn_on_ouput(output, ground_truth, self.loss)
        loss = self.loss(output, ground_truth)

        # Waiting for the step_end to compute metrics is dataparallel compatible, so pass pred and ground_truth through
        return (
            loss,
            {
                "loss": loss,
                "pred": output.detach().cpu(),
                "ground_truth": ground_truth.to(int).detach().cpu(),
            },
        )

    # def compute_fn_on_ouput(
    #     self,
    #     pred: Tensor,
    #     true: Tensor,
    #     fn: Callable[[Tensor, Tensor], float],
    #     orig_seq_lens: Optional[Tensor] = None,
    # ) -> float:
    #     """
    #     Because we're padding, we want to compute fn's ignoring the padding.
    #     https://gist.github.com/williamFalcon/f27c7b90e34b4ba88ced042d9ef33edd#file-pytorch_lstm_variable_mini_batches-py
    #     """
    #     # mask: samples that ARE NOT padding  (features on last axis)
    #     # TODO: do i need a mask?? it's just binary classification
    #     return fn(pred.detach().cpu(), true.to(int).detach().cpu())

    def shared_logging_step_end(self, outputs: Dict[str, float], split: str):
        """Log metrics + loss at end of step.
        Waiting for the step_end to compute metrics is dataparallel compatible.
        https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html
        """
        # Log loss
        self.log(
            f"{split}-loss",
            outputs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # NOTE: if you add too many metrics it will mess up the progress bar
        # Log all metrics
        for metricfn in self.metrics:
            self.log(
                f"{split}-{metricfn._get_name()}",
                self.compute_fn_on_ouput(
                    outputs["pred"], outputs["ground_truth"], metricfn
                ),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

    ############################
    #  Initialization Helpers  #
    ############################
    def build_model(self,) -> Union[Module, BaseClassifier]:
        # https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.classification.hybrid.HIVECOTEV1.html
        # TODO: when calling fit in the wrapper class, just call fit on the model on the data, they should work liike sklearn models so the pytorch datamodule can be used even then
        if self.hparams.modeln == "hivecote":
            return HIVECOTEV1(n_jobs=-1, random_state=self.seed)
        # https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.classification.kernel_based.ROCKETClassifier.html
        elif self.hparams.modeln == "rocket":
            return ROCKETClassifier(n_jobs=-1, random_state=self.seed)
        elif self.hparams.modeln == "lstm":
            return ModuleDict(
                {
                    "lstm": LSTM(
                        input_size=self.hparams.nfeatures,
                        hidden_size=self.hparams.hidden_layers[
                            0
                        ],  # TODO: allow stacked lstm?
                        batch_first=True,
                    ),
                    # binary class prediction
                    "fc": Linear(self.hparams.hidden_layers[0], 1),
                    "output_activation": Sigmoid(),
                }
            )

    # This is Pytorch Overriden
    def configure_optimizers(self):
        """Pick optimizer."""
        assert hasattr(optim, self.hparams.optim_name), (
            f"{self.hparams.optim_name} is not valid optimizer name."
            " Must match pytorch optimizer names (e.g. 'Adam')."
        )
        kwargs = ("lr", "weight_decay")
        kwargs = {
            name: getattr(self.hparams, name)
            for name in kwargs
            if getattr(self.hparams, name) is not None
        }

        return getattr(optim, self.hparams.optim_name)(self.parameters(), **kwargs)

    def configure_loss(self, loss_name: str) -> Module:
        """Pick loss."""
        loss_name = f"{loss_name}Loss"
        assert hasattr(torch.nn, loss_name), (
            f"{loss_name} is not valid loss name."
            " Must match pytorch.nn loss names (e.g. 'MSE' for MSELoss())."
        )
        return getattr(torch.nn, loss_name)()

    def configure_metrics(self, metric_names: List[str]) -> List[Callable]:
        """Pick metrics."""
        for metric in metric_names:
            assert hasattr(torchmetrics, metric), (
                f"{metric} is not valid metric name."
                " Must match torchmetrics loss names (e.g. 'Accuracy'."
            )
        return [getattr(torchmetrics, metric)() for metric in metric_names]

    @staticmethod
    def add_model_args(parent_parser: ArgumentParser) -> ArgumentParser:
        # TODO: Add required when using ctn learning or somethign
        p = ArgumentParser(parents=[parent_parser], add_help=False)
        p.add_argument(
            "--modeln",
            type=str,
            default="lstm",
            choices=["lstm", "hivecote", "rocket"],
            help="Name of model to use for continuous learning.",
        )
        p.add_argument(
            "--metrics",
            type=str,
            action=YAMLStringListToList(str),
            help="(List of comma-separated strings no spaces) Name of Pytorch Metrics from torchmetrics.",
        )
        p.add_argument(
            "--loss-name",
            type=str,
            help="Name of Pytorch Loss function (Without 'Loss') from nn.",
        )
        p.add_argument(
            "--optim-name",
            type=str,
            default="Adam",
            help="Name of Pytorch Optimizer from optim.",
        )

        # Traditional HPARAMS
        p.add_argument(
            "--lr", type=float, help="Learning rate for specified optimizer."
        )
        p.add_argument(
            "--weight_decay",
            type=float,
            help="Weight decay, or L2 penalty for specified optimizer.",
        )

        # LSTM args
        p.add_argument(
            "--hidden-layers",
            type=str,
            action=YAMLStringListToList(int),
            help="Hidden size for LSTM.",  # TODO: better help method when decided on stacked lstm or not
        )

        # Trainer-specific args that a trainer that uses this class will need
        # Refer to init of CRRTPredictor class
        # NOTE: num_gpus already from data args, don't add it here
        p.add_argument(
            "--max-epochs",
            type=int,
            default=100,
            help="Maximum number of epochs to run during training.",
        )
        p.add_argument(
            "--patience",
            type=int,
            help="Patience in number of steps to wait for validation to stop improving for early stopping callback during training.",
        )
        return p


class CRRTPredictor(TransformerMixin, BaseEstimator):
    """
    Wrapper predictor class, compatible with sklearn.
    Uses longitudinal model to do time series classification on tabular data.
    Implements fit and transform.
    """

    def __init__(
        self, patience: int = 5, max_epochs: int = 100, num_gpus: int = 1, **kwargs,
    ):
        self.seed = kwargs["seed"]
        self.patience = patience
        self.max_epochs = max_epochs
        self.num_gpus = num_gpus
        self.longitudinal_model = LongitudinalModel(**kwargs)

        callbacks = [EarlyStopping(monitor="val-loss", patience=self.patience)]
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            deterministic=True,
            gpus=self.num_gpus,
            accelerator="ddp" if self.num_gpus > 1 else None,
            checkpoint_callback=False,
            callbacks=callbacks,
            profiler="simple",  # or "advanced" which is more granular
            weights_summary="full",
        )

    def load_model(self, serialized_model_path: str) -> None:
        """Loads the underlying autoencoder state dict from path."""
        self.longitudinal_model.load_state_dict(load(serialized_model_path))

    def fit(self, data: CRRTDataModule):
        """Trains the autoencoder for imputation."""
        pl.seed_everything(self.seed)
        self.data = data
        # self.data.setup()

        self.trainer.fit(self.longitudinal_model, datamodule=self.data)
        if self.runtest:
            self.trainer.test(self.longitudinal_model, datamodule=self.data)

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame],) -> np.ndarray:
        """Applies trained model to given data X."""
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(
                X.values * 1, device=self.longitudinal_model.device, dtype=torch.float
            )
        else:
            X = torch.tensor(
                X, device=self.longitudinal_model.device, dtype=torch.float
            )
        outputs = self.longitudinal_model(X)  # .detach().cpu().numpy()

        return outputs

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs
    ) -> "CRRTPredictor":
        """
        Create an instance from CLI arguments.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
        # Ref: https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.8.3/pytorch_lightning/trainer/trainer.py#L750
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid args, the rest may be user specific
        # returns a immutable dict MappingProxyType, want to combine so copy
        valid_kwargs = inspect.signature(cls.__init__).parameters.copy()
        valid_kwargs.update(
            inspect.signature(LongitudinalModel.__init__).parameters.copy()
        )
        data_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        data_kwargs.update(**kwargs)

        return cls(**data_kwargs)
