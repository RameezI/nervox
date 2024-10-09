# Copyright(c) 2023 Rameez Ismail - All Rights Reserved
# Author: Rameez Ismail
# Email: rameez.ismaeel@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict
from nervox.protocols.protocol import Protocol
from nervox.utils.progress_bar import ProgressBar
from nervox.utils.types import Number, TensorLike
from nervox.utils.auxiliaries import VerbosityLevel
from datetime import datetime as dt
from typing import Union
import logging

from nervox.callbacks import Callback
from nervox.utils import VerbosityLevel

logger = logging.getLogger(__name__)


class CSVLogger(Callback):
    def __init__(self):
        super(CSVLogger, self).__init__()

    def on_epoch_end(self, epoch, logs: Union[None, Dict[str, Number]] = None):
        pass


class CheckPointer(Callback):
    def __init__(
        self,
        checkpoint_dir: os.PathLike,
        monitor: str = "loss/val",
        silent: bool = False,
        keep_best: bool = True,
        mode: str = "auto",
        max_to_keep: int = 3,
        **kwargs,
    ):
        """
        The check-pointer callback is used to save the model's weights and optimizer's state at the end of each epoch.
        The check-pointer can be configured to keep the best model based on the monitored variable, or the last checkpoint.
        It saves the model's weights and optimizer's state in a checkpoint file, which can be loaded using the
        `tf.train.Checkpoint.restore` method.

        Args:
            checkpoint_dir:         The directory where the checkpoints are created.
            monitor:                The monitored variable
            silent:                 Weather to print updates on checkpoint event
            keep_best:              Weather to keep the best, based on the monitored value, or the last checkpoint.
            mode:                   How to determine the best checkpoint based on the monitored variable,
                                    accepts one of [`min`, `max`, `auto`]
            max_to_keep:            Maximum number of checkpoint to keep.

        Keyword Args [optional]:
            progress_bar:           Optionally accepts a progress bar used to update the check-point statement.

        """

        super(CheckPointer, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.silent = silent
        self.keep_best = keep_best
        self.monitor = monitor
        self.max_to_keep = max_to_keep
        self._current_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._progress_bar = kwargs.get("progress_bar", None)

        (
            os.makedirs(str(self.checkpoint_dir))
            if not os.path.exists(self.checkpoint_dir)
            else None
        )

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                f"ModelCheckpoint mode {mode} is unknown, fallback to auto mode."
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if self.monitor.startswith(("acc", "mAP", "fmeasure")):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        # Lazily create, at the start of the training, a checkpoint manager
        self.ckpt_manager = None

    def on_eval_end(self, metrics: Dict[str, TensorLike]):
        if self.keep_best and metrics:
            initial_best = metrics.get(self.monitor)
            if initial_best is None:
                logger.warning(
                    f"\n{'Missing monitored-key':-^35}\n"
                    f"The checkpointer callback did not find the monitored key in the initial logs,"
                    f"\nif it is not found during training; will keep the {self.max_to_keep} most"
                    f" recent points instead!\n"
                )
        self.best = metrics.get(self.monitor, self.best)

    def on_train_begin(self):
        checkpoint = tf.train.Checkpoint(
            epoch=self._current_epoch,
            **self.models,
            objectives=self._protocol.objectives,
        )

        self.ckpt_manager = tf.train.CheckpointManager(
            checkpoint,
            self.checkpoint_dir,
            checkpoint_name="ckpt",
            max_to_keep=self.max_to_keep,
        )

    def on_epoch_end(self, epoch, metrics: Dict[str, TensorLike]):

        self._current_epoch.assign(epoch)

        if self.monitor not in metrics:
            self.keep_best = False

        if self.keep_best:
            current = metrics.get(self.monitor, None)
            if self.monitor_op(current, self.best):
                self.ckpt_manager.save()
                if self._progress_bar is not None:
                    self._progress_bar.ckpt_statement = {"is_ckpt": True}
                else:
                    (
                        print(f"checkpoint: {self.best}-->{current}!")
                        if not self.silent
                        else None
                    )
                self.best = current
        else:
            self.ckpt_manager.save()
            if self._progress_bar is not None:
                self._progress_bar.ckpt_statement = {"is_ckpt": True}
            else:
                print("checkpoint!") if not self.silent else None


class ProgressParaphraser(Callback):
    """Callback that prints (formatted) progress and metrics to stdout."""

    def __init__(
        self,
        progress_bar: ProgressBar,
        verbose: VerbosityLevel = VerbosityLevel.UPDATE_AT_BATCH,
    ) -> None:
        super(ProgressParaphraser, self).__init__()
        self._progress_bar = progress_bar
        self.eval_statement_only = False
        self.verbose = verbose

    def on_train_begin(self):
        # get max epochs and pass it to the progress bar
        self._progress_bar.max_epochs = self.trainer.max_epoch
        (
            print(f"Initiating Train/Validate Cycle: ", end="")
            if self.verbose not in [VerbosityLevel.KEEP_SILENT]
            else None
        )

    def on_epoch_begin(self, epoch):
        # prepare statement
        self._progress_bar.reset_statements()
        self._progress_bar.epoch = epoch
        self._progress_bar.train_samples = self.trainer.train_samples_count
        self._progress_bar.epoch_start_time = dt.now()
        print(end="\n") if self.verbose not in [VerbosityLevel.KEEP_SILENT] else None

    def on_eval_begin(self):
        # prepare statement
        self._progress_bar.eval_samples = self.trainer.eval_samples_count
        self._progress_bar.eval_start_time = dt.now()

    def on_train_batch_end(self, step, metrics: Dict[str, TensorLike]):
        # update statement
        if step == 0:
            self._progress_bar.time_after_first_batch = dt.now()
        step_count = step + 1
        samples_done = metrics.pop("samples_done", np.nan)
        self._progress_bar.train_metrics = metrics
        self._progress_bar.train_statement = {
            "step_count": step_count,
            "samples_done": samples_done,
        }
        (
            print(f"\r{self._progress_bar}", end="")
            if self.verbose in [VerbosityLevel.UPDATE_AT_BATCH]
            else None
        )

    def on_eval_batch_end(self, step, metrics: Dict[str, TensorLike]):
        # update statement
        if step == 0:
            self._progress_bar.time_after_first_batch = dt.now()
        step_count = step + 1
        samples_done = metrics.pop("samples_done", np.nan)
        self._progress_bar.eval_metrics = metrics
        self._progress_bar.eval_statement = {
            "step_count": step_count,
            "samples_done": samples_done,
        }
        (
            print(f"\r{self._progress_bar}", end="")
            if self.verbose in [VerbosityLevel.UPDATE_AT_BATCH]
            else None
        )

    def on_eval_end(self, _):
        (
            print(f"\r{self._progress_bar}", end="")
            if self.verbose not in [VerbosityLevel.KEEP_SILENT]
            else None
        )

    def on_epoch_end(self, epoch, _):
        self._progress_bar.epoch = epoch
        self._progress_bar.train_samples = self.trainer.train_samples_count
        self._progress_bar.eval_samples = self.trainer.eval_samples_count
        (
            print(f"\r{self._progress_bar}", end="")
            if self.verbose not in [VerbosityLevel.KEEP_SILENT]
            else None
        )
