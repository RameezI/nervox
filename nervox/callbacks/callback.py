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

import tensorflow as tf
from abc import abstractmethod
from typing import final, Sequence, Dict
from nervox import Trainer
from nervox.protocols import Protocol
from nervox.utils.types import TensorLike


class Callback:
    """Abstract base class used to build new callbacks.
    Callbacks can be passed to nervox trainer through `spin`, in order to hook into the
    various stages of the model training and evaluation. To create a custom callback, just
    subclass `nervox.callbacks.Callback` and override the method associated with the stage
    of interest. The callbacks methods are called automatically at specific stages of the
    training/evaluation and has full access to all the features of the trainer and the
    training protocol.
    """


    @final
    def setup(self, trainer: Trainer, protocol: Protocol):
        """Links the callback object to the trainer and protocol.
        This method is called by the trainer during callback setup."""
        setattr(self, "trainer", trainer)
        setattr(self, "protocol", protocol)

    @abstractmethod
    def on_train_batch_begin(self, step: int):
        """Called at the beginning of a batch processing in the training loop.
        Subclasses should override for any actions to run.
        Args:
            step (int): The index of the current batch being processed.
        """
        ...

    @abstractmethod
    def on_eval_batch_begin(self, step: int):
        """Called at the beginning of a batch processing in the evaluation loop.
        Subclasses should override for any actions to run.
        Args:
            step (int): The index of the current batch being processed.
        """
        ...

    @abstractmethod
    def on_train_batch_end(self, step, metrics: Dict[str, TensorLike]):
        """Called at the end of a batch processing in the training loop.
        Subclasses should override for any actions to run.
        Args:
            step (int):             The index of the batch that is just finalized.

            metrics (dict):         The metrics returned by the protocol at the end
                                    of the batch during training. These are generally
                                    the running averages of the training metrics.
                                    For certain protocols, these can also be the raw
                                    metrics computed at the end of the batch.
        """
        ...

    @abstractmethod
    def on_eval_batch_end(self, step, metrics: Dict[str, TensorLike]):
        """Called at the end of a batch processing in the evaluation loop.
        Subclasses should override for any actions to run.
        Args:
            step (int):             The index of the batch that is just finalized.

            metrics (dict):         The metrics returned by the protocol at the end
                                    of a batch during evaluation. These are generally
                                    the running averages of the evaluation metrics.
                                    For certain protocols, these can also be the raw
                                    metrics computed at the end of the batch.
        """
        ...

    @abstractmethod
    def on_epoch_begin(self, epoch: int):
        """Called at the start of an epoch.
        Subclasses should override for any actions to run.
        Args:
            epoch (int):            The epoch number that is starting.
        """
        ...

    @abstractmethod
    def on_epoch_end(self, epoch, metrics: Dict[str, TensorLike]):
        """Called at the end of an epoch during a training loop.
        Subclasses should override for any actions to run.
        Args:
            epoch (int):            The epoch number that is finalized.
            metrics (dict):         The metrics returned by the protocol at the end
                                    of the epoch. These are generally the running
                                    averages over all batches in the epoch. The
                                    metrics are always reset at the beginning of
                                    the next epoch.
        """
        ...

    @abstractmethod
    def on_train_begin(self):
        """Called at the beginning of training and validation cycle.
        Subclasses should override for any actions to run.
        """
        ...

    @abstractmethod
    def on_train_end(self, metrics: Dict[str, TensorLike]):
        """Called at the end of training.
        Subclasses should override for any actions to run.
        Args:
            metrics (dict):     The metrics, training as well the evaluation metrics
                                returned by the protocol at the end of the training.
        """
        ...

    @abstractmethod
    def on_eval_begin(self):
        """Called at the beginning of evaluation or validation.
        Subclasses should override for any actions to run.
        """
        ...

    @abstractmethod
    def on_eval_end(self, metrics: Dict[str, TensorLike]):
        """Called at the end of evaluation or validation.
        Subclasses should override for any actions to run.
        Args:
            metrics (dict):     The evaluation metrics returned by the protocol at
                                the end of current evaluation.
        """
        ...

class CallbackGroup:
    """Container abstracting a sequence of callbacks."""

    def __init__(self, callbacks: Sequence[Callback]):
        """Creates a group  for `Callback` instances.
        This object wraps a sequence of `Callback` instances, making it possible
        to call them all at once via a single endpoint
        (e.g. `cb_group.on_epoch_end(...)`).
        Args:
          callbacks: A sequence of `Callback` instances. The callback methods
                     methods of these instances will be called in the order
                    they appear in this sequence.

        """
        self.callbacks = tf.nest.flatten(callbacks) if callbacks else []
        self.callbacks_hooks = self._extract_hooks_presence()

    def _extract_hooks_presence(self):
        """Extracts the hooks from the callbacks.
        Returns:
            A dictionary mapping hook names to lists of callbacks that implement
            them.
        """

        def _extract(hook_name):
            return [
                getattr(cb, hook_name) is not getattr(Callback, hook_name)
                for cb in self.callbacks
            ]

        hooks = {
            "on_train_batch_begin": _extract("on_train_batch_begin"),
            "on_train_batch_end": _extract("on_train_batch_end"),
            "on_eval_batch_begin": _extract("on_eval_batch_begin"),
            "on_eval_batch_end": _extract("on_eval_batch_end"),
            "on_epoch_begin": _extract("on_epoch_begin"),
            "on_epoch_end": _extract("on_epoch_end"),
            "on_train_begin": _extract("on_train_begin"),
            "on_train_end": _extract("on_train_end"),
            "on_eval_begin": _extract("on_eval_begin"),
            "on_eval_end": _extract("on_eval_end"),
        }
        return hooks

    def setup(self, trainer: Trainer, protocol: Protocol):
        """Sets up each callback in the group. Takes care of calling the `setup`
        method of each callback in the group.

        Args:
            trainer (Trainer):      The trainer instance that is using this callback
                                    group.
            protocol (Protocol):    The protocol instance that is using this callback
                                    group.

        """
        for callback in self.callbacks:
            callback.setup(trainer, protocol)

    def on_train_batch_begin(self, step: int):
        """Called at the beginning of a batch processing in the training loop.
         Takes care of calling the `on_train_batch_begin` method of each callback
         in the group.

        Args:
            step (int):     The index of the batch that is about to be processed.
        """
        for callback, is_implemented in zip(
            self.callbacks, self.callbacks_hooks["on_train_batch_begin"]
        ):
            callback.on_train_batch_begin(step) if is_implemented else None

    def on_train_batch_end(self, step: int, metrics: Dict[str, TensorLike]):
        """Called at the end of a batch processing in the training loop.
        Takes care of calling the `on_train_batch_end` method of each callback
        in the group.

        Args:
            step (int):     The index of the batch that has just been processed.
            metrics (dict): The metrics returned by the protocol at the end of the
                            training step.
        """
        for callback, is_implemented in zip(
            self.callbacks, self.callbacks_hooks["on_train_batch_end"]
        ):
            callback.on_train_batch_end(step, metrics) if is_implemented else None

    def on_eval_batch_begin(self, step: int):
        """Called at the beginning of a batch processing in the evaluation loop.
        Takes care of calling the `on_eval_batch_begin` method of each callback
        in the group.

        Args:
            step (int):     The index of the batch that is about to be processed.
        """
        for callback, is_implemented in zip(
            self.callbacks, self.callbacks_hooks["on_eval_batch_begin"]
        ):
            callback.on_eval_batch_begin(step) if is_implemented else None

    def on_eval_batch_end(self, step: int, metrics: Dict[str, TensorLike]):
        """Called at the end of a batch processing in the evaluation loop.
        Takes care of calling the `on_eval_batch_end` method of each callback
        in the group.

        Args:
            step (int):         The index of the batch that has just
                                been processed.
            metrics (dict):     The metrics  at the end of the batch
                                processing during evaluation.
        """
        for callback, is_implemented in zip(
            self.callbacks, self.callbacks_hooks["on_eval_batch_end"]
        ):
            callback.on_eval_batch_end(step, metrics) if is_implemented else None

    def on_epoch_begin(self, epoch: int):
        """Called at the beginning of an epoch. Takes care of calling the
        `on_epoch_begin` method of each callback in the group.

        Args:
            epoch (int):   The index of the epoch that is about to start.

        """
        for callback, is_implemented in zip(
            self.callbacks, self.callbacks_hooks["on_epoch_begin"]
        ):
            callback.on_epoch_begin(epoch) if is_implemented else None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, TensorLike]):
        """Called at the end of an epoch. Takes care of calling the
        `on_epoch_end` method of each callback in the group.

        Args:
            epoch (int):    The index of the epoch that has just ended.
            metrics (dict): The metrics that have been computed at the
                            end of the epoch.
        """
        for callback, is_implemented in zip(
            self.callbacks, self.callbacks_hooks["on_epoch_end"]
        ):
            callback.on_epoch_end(epoch, metrics) if is_implemented else None

    def on_train_begin(self):
        """Called at the beginning of the training. Takes care of calling the
        `on_train_begin` method of each callback in the group.
        """
        for callback, is_implemented in zip(
            self.callbacks, self.callbacks_hooks["on_train_begin"]
        ):
            callback.on_train_begin() if is_implemented else None

    def on_train_end(self, metrics: Dict[str, TensorLike]):
        """Called at the end of the training. Takes care of calling the
        `on_train_end` method of each callback in the group.

        Args:
            metrics (dict): The metrics that have been finalized at the end
                             of the training.
        """
        for callback, is_implemented in zip(
            self.callbacks, self.callbacks_hooks["on_train_end"]
        ):
            callback.on_train_end(metrics) if is_implemented else None

    def on_eval_begin(self):
        """Called at the beginning of the evaluation. Takes care of calling the
        `on_eval_begin` method of each callback in the group.
        """
        for callback, is_implemented in zip(
            self.callbacks, self.callbacks_hooks["on_eval_begin"]
        ):
            callback.on_eval_begin() if is_implemented else None

    def on_eval_end(self, metrics: Dict[str, TensorLike]):
        """Called at the end of the evaluation. Takes care of calling the
        `on_eval_end` method of each callback in the group.

        Args:
            metrics (dict): The metrics that have been finalized at the end
                            of the evaluation.
        """
        for callback, is_implemented in zip(
            self.callbacks, self.callbacks_hooks["on_eval_end"]
        ):
            callback.on_eval_end(metrics) if is_implemented else None

    def __add__(self, other):
        if not isinstance(other, (Callback, CallbackGroup)):
            raise TypeError(
                "A CallbackGroup only accepts a Callback or "
                "CallbackGroup instance as its operand for addition."
            )
        if isinstance(other, Callback):
            group = CallbackGroup(self.callbacks + [other])
        else:
            group = CallbackGroup(self.callbacks + other.callbacks)
        return group
