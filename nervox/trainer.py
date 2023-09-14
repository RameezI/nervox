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
import json
import inspect
import contextlib
import importlib
import locale
import uuid
import logging
import numpy as np


# Set environment variable to silence tensorflow warnings/errors/info/debug.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "3")
""""
This doesn't work if tensorflow is imported before setting the env variable.
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""

import tensorflow as tf
from pathlib import Path
from nervox.callbacks.callbacks import Callback
from typing import Union, List, Tuple, Dict, Iterable, Type, Any

# graph analysis tools
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

# Bring in some core types/modules of nervox
from nervox.data import DataStream
from nervox.protocols import Protocol
from nervox.utils import ProgressBar, ModeProgressBar
from nervox.callbacks.callbacks import CheckPointer, ProgressParaphraser

from nervox.utils import (
    get_urid,
    make_spec_concrete,
    ComputeComplexity,
    VerbosityLevel,
)

# global settings
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
tf_logger = tf.get_logger()
tf_logger.setLevel(logging.ERROR)
tf.autograph.set_verbosity(10)  # put 10 for high verbosity


# nervox logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Aliases
CallbackList = tf.keras.callbacks.CallbackList
margin = 0


class Trainer:
    # TensorFlow Distribution Schemes Validated by the nervox trainer
    named_distribution_schemes = {
        "one_device_cpu": tf.distribute.OneDeviceStrategy(device="/cpu:0"),
        "one_device_gpu": tf.distribute.OneDeviceStrategy(device="/gpu:0"),
        "mirrored": tf.distribute.MirroredStrategy(),
    }

    def __init__(
        self,
        train_stream: Union[DataStream, tf.data.Dataset],
        eval_stream: Union[None, DataStream, tf.data.Dataset] = None,
        logs_dir: Union[None, str] = None,
        name: Union[str, None] = None,
        run_id: Union[None, str] = None,
        distributor: Union[None, tf.distribute.Strategy] = None,
        ckpt_opts: Union[None, Dict[str, Any]] = None,
        # export_opts: Union[None, Dict[str, Any]] = None,
    ):
        """
        Args:
            train_stream:      A `DataStream` or `tf.data.Dataset` object, which provides the training data.
            eval_stream:       A `DataStream` or `tf.data.Dataset` object, which provides the evaluation data.
            logs_dir:          The directory where the logs will be stored. If not provided, the logs will be stored in a default location.
                                default: ~/tensorflow_logs
            name:              The pursuit name, this defines a collection of training runs. 
            run_id:            Unique identifier for the run. If not provided, a timestamp will be generated as UUID.
            distributor:       The distribution strategy to be used for training. If not provided, the training will be done on a single device.
                               Refer to the documentation on [distribution strategies](https://www.tensorflow.org/guide/distributed_training)
                               for further details.
            ckpt_opts:         Checkpoint options for the trainer. User can supply a dictionary of desired options.
        """

        default_log_dir = os.path.join(os.path.expanduser("~"), "tensorflow_logs")
        self._logs_dir = default_log_dir if logs_dir is None else logs_dir
        self._name = name
        self._run_id = get_urid() if run_id is None else run_id
        self._models = {}

        # convert the streams to a canonical form
        if isinstance(train_stream, tf.data.Dataset):
            train_stream = DataStream(train_stream, split="train")

        if isinstance(eval_stream, tf.data.Dataset):
            eval_stream = DataStream(eval_stream, split="test")

        # examples count
        self._train_samples_count = np.nan
        self._eval_samples_count = np.nan
        self._sample_done_in_epoch = 0
        self._data_streams = self._connect_streams(train_stream, eval_stream)
        self._epoch: tf.Variable = tf.Variable(0, trainable=False, dtype=tf.int64)

        try:
            self._distributor_scope = (
                distributor.scope()
                if distributor is not None
                else contextlib.suppress()
            )
        except Exception as e:
            logger.error("\n\nFailed to create the requested distribution context!\n")
            raise

        self._checkpointer_config: Union[Dict[str, Any], None] = None
        # self._export_config: Union[Dict[str, Any], None] = None
        self.predict_complexity: Union[None, ComputeComplexity] = None

        self.checkpointer_config = ckpt_opts
        # self.export_config = export_opts

        # Tensorboard Summaries
        self._train_summarizer: Union[None, tf.summary.SummaryWriter] = None
        self._eval_summarizer: Union[None, tf.summary.SummaryWriter] = None

        # Progress bar
        self._verbose = VerbosityLevel.UPDATE_AT_BATCH
        self._progress_bar = ProgressBar()

        # create run_dir if it does not exit
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def verbosity(self):
        return self._verbose

    @property
    def train_summarizer(self):
        return self._train_summarizer

    @property
    def eval_summarizer(self):
        return self._eval_summarizer

    @property
    def logs_dir(self):
        return self._logs_dir

    @property
    def name(self) -> str:
        return str(self)

    @property
    def run_id(self):
        return self._run_id

    @logs_dir.setter
    def logs_dir(self, value):
        self._logs_dir = value

    @name.setter
    def name(self, value):
        self._name = value

    @run_id.setter
    def run_id(self, value: str):
        self._run_id = value

    @property
    def run_dir(self):
        run_dir = None
        if self.name is not None:
            run_dir = Path(self.logs_dir, self.name, self.run_id)
        return run_dir

    @property
    def checkpoint_dir(self) -> os.PathLike:
        checkpoint_dir = None
        if self.run_dir is not None:
            checkpoint_dir = Path(self.run_dir, "checkpoints")
        return checkpoint_dir

    @property
    def epoch(self) -> int:
        return self._epoch.value().numpy()

    @property
    def train_samples_count(self) -> int:
        return self._train_samples_count

    @property
    def eval_samples_count(self) -> int:
        return self._eval_samples_count

    @property
    def samples_done(self) -> int:
        return self._sample_done_in_epoch

    @property
    def tensorboard_dir(self) -> str:
        tb_summaries_dir = None
        if self.run_dir is not None:
            tb_summaries_dir = os.path.join(self.run_dir, "summaries")
        return tb_summaries_dir

    @property
    def summary_writers(self):
        uid = str(uuid.uuid4())[:8]
        if self._train_summarizer is None:
            self._train_summarizer = tf.summary.create_file_writer(
                str(Path(self.tensorboard_dir, "train")), name=f"spin_{uid}"
            )
        if self._eval_summarizer is None:
            self._eval_summarizer = tf.summary.create_file_writer(
                str(Path(self.tensorboard_dir, "val")), name=f"spin_{uid}"
            )
        return {"train": self._train_summarizer, "test": self._eval_summarizer}

    @property
    def models(self):
        keys = list(self._models.keys())
        models = None if not keys else self._models
        return models

    # Configure the underlying Checkpointer Callback
    @staticmethod
    def _configure_ckpt(user_config: Union[None, dict] = None):
        user_config = {} if user_config is None else user_config
        ckpt_config = {
            "monitor": "loss/val",
            "silent": False,
            "keep_best": True,
            "max_to_keep": 3,
            "mode": "auto",
        }
        ckpt_config.update(user_config)
        return ckpt_config

    @property
    def checkpointer_config(self) -> Dict:
        if self._checkpointer_config is None:
            self._checkpointer_config: Dict[str, Any] = self._configure_ckpt(None)
        return self._checkpointer_config

    @checkpointer_config.setter
    def checkpointer_config(self, value: Union[Dict, None]):
        self._checkpointer_config = self._configure_ckpt(value)

    # Connect Data Streams with the trainer
    def _connect_streams(
        self, train: DataStream, evaluate: DataStream = None
    ) -> Dict[str, DataStream]:
        """
        This Connects the data_streams with the trainer. The data streams are mutable objects, therefore, any changes
        to data streams in the outer scope will be reflected within the trainer as well and vice-versa.
        Args:
            train: The 'DataSteam' object that will be used for the training purpose
            evaluate: The 'DataSteam' object that will be used for the validation purpose
        Returns: None
        """
        self._train_samples_count = train.examples_count
        self._eval_samples_count = (
            evaluate.examples_count if evaluate is not None else np.nan
        )
        return {"train": train, "evaluate": evaluate}

    @staticmethod
    def assert_inception_sanity(matched_keys, module_name):
        no_match_msg = (
            f"\n Unable to match an inception_class in the `{module_name}` module:\n"
            f" In case there are multiple models (tf.keras.Model) in the module, the top level"
            f" class, i.e. the inception class, must have the same name\n"
            f" as the module itself (excluding the underscores, case-insensitive)."
            f" When a different name is desired, please provide it explicitly via\n"
            f" the `inception_class` keyword argument."
        )

        multi_match_msg = (
            f"\nUnable to determine a unique inception_class:\n"
            f"Multiple top level classes identified: {matched_keys}\n"
            f"A possible failure scenario is the use of the case-sensitive class"
            f" names in the module, if the error persists,\nplease provide the name"
            f" of the inception class explicitly via the `inception_class` keyword argument."
        )

        if not matched_keys:
            raise RuntimeError(no_match_msg)

        if len(matched_keys) > 1:
            raise RuntimeError(multi_match_msg)

    def push_module(
        self,
        module: Tuple[str, Type[tf.keras.Model]],
        config: Union[dict, None] = None,
        alias: Union[None, str] = None,
        pkg: Union[None, str] = None,
        inception_class: Union[None, str] = None,
        overwrite: bool = False,
    ) -> None:
        """
        This pushes a model to the trainer. Multiple models can be added by repeated calls to the method, in which case
        a dictionary of models is maintained by the trainer. The trainer forwards these models to the protocol for training.

        Args:
            model:              A type derived from nervox.Module or a string referring to the module that implements
                                the subclassed type.
            config:             Configuration required to instantiate the `module`.
            alias:              The key under which the module will be saved, the module will be renamed accordingly
                                The module is accessible in the protocol under this alias.
            inception_class:    This is used when instead of a `type`, a module name is provided that contains the module.
                                to be instantiated. The trainer then need to correctly instantiate the top level class
                                in the python module, which is  determined by the `inception_class` argument.
                                In case the value is `None` the trainer tries to determine the class itself:
                                    - When a single class in present in the module it is selected as inception class.
                                    - When more than classes are in the module, the class with the same name as the
                                      module file (camelcase) is searched for and selected as the inception class.
                                When a type is provided instead, the `inception_class` argument is ignored.
            overwrite:          When True, trainer overwrites any existing model with the same alias.

        Returns: None

        Raises:
            RuntimeError: When the inception class cannot be determined.
            RuntimeError: When multiple inception classes are wrongly detected in the module.
                          This can happen when the module contains multiple classes and the
                          classes with the same name but different cases are present.
        """

        with self._distributor_scope:
            config = {} if config is None else config

            def get_unique_key(models: dict, candidate_key: str):
                for post_fix in range(1, len(models) + 1):
                    if candidate_key not in models.keys():
                        break
                    else:
                        candidate_key = f"{candidate_key}_{post_fix}"
                return candidate_key

            if isinstance(module, str):
                pkg = "nervox.modules" if pkg is None else pkg
                module = importlib.import_module(f".{module}", package=pkg)
                search_term = module if inception_class is None else inception_class
                class_members = inspect.getmembers(module, inspect.isclass)
                class_members = dict(
                    filter(
                        lambda item: issubclass(item[1], tf.keras.Model), class_members
                    )
                )

                if inception_class is None:
                    matched_keys = [
                        key
                        for key in class_members.keys()
                        if key.lower() == search_term.replace("_", "").lower()
                        or len(class_members) == 1
                    ]
                else:
                    matched_keys = [
                        key for key in class_members.keys() if key == search_term
                    ]

                self.assert_inception_sanity(matched_keys, module)
                inception_class = class_members[matched_keys[0]]
            else:
                inception_class = module

            if not (
                isinstance(inception_class, type(tf.keras.Model))
                and issubclass(inception_class, tf.keras.Model)
            ):
                raise TypeError(
                    "The model cannot be added due to type mismatch, please"
                    " provide a class derived from `tf.keras.Model`.\n"
                    f"Expected: {tf.keras.Model}\n"
                    f"Received: {inception_class}\n"
                )

            module = inception_class(**config)
            default_alias = (
                get_unique_key(self._models, "model") if not overwrite else "model"
            )
            alias = default_alias if alias is None else alias

            if not overwrite:
                assert alias not in self._models.keys(), (
                    f"Alias reuse!\n"
                    f"The model with alias=`{alias}` already exists. Use `force` option if an overwrite is indented"
                )

            if module.built:
                logger.warning(
                    """ The model name can not be changed recursively! The model is already built and
                 weights are assigned names at the inception phase. The weights name are not consistent with the
                 provided alias, which could lead to errors and confusion!"""
                )
            module._name = alias
            self._models.update({alias: module})

    def _compile_protocol(self, protocol: Protocol) -> bool:
        """
        This method compiles the protocol. The protocol is compiled once during training,
        triggered by the spin request. The protocol compilation perform following tasks:
            1- Instantiate loss functions, metrics and optimizers, grouped in organization units of `objectives`.
            2- Links instantiated models from the trainer to the protocol.
            3- Caches the the distribution strategy for the protocol.
            4- Compiles the train/validate/predict steps into a tensorflow graph, if eager execution is `False`,

        Args:
            protocol: The protocol to be compiled

        Returns: True if the compilation is successful, otherwise raises an exception.
        """
        return_status = False
        with self._distributor_scope:
            if not isinstance(protocol, Protocol):
                raise TypeError(
                    f"The protocol must be a subclass of {Protocol.__name__}"
                    f"Expected: {Protocol.__name__}"
                    f"Received: {type(protocol).__name__}"
                )
            if not self._models:
                raise RuntimeError(
                    f"The trainer has no computational graph, unable to compile the strategy.\n"
                    f"You can add one or more models to the trainer via add_model method:\n"
                    f"{self.push_module.__doc__}"
                )
            try:
                protocol.compile(self._models)

            except Exception as e:
                logger.error(str(e))
                logger.error("The strategy compilation failed!")

            else:
                return_status = True

        return return_status

    def _configure_callbacks(
        self, callback_list: List[Callback], protocol: Protocol, verbose: VerbosityLevel
    ):
        """
        This method sets up the callbacks for the train/eval spin, it also installs default
        callbacks to the user supplied list and drops default callbacks if a relevant callback
        is found in the user callbacks list.

         Args:
            callback_list:      A list of user supplied callbacks.
            protocol:           The compiled protocol to be used for training.
            verbose:            The verbosity of the progress reporting callback.

        Returns: None
        """
        if callback_list:
            logger.warning(
                "\n\nExternal supply of a callback list is not yet"
                " fully tested, please make sure your callback does"
                " not conflict with the defaults "
            )

        callback_list.append(
            CheckPointer(
                self.checkpoint_dir,
                **self.checkpointer_config,
                progress_bar=self._progress_bar,
            )
        )
        # callback_list.append(TensorBoardLogger(self.tensorboard_dir, mode='batch',
        #                                                  produce_graph=True))
        # callback_list.append(Exporter(self.checkpoint_dir, **self._export_config))
        callback_list.append(ProgressParaphraser(self._progress_bar, self._verbose))

        for cb in callback_list:
            cb.trainer = self
            cb.protocol = protocol

        return callback_list

    def _load_checkpoint(self, checkpoint, silent=True):
        latest_ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
        status = checkpoint.restore(latest_ckpt)
        logger.info(f"Restored from: \n{latest_ckpt} \n") if not silent else None
        return status

    def spin(
        self,
        protocol: Protocol,
        max_epochs: int = 100,
        callback_list: Union[None, List[Callback]] = None,
        warm_start: bool = True,
        run_eagerly: bool = False,
        verbose: VerbosityLevel = VerbosityLevel.UPDATE_AT_BATCH,
        **kwargs,
    ) -> None:
        """
        When spin method is invoked: it compiles the protocol and executes the training/evaluation tasks
        as per the given protocol.

        Args:
            protocol:           This is the protocol that defines the training/evaluation loops and the logic.
                                The protocol must be a subclass of `Protocol` class and must be configured with
                                to contain one or more `Objective` instances.
            max_epochs:         The maximum number of epochs to train the model.
            callback_list:      A list of user supplied callbacks.
            warm_start:         Whether to load the latest checkpoint from the checkpoint directory.
            run_eagerly:        Whether to run the training/evaluation loops eagerly or not.
            serving_signature:  The signature of the serving function.
            verbose:            The verbosity level of the trainer progress, it can be one of the following:
                                    VerbosityLevel.SILENT, VerbosityLevel.UPDATE_AT_BATCH, VerbosityLevel.UPDATE_AT_EPOCH
                                    Default: VerbosityLevel.UPDATE_AT_BATCH
            **kwargs:           Additional keyword arguments.

        Returns: None

        """

        with self._distributor_scope:
            self._compile_protocol(protocol)

        if max_epochs < 1:
            raise ValueError(
                "Invalid value for `max_epochs`."
                "The `max_epochs` must be a positive integer.\n"
                f"max_epoch == {max_epochs}"
            )

        if not protocol.compiled:
            raise RuntimeError(
                f"The training/evaluation protocol is not in place yet!"
                f"It might be that compilation was unsuccessful."
            )

        self._epoch.assign(0)
        self._verbose = verbose

        assert self.name, self.run_id
        callback_list = [] if callback_list is None else callback_list
        callback_list = self._configure_callbacks(callback_list, protocol)
        callbacks = CallbackList(callback_list)

        do_validation = False if self._data_streams["evaluate"] is None else True
        eval_stream = (
            self._data_streams["evaluate"]
            if do_validation
            else self._data_streams["train"]
        )

        self.write_config_to_disk()

        # Printing the configuration
        print(
            f"\n{'':-<{self._progress_bar.terminal_size - margin}}"
        ) if verbose not in [VerbosityLevel.KEEP_SILENT] else None
        print(f"\n{self.name}/{self.run_id}:")
        print(json.dumps(self.to_json(), indent=4)) if verbose not in [
            VerbosityLevel.KEEP_SILENT
        ] else None

        try:
            # Profile the underlying predict method to estimate FLOPS & Parameters
            silent = verbose == VerbosityLevel.KEEP_SILENT
            self._profile_inference_graph(protocol, eval_stream, silent=silent)
        except NotImplementedError:
            logger.warning(
                f"\n{'Lacking a predict step':-^35}\n"
                "The strategy does not provide a `call` method,"
                " skipping inference graph analysis!\n"
            )

        # Attempt a warm restart if checkpoint exists and warm_start is set to true
        if warm_start and os.path.isfile(Path(self.checkpoint_dir, "checkpoint")):
            checkpoint = tf.train.Checkpoint(
                epoch=self._epoch, **self._models, objectives=protocol.objectives
            )
            try:
                silent = verbose == VerbosityLevel.KEEP_SILENT
                ckpt_load_status = self._load_checkpoint(checkpoint, silent=silent)
                ckpt_load_status.assert_existing_objects_matched()
            except AssertionError as e:
                self._epoch.assign(0)
                logging.error(str(e))
                raise AssertionError(
                    "Checkpoint load failure!\n"
                    f"checkpoint directory: {self.checkpoint_dir}"
                )

        # Initial evaluation
        if not kwargs.pop("skip_initial_evaluation", False):
            self._progress_bar.mode = ModeProgressBar.EVAL_ONLY
            split_name = eval_stream.split_name
            print(f"Evaluation on the `{split_name}` split:") if verbose not in [
                VerbosityLevel.KEEP_SILENT
            ] else None

            samples_init_eval = (
                self._eval_samples_count if do_validation else self._train_samples_count
            )
            callbacks.on_test_begin(logs={"samples/eval": samples_init_eval})
            logs_init_eval = protocol.evaluate(
                eval_stream, callbacks, run_eagerly=run_eagerly
            )
        else:
            logs_init_eval = {}
        #
        callbacks.on_test_end(logs_init_eval)
        self._progress_bar.mode = ModeProgressBar.DEFAULT

        # sanity check : tests if the ckpt is fully consumed at this stage.
        # if warm_start and os.path.isfile(Path(self.checkpoint_dir, 'checkpoint')):
        #     ckpt_load_status.assert_consumed()

        print(
            f"\n{'':-^{self._progress_bar.terminal_size - margin}}"
        ) if verbose not in [VerbosityLevel.KEEP_SILENT] else None

        callbacks.on_train_begin(logs=logs_init_eval)
        if self._epoch.value() >= max_epochs:
            print(
                f"Exiting ...\n"
                f"The maximum epochs have already been reached.\n"
                f"Restored epoch == {int(self._epoch.value())}\n"
                f"Allowed epochs == {max_epochs}\n"
            ) if verbose not in [VerbosityLevel.KEEP_SILENT] else None
            callbacks.on_train_end()
        else:
            start_epoch = int(self._epoch.assign_add(1).value())
            # Training - transverse through epochs executing the protocol
            for epoch in range(start_epoch, max_epochs + 1):
                self._epoch.assign(epoch)
                callbacks.on_epoch_begin(epoch)

                # training
                logs_train = protocol.train(
                    self._data_streams["train"], callbacks, run_eagerly=run_eagerly
                )
                self._train_samples_count = (
                    logs_train.get("samples_done/train", np.nan)
                    if np.isnan(self._train_samples_count)
                    else self._train_samples_count
                )

                # evaluation
                if do_validation:
                    callbacks.on_eval_begin()  # Evaluation begin callbacks

                    logs_eval = protocol.evaluate(
                        self._data_streams["evaluate"],
                        callbacks,
                        run_eagerly=run_eagerly,
                    )
                    self._eval_samples_count = (
                        logs_eval.get("samples_done/val", np.nan)
                        if np.isnan(self._eval_samples_count)
                        else self._eval_samples_count
                    )
                    callbacks.on_test_end()
                else:
                    # when validation data is absent, train metrics are used as val metrics.
                    logs_eval = {
                        k.replace("/train", "/val"): v for k, v in logs_train.items()
                    }
                callbacks.on_epoch_end(
                    epoch,
                    logs={**logs_train, **logs_eval},
                )
            callbacks.on_train_end()

            print(
                f"\n{'':-^{self._progress_bar.terminal_size - margin}}\n"
            ) if verbose not in [VerbosityLevel.KEEP_SILENT] else None

    def to_json(self):
        config = {
            "logs_dir": self._logs_dir,
            "name": self.name,
            "run_id": self.run_id,
            "data_streams": {
                key: getattr(value, "params") if value is not None else None
                for key, value in self._data_streams.items()
            },
            "models": {
                key: getattr(value, "params") if value is not None else None
                for key, value in self.models.items()
            },
            "checkpointer": self._checkpointer_config,
        }
        return config

    def write_config_to_disk(self):
        config_file = Path(os.path.join(self.run_dir, "params.json"))
        with open(config_file, "w") as cf:
            json.dump(self.to_json(), cf, indent=4, sort_keys=False)

    def _profile_inference_graph(
        self, protocol: Protocol, data_stream: DataStream, silent=True
    ):
        elements_spec = data_stream.as_dataset().element_spec
        if not isinstance(elements_spec, Iterable):
            elements_spec = make_spec_concrete(elements_spec)
        elif isinstance(elements_spec, dict):
            elements_spec = dict(map(make_spec_concrete, elements_spec.items()))
        elif isinstance(elements_spec, list):
            elements_spec = dict(map(make_spec_concrete, elements_spec))
        else:
            raise ValueError("The element specs received are not in supported format")
        predict_function = tf.function(
            protocol.predict_step, input_signature=[elements_spec]
        )

        graph = predict_function.get_concrete_function().graph
        profile_op = ProfileOptionBuilder(ProfileOptionBuilder.float_operation())
        profile_opts = (
            profile_op.with_empty_output().build() if silent else profile_op.build()
        )
        flops_info = profile(graph, options=profile_opts)
        total_parameters = int(np.sum([np.prod(var.shape) for var in graph.variables]))
        flops = flops_info.total_float_ops
        trainable_parameters = int(
            np.sum([np.prod(var.shape) for var in graph.trainable_variables])
        )
        non_trainable_parameters = total_parameters - trainable_parameters
        self.predict_complexity = ComputeComplexity(
            flops=flops,
            trainable_parameters=trainable_parameters,
            non_trainable_parameters=non_trainable_parameters,
        )
        if not silent:
            print(f"\n{'':-^{self._progress_bar.terminal_size - margin}}")
            print(f"Compute Complexity [Prediction]:\n")
            print(self.predict_complexity)
            print(f"{'':-^{self._progress_bar.terminal_size - margin}}")

    def __str__(self):
        name = "unnamed" if self._name is None else self._name
        return name


if __name__ == "__main__":
    trainer = Trainer(DataStream("cifar10"), name="cifar10Classifier")
    logger.info(str(trainer))
