"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import inspect
import logging
from pathlib import Path
from typing import Collection
import tensorflow as tf
from tensorflow import distribute
from typing import Any, Dict, List, Union, final
from nervox.core.objective import Objective
from nervox.data import DataStream
from nervox.utils import snake_to_camel, Signatures

# Aliases
# CallbackList = tf.keras.callbacks.CallbackList


class Protocol(tf.Module):
    ClusterCoordinator = distribute.experimental.coordinator.ClusterCoordinator

    # def __init_subclass__(cls):
    #     super().__init_subclass__()
    #     _subclass_init = cls.__init__

    #     def init_protocol_subclass(self, configurator=None, **kwargs):
    #         super(cls, self).__init__(configurator=configurator)
    #         _subclass_init(self, **kwargs)

    #     cls.__init__ = init_protocol_subclass

    def __init__(self, configurator=None):
        name = snake_to_camel(self.__module__, splitter=".")
        super(Protocol, self).__init__(name=name)
        self._cluster_coordinator: Union[None, Protocol.ClusterCoordinator] = None
        self._modules: Union[None, Dict[str, tf.keras.Mode]] = None
        self._objectives: Union[None, Dict[str, Objective]] = None
        self._metrics: List[tf.keras.metrics.Metric] = []
        self._distributor = None

        # boolean flags
        self._base_strategy_initialized: bool = True
        self._run_eagerly: bool = False
        self._is_compiled: bool = False

        # cached step_tf functions
        self.train_step_tf = None
        self.evaluate_step_tf = None
        self.predict_step_tf = None

        self.objective_configurator = (
            type(self).objective_configurator if configurator is None else configurator
        )

    @property
    def is_initialized(self):
        try:
            self._base_strategy_initialized
        except AttributeError:
            raise RuntimeError(
                "It looks like you are subclassing `Strategy` and forgot to call"
                "`super(YourClass, self).__init__()`.\n"
                "Always start your subclassed strategy with this line."
            )
        return self._base_strategy_initialized

    def compiled(self):
        return self._is_compiled

    @property
    def cluster_coordinator(self):
        return self._cluster_coordinator

    @property
    def metrics(self):
        self._metrics = [
            metric
            for objective in self.objectives.values()
            for metric in objective.metrics
        ]
        return self._metrics

    @property
    def objective(self):
        objective: Union[None, Objective] = None
        if isinstance(self._objectives, dict) and len(self._objectives) > 1:
            raise AttributeError(
                f"Invalid use of attribute objective. This property is only populated in "
                f"case of a single objective, while multiple objectives were spotted.\n "
                f" Please make use of `objectives` property instead \n"
                f" Spotted objectives: {list(self._objectives.keys())}"
            )
        elif isinstance(self.objectives, dict) and len(self._objectives) == 1:
            objective = [value for value in self._objectives.values()][0]
        return objective

    @property
    def objectives(self):
        return self._objectives

    @property
    def module(self):
        module: Union[None, tf.keras.Model] = None
        if isinstance(self._modules, dict) and len(self._modules) > 1:
            raise AttributeError(
                f"Invalid use of attribute module. This property is only populated in case of a"
                f"single module setting. While, multiple modules setting is spotted.\n"
                f" Please make use of `modules` instead \n"
                f"Spotted modules keys:{list(self._modules.keys())}"
            )

        elif isinstance(self._modules, dict) and len(self._modules) == 1:
            module = [value for value in self._modules.values()][0]
        return module

    @property
    def modules(self):
        return self._modules
    
    def reset_metrics(self):
        if self._metrics:
            [metric.reset() for metric in self._metrics]

    def _set_modules(self, value: Union[Dict[str, tf.keras.Model], tf.keras.Model]):
        modules = value if isinstance(value, dict) else {"module": value}
        self._assert_type_conformity(modules, expected_type=tf.keras.Model)
        self._modules = modules

    def _set_objectives(self, value: Union[Dict[str, Objective], Objective]):
        objectives = value if isinstance(value, dict) else {"objective": value}
        self._assert_type_conformity(objectives, expected_type=Objective)
        self._objectives = objectives

    @staticmethod
    def _assert_type_conformity(collection: Dict[str, Any], expected_type: type):
        faulty_keys = filter(
            lambda value: not isinstance(value, expected_type), collection.values()
        )
        if list(faulty_keys):
            faulty_types = [
                list(map(lambda x: x.__name__, map(type, v)))
                for k, v in collection.items()
                if k in faulty_keys
            ]
            faulty_formulations = [
                f"{k}:{v}" for k, v in zip(faulty_keys, faulty_types)
            ]
            raise ValueError(
                f"Cannot set objectives!\n"
                f"One or more objective does match the expected type restrictions: \n"
                f"Expected: An objective of type: `{expected_type}`\n"
                f"Received: {faulty_formulations}"
            )

    def _compile_compliance_check(self, modules):
        params = inspect.signature(self.objective_configurator).parameters.values()
        unsupported_kinds = [
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ]

        if any([param.kind in unsupported_kinds for param in params]):
            _unsupported_kinds_str = [str(kind) for kind in unsupported_kinds]
            raise TypeError(
                f"Unsupported arguments kind spotted in the `configure_objective`"
                f" method of the strategy class:{str(type(self).__name__)}.\n"
                f"The following kinds are not supported:\n"
                f"{_unsupported_kinds_str}"
            )

        params = list(filter(lambda x: x.kind not in unsupported_kinds, params))
        required_named_args = [
            param.name for param in params if param.default == inspect.Parameter.empty
        ]

        protocol_module: List[str] = str(self.__module__)
        if (
            isinstance(modules, dict)
            and all([isinstance(module, tf.keras.Model) for module in modules.values()])
            and all([isinstance(key, str) for key in modules.keys()])
        ):
            if not all([arg in modules.keys() for arg in required_named_args]):
                raise RuntimeError(
                    f"Unable to populate modules!\n"
                    f"The modules dictionary handed over to the compile function does not comply with"
                    f" the `objective_configurator` signature of the `{protocol_module}` module.\n"
                    f"The method must accept (as arguments) the same names as the keys in the module"
                    f" dictionary of the trainer instance.\n"
                    f"A subset of these alias also constitutes a valid argument list!\n"
                    f"The discrepancy:\n"
                    f"`configure` signature: {inspect.signature(self.objective_configurator)}\n"
                    f"module keys: {list(modules.keys())}"
                )
        else:
            raise ValueError(
                "The first argument of the compile must be a dictionary of tf.keras.Model(s)"
            )
        return required_named_args

    def get_global_batch_size(self, batch, batch_idx=0):
        distributor = self._distributor
        size = set([tf.shape(k)[batch_idx].numpy() for k in batch.values()])
        if not len(size) == 1:
            raise RuntimeError(
                f"Cannot estimate batch size!"
                f" Spotted different values at the batch index for different features,"
                f" all features must have the same dimension at the batch index.\n"
                f" Found {len(size)} unique batch lengths {size} at index={batch_idx}"
            )
        else:
            # TODO: Apply correction based on the distributor!
            size = next(iter(size))
        return size

    def _enable_auto_reduction(self):
        if isinstance(self._, dict):
            for key in self._mo:
                self._objectives[key]._losses._allow_sum_over_batch_size = True
        else:
            self.module.loss._allow_sum_over_batch_size = True

    def compile(
        self,
        modules: Dict[str, tf.keras.Model],
    ):
        """
        The protocol compilation perform following tasks:
            1- Create loss functions, metrics and optimizers, grouped in organization units called `objectives`.
            2- Instantiate the `objective(s)` that the protocol must optimize through training.
            3- Links the instantiated modules from the trainer to the protocol.
            4- Caches the the distribution strategy for the protocol.
            5- If eager execution is `False`, compiles the train/validate/predict steps into a tensorflow graph.

        Args:
            modules:  A dictionary of instantiated modules to be linked to the protocol.
        """
        self._distributor = tf.distribute.get_strategy()
        required_args = self._compile_compliance_check(modules)
        required_modules = {
            key: module for key, module in modules.items() if key in required_args
        }

        self._set_modules(modules)  # Link modules from the outer scope with the protocol.
        self._set_objectives(
            self.objective_configurator(**required_modules)
        )  # set objective/objectives.

        # compiled_step functions for train and evaluate
        self.train_step_tf = tf.function(self.train_step)
        self.evaluate_step_tf = tf.function(self.evaluate_step)
        self.predict_step_tf = tf.function(self.predict_step)
        self._is_compiled = True

    @staticmethod
    def objective_configurator(
        **kwargs: tf.keras.Model,
    ) -> Union[Dict[str, Objective], Objective]:
        """
        Constructs a learning objective for the trainer. The method optionally accepts modules, using their aliases as
         keyword/named arguments. Any module that need modification/inspection can be listed as named argument in the
        `configure` method, using its alias registered with the trainer.
        Args:
            **kwargs: Accepts (zero or more) modules provided as keyword arguments using their aliases.
        Returns:
            A nervox.core.Objective or a named collection of such objectives.
        """
        raise NotImplementedError(
            "Lacking `objective_configurator` method!\n"
            "The  protocol does not provide an `objective_configurator` method.\n"
            "Please provide an `objective_configurator` for your protocol, which describes the objective(s) to be optimized.\n"
        )

    def train_step(self, input):
        raise NotImplementedError(
            """Lacking `train_step` definition!
            The training strategy does not provide a `train_step` method. A strategy without this method is unable
            be perform training/learning. Please provide a `train_step` method for your strategy, which describes
            a single step for training the underlying modules."""
        )

    def evaluate_step(self, input):
        raise NotImplementedError(
            """Lacking `evaluate_step` definition!
            The strategy does not provide an `evaluate_step` method. A strategy without this method is unable to
            determine the evaluation logic. Please provide an `evaluate_step` for your strategy, which describes
            a single step for evaluating the underlying module(s) on a single batch."""
        )

    def predict_step(self, input):
        raise NotImplementedError(
            """Lacking `predict_step` definition!
            The strategy does not provide a `predict_step` method. A strategy without this method is unable
            to determine the prediction logic. Please provide a `predict_step` method for your strategy,
            which describes a single step for predicting based on the the underlying trained module(s)."""
        )

    @final
    def train(
        self,
        dataset: DataStream,
        callbacks: CallbackList = CallbackList(),
        run_eagerly=False,
        postfix="train",
        **kwargs,
    ) -> Dict[str, any]:
        """The is the training loop for the protocol. It is responsible for training the underlying module(s).
        Args:
            dataset (DataStream): A stream of data batches.
            callbacks (CallbackList, optional): A list of callbacks to be . Defaults to CallbackList() an empty callback list.
            run_eagerly (bool, optional): Whether to run the training loop eagerly(True) or in graph-mode (False). Defaults to False.
            postfix (str, optional): Adds a suffix to each metric to identify them connected to training data. Defaults to "train".
        Returns:
            Dict[str, any]: A dictionary of metrics and progress variables at the end of the training.
        """

        examples_processed = 0
        self.reset_metrics()

        logs = {}
        for step, batch in enumerate(dataset):
            callbacks.on_train_batch_begin(step)
            if run_eagerly:
                self._distributor.run(self.train_step, args=[batch], kwargs=kwargs)
            else:
                self._distributor.run(self.train_step_tf, args=[batch], kwargs=kwargs)

            examples_processed += self.get_global_batch_size(batch)

            logs["samples_done"] = examples_processed
            logs.update(
                {f"{metric.name}/{postfix}": metric.result() for metric in self.metrics}
            )
            callbacks.on_train_batch_end(step, logs)
        return logs

    @final
    def evaluate(
        self,
        dataset: DataStream,
        callbacks: CallbackList = CallbackList(),
        run_eagerly=False,
        postfix="val",
        **kwargs,
    ) -> Dict[str, any]:
        """This is the evaluation loop for the protocol. It is responsible for evaluating the underlying modules.
        Args:
            dataset (DataStream): A stream of data batches.
            callbacks (CallbackList, optional): A list of callbacks to be . Defaults to CallbackList() an empty callback list.
            run_eagerly (bool, optional): Whether to run the training loop eagerly(True) or in graph-mode (False). Defaults to False.
            postfix (str, optional): Adds a suffix to each metric to identify them connected to validation/evaluation datastream. Defaults to "val".

        Returns:
            Dict[str, any]: A dictionary of metrics and progress variables at the end of the evaluation.
        """

        examples_processed = 0
        self.reset_metrics()

        logs = {}
        for step, batch in enumerate(dataset):
            callbacks.on_test_batch_begin(step)
            if run_eagerly:
                self._distributor.run(self.evaluate_step, args=[batch], kwargs=kwargs)
            else:
                self._distributor.run(
                    self.evaluate_step_tf, args=[batch], kwargs=kwargs
                )

            examples_processed += self.get_global_batch_size(batch)

            logs["samples_done"] = examples_processed
            logs.update(
                {f"{metric.name}/{postfix}": metric.result() for metric in self.metrics}
            )
            callbacks.on_test_batch_end(step, logs)
        return logs
    
    def __str__(self):
        return self.name