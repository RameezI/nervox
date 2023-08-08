# Copyright © 2023 Rameez Ismail - All Rights Reserved
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
#
# Author(s): Rameez Ismail
# Email(s):  rameez.ismaeel@gmail.com


"""
This module contains all auxiliary functions for the Nervox framework.
"""

import sys
import os
import io
import re
import json
import datetime
import logging
import contextlib
import inspect
import argparse
import io
import numpy as np
import tensorflow as tf
import functools
from functools import wraps
from dataclasses import dataclass
from pathlib import Path
from functools import partial
import tensorflow as tf
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Union, Tuple, Callable, Dict, Any
from types import FunctionType
from enum import Enum

from nervox.utils.types import *
from tensorflow.python.keras.utils import tf_utils
from .serializers import SerializerRegistry


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_SPATIAL_CHANNELS_FIRST = re.compile("^NC[^C]*$")
_SPATIAL_CHANNELS_LAST = re.compile("^N[^C]*C$")
_SEQUENTIAL = re.compile("^((BT)|(TB))[^D]*D$")


class VerbosityLevel(Enum):
    KEEP_SILENT = "silent"
    UPDATE_AT_EPOCH = "epoch"
    UPDATE_AT_BATCH = "batch"


@dataclass(frozen=True)
class Signatures(tf.Module):
    train: tf.TensorSpec = None
    predict: tf.TensorSpec = None
    evaluate: tf.TensorSpec = None


@dataclass(frozen=True)
class ComputeComplexity:
    flops: int = np.nan
    trainable_parameters: int = np.nan
    non_trainable_parameters: int = np.nan

    @property
    def parameters(self):
        return self.trainable_parameters + self.non_trainable_parameters

    def __repr__(self):
        return (
            f"{'Total FLOPs':<20}: {self.flops:>20n}\n"
            f"{'Total Parameters':<20}: {self.parameters:>20n}\n"
            f"{'-> Trainable':<20}: {self.trainable_parameters:>20n}\n"
            f"{'-> Non-Trainable':<20}: {self.non_trainable_parameters:>20n}"
        )


# fmt:off
def base_parser(training_component=True, data_component=True) -> argparse.ArgumentParser:
    """
    General cmd-line args useful when working with experiments produced by nervox
    Using these standard args make it easier to write consistent launch files. Users
    are encouraged to use this base parser and add their own arguments on top of it.
    This will ensure that all experiments have a consistent set of arguments and
    that they can be launched in a consistent manner.

    Example:
        ```
        import argparse
        parser =  argparse.ArgumentParser(parents=[base_parser()])
        parser.add_argument('--my_new_arg', required=True, type=int, default=32,
                            help='This is a new argument that I want to add to my experiment')
        args = parser.parse_args()
        print(args)
        ```

    Args:
        training_component (bool, optional): Weather to include options relevant for training.
                                             Defaults to True.                                             
        data_component (bool, optional):     Weather to include options relevant for data streams.
                                             Defaults to True.

    Returns:
        argparse.ArgumentParser: _parser     A CLI parser with the common arguments.
    """
    parser = argparse.ArgumentParser(description = 
                                     "command line arguments for the nervox experiments",
                                     add_help=False)
    
    parser.add_argument('--logs_dir', required=False, type=str,
                        default=str(Path(Path.home(), 'tensorflow_logs')),
                        help='path to the top level tensorflow_log directory')
    parser.add_argument('--name', required=False, type=str, default=None,
                        help='name of the training job')
    parser.add_argument('--run_id', required=False, type=str, default=None,
                        help='run_id/version for the job')
    parser.add_argument('--batch_size', required=False, type=int, default=32,
                        help='The batch size for all dataloaders')

    if data_component:
        parser.add_argument('--datasets_dir', required=False, type=str,
                            default=str(Path(Path.home(), 'tensorflow_datasets')),
                            help='path to the directory that contains tensorflow datasets')
    
    # Add common arguments normally needed by the nervox trainer component. 
    if training_component:
        parser.add_argument('--max_epochs', required=False, type=int, default=100,
                            help='The maximum number of allowed epochs for training/fine-tuning jobs')
        parser.add_argument('--lr', required=False, type=float, default=1e-3,
                            help='The base learning rate for the training jobs.')
        parser.add_argument('--run_eagerly', action='store_true', default=False,
                            help='Weather to execute the training protocols eagerly or in graph mode')
        verbose_choices = [e.value for e in VerbosityLevel]
        parser.add_argument('--verbose', required=False, default='batch',
                type=VerbosityLevel, choices=verbose_choices,
                help='verbosity level of the nervox trainer')
        logging_choices = ['debug', 'info', 'warning', 'error', 'critical']                
        parser.add_argument('--logging_level', required=False, default='warning',
                        type=str.lower, choices=logging_choices,
                        help='log level of the nervox')
    return parser
# fmt:on


def get_channel_index(data_format: str) -> int:
    """
    Returns the channel index for a given data format.

    Args:
        data_format (str): The data format to get the channel index from. Valid data formats are
                           spatial (e.g.`NCHW`), sequential (e.g. `BTHWD`), `channels_first` and
                           `channels_last`.
    Returns:
        int:    The channel index as an int - either 1 or -1.

    Raises:
        ValueError: When the data format is unrecognized.
    """

    if data_format == "channels_first":
        return 1
    if data_format == "channels_last":
        return -1
    if _SPATIAL_CHANNELS_FIRST.match(data_format):
        return 1
    if _SPATIAL_CHANNELS_LAST.match(data_format):
        return -1
    if _SEQUENTIAL.match(data_format):
        return -1
    raise ValueError(
        "Unable to extract channel information from '{}'. Valid data formats are "
        "spatial (e.g.`NCHW`), sequential (e.g. `BTHWD`), `channels_first` and "
        "`channels_last`).".format(data_format)
    )


def to_tensor_shape(input_shape):
    """Converts a nested structure of tuples of int or None to TensorShapes
    Valid objects to be converted are:
    - tuples with elements of type int or None.
    - ints
    - None

    Args:
    input_shape: A nested structure of objects to be converted to TensorShapes.

    Returns:
    A nested structure of TensorShapes.

    Raises:
    ValueError: when the input tensor shape can't be converted.
    """
    to_tensor_shapes = partial(tf_utils.convert_shapes, to_tuples=False)
    return to_tensor_shapes(input_shape)


def compose(*functions: Callable) -> Callable:
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


def make_spec_concrete(
    spec: Union[Tuple[str, tf.TensorSpec], tf.TensorSpec], concrete_value: int = 1
) -> Union[Tuple[str, tf.TensorSpec], tf.TensorSpec]:
    """
    Remove None values from the tensor spec and replace them with a specific value
    Args:
        spec:               The tensor spec with possible None dimensions
        concrete_value:     The replacement value for None dimensions
    Returns:
        TensorSpec with concrete values
    """
    if isinstance(spec, tuple):
        key, _spec = spec
        shape = _spec.shape
        if None in shape:
            shape = list(shape)
            shape[shape.index(None)] = concrete_value
            shape = tf.TensorShape(shape)
        concrete_spec = (key, tf.TensorSpec(shape, _spec.dtype))

    else:
        if not isinstance(spec, tf.TensorSpec):
            raise TypeError(
                "The input type is not supported!\n"
                "expected type: tf.TensorSpec\n"
                f"received: {type(spec).__name}"
            )
        shape = spec.shape
        if None in shape:
            shape = list(shape)
            shape[shape.index(None)] = concrete_value
            shape = tf.TensorShape(shape)
        concrete_spec = tf.TensorSpec(shape, spec.dtype)

    return concrete_spec


def camel_to_snake(name: str):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def snake_to_camel(name: str, splitter="_"):
    components = name.split(splitter)
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + "".join(x.title() for x in components[1:])


def get_io_specs_tflite(export_dir, name="model.tflite"):
    interpreter = tf.lite.Interpreter(model_path=os.path.join(export_dir, name))
    interpreter.allocate_tensors()
    io_specs_tflite = {"input": interpreter.get_input_details()}
    io_specs_tflite.update({"output": interpreter.get_output_details()})
    return io_specs_tflite


def assign_lazy_configs(config, locals_dict=None):
    def evaluate(_value_literals):
        value = eval(_value_literals, {"__builtins__": None}, locals_dict)
        return value

    keys = [
        key
        for key, value in config.items()
        if isinstance(key, str) and isinstance(value, str)
    ]

    lazy_configs = {
        key: evaluate(value[1:])
        for key, value in config.items()
        if key in keys and value.startswith("@")
    }
    config.ClassificationMetricUpdater(lazy_configs)


def get_default_exp_dir(dataset_name=None, model_name=None):
    exp_dir = (
        os.path.join(dataset_name, model_name)
        if None not in [model_name, dataset_name]
        else "untitled"
    )
    return os.path.expanduser(os.path.join("~", "tensorflow_logs", exp_dir))


def get_urid():
    now = datetime.datetime.now()
    urid = "{0:04d}{1:02d}{2:02d}-{3:02d}{4:02d}".format(
        now.year, now.month, now.day, now.hour, now.minute
    )
    return urid


def plot_to_tf_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it. The
    supplied figure is closed and inaccessible after this call."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


@contextlib.contextmanager
def redirect_stdout(target: Union[os.PathLike, bytes, io.StringIO]):
    """Redirects the stdout to a file or a StringIO object
    Args:
        target (Union[os.PathLike, bytes, io.StringIO]): The target to redirect the stdout to,
                                                         this can be a file path, a buffer or
                                                         a StringIO object.
    Yields:
        target:     The target object is returned from the context manager, this can be a file
                    path, buffer or a StringIO object. The redirection happens when you enter
                    the context and is reverted back to the original stdout after the context
                    is exited.
    """
    sys.stdout.flush()
    std_out = sys.stdout
    log_file = None
    try:
        if isinstance(target, io.StringIO):
            sys.stdout = target
            yield target
        else:
            log_file = open(target, "a")
            sys.stdout = log_file
            yield log_file
    finally:
        sys.stdout.flush()
        sys.stdout = std_out
        log_file.close() if log_file else None
