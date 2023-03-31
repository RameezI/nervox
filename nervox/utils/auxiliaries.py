"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import sys
import os
import re
import json
import datetime
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
import matplotlib.pyplot as plt
from typing import Union, Tuple, Callable
from enum import Enum


class VerbosityLevel(Enum):
    KEEP_SILENT = 'silent'
    UPDATE_AT_EPOCH = 'epoch',
    UPDATE_AT_BATCH = 'batch'


import tensorflow as tf
from dataclasses import dataclass

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
        return f"{'Total FLOPs':<20}: {self.flops:>20n}\n" \
               f"{'Total Parameters':<20}: {self.parameters:>20n}\n" \
               f"{'  Trainable':<20}: {self.trainable_parameters:>20n}\n" \
               f"{'  Non-Trainable':<20}: {self.non_trainable_parameters:>20n}\n"

#fmt:off
def base_parser(training_component=True, data_component=True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="command line arguments for the nervox experiments",
                                     add_help=False)
    
    # General cmd-line args useful when working with experiments produced by nervox
    # Using these standard args make it easier to write consistent launch files.
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
#fmt:on

def compose(*functions: Callable) -> Callable:
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


def make_spec_concrete(spec: Union[Tuple[str, tf.TensorSpec], tf.TensorSpec],
                       concrete_value: int = 1) \
        -> Union[Tuple[str, tf.TensorSpec], tf.TensorSpec]:
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
            raise TypeError('The input type is not supported!\n'
                            'expected type: tf.TensorSpec\n'
                            f'received: {type(spec).__name}')
        shape = spec.shape
        if None in shape:
            shape = list(shape)
            shape[shape.index(None)] = concrete_value
            shape = tf.TensorShape(shape)
        concrete_spec = tf.TensorSpec(shape, spec.dtype)
    
    return concrete_spec


def camel_to_snake(name: str):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def snake_to_camel(name: str, splitter='_'):
    components = name.split(splitter)
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + ''.join(x.title() for x in components[1:])


def get_io_specs_tflite(export_dir, name='model.tflite'):
    interpreter = tf.lite.Interpreter(model_path=os.path.join(export_dir, name))
    interpreter.allocate_tensors()
    io_specs_tflite = {'input': interpreter.get_input_details()}
    io_specs_tflite.update({'output': interpreter.get_output_details()})
    return io_specs_tflite


# def get_shared_directory_path():
#     if os.name == 'nt':
#         shared_drv = "Z:\\"
#         if not os.path.isdir(shared_drv):
#             raise "shared-drive is not mapped, I am  looking it under @ '{}' ".format(shared_drv)
#     else:
#         shared_drv = "/home/2017-0324_ai4ph"
#     return shared_drv


# def get_test_dataset(run_dir, batch_size=1, transform=True):
#     params = read_params_from_file(os.path.join(run_dir, 'params.json'))
#     datasets_pkg = params['trainer']['datasets_pkg']
#     dataset_name = params['trainer'].get('dataset_name', None)
#     dataset_conf = params['dataset']
#
#     # if not os.path.isdir(dataset_conf['datasets_dir']):
#     #     # TODO: ALso validate that existing path has valid files
#     #     print('The requested datasets directory is not found. '
#     #           'Attempting to load from the local datasets location, instead....')
#     #     dataset_conf.update({'datasets_dir': None})
#     #
#     dataset = __import__('{}.{}'.format(datasets_pkg, dataset_name), fromlist=[dataset_name]).Dataset(**dataset_conf)
#     _, ds_test = dataset(batch_size=batch_size, transform=transform)
#     return ds_test

def serialize_to_json(self):
    def default_serializer(obj):
        _config = getattr(obj, 'params', None)
        
        if _config is None:
            raise ValueError('Serialization to JSON failed!\n'
                             ' The object does not have `params` attribute set, this is required attribute '
                             'which must provide the minimum config to reconstruct the object. If you are\n'
                             ' serializing a custom class object, do not forgot to decorate your class '
                             '`__init__` method with `capture_params` decorator, which can create `params`\n'
                             ' attribute for your object automatically. You can also supply the config'
                             ' manually by setting the `params` attribute of your object.\n'
                             f' The concerned class is : `{type(obj).__name__}`')
        
        _config.ClassificationMetricUpdater({'class_name': type(obj).__name__})
        return _config
    
    json_serialization = ''
    assert isinstance(self, object), \
        'The serialization candidate must be an instance of a class'
    try:
        json_serialization = json.dumps(self, default=default_serializer,
                                        sort_keys=True, indent=4)
    except ValueError as e:
        raise ValueError('\n\nSerialization Failure!:\n'
                         f'Failed while attempting to serialize an object of class `{type(self).__name__}`\n'
                         f' {str(e)}')
    
    return json_serialization


def capture_params(*args_outer, **kwargs_outer):
    ignore_list = kwargs_outer.get('ignore', [])
    apply_local_updates = kwargs_outer.get('apply_local_updates', False)
    
    # ignore_list.extend(['self'])
    
    def _capture_params(func):
        @wraps(_capture_params)
        def _wrapper_capture_params_(obj, *args, **kwargs):
            assert isinstance(obj, object), 'The capture params must be used on non-static methods of a class'
            _profile = sys.getprofile()
            
            parameters = [param for param in inspect.signature(func).parameters.values()
                          if param.kind not in [inspect.Parameter.VAR_KEYWORD]][1:]
            
            assert all([param.kind not in [inspect.Parameter.POSITIONAL_ONLY,
                                           inspect.Parameter.VAR_POSITIONAL]
                        for param in parameters]), "The use of 'POSITIONAL_ONLY' and 'VAR_POSITIONAL' arguments" \
                                                   " is currently NOT supported by the capture function." \
                                                   " This feature is likely to be not supported in future as well." \
                                                   " Please use keywords only or positional arguments" \
                                                   "that support keywords instead."
            
            obj.params = {param.name: param.default for param in parameters if param.name not in ignore_list}
            obj.params.update({key: value for key, value in kwargs.items() if key not in ignore_list})
            positional_params = list(inspect.signature(func).parameters.values())[1:len(args) + 1]
            obj.params.update({param.name: value for param, value in zip(positional_params, args)})
            
            def profiler(frame, event, arg):
                if event == 'return' and frame.f_back.f_code.co_name in ['_wrapper_capture_params_']:
                    frame_locals = frame.f_locals
                    updates = {key: value for key, value in frame_locals.items() if key in obj.params}
                    obj.params.update(updates) if apply_local_updates else None
            
            try:
                sys.setprofile(profiler)
                func(obj, *args, **kwargs)
            finally:
                sys.setprofile(_profile)
        
        return _wrapper_capture_params_
    
    if args_outer and callable(args_outer[0]):
        return _capture_params(args_outer[0])
    else:
        return _capture_params


def assign_lazy_configs(config, locals_dict=None):
    def evaluate(_value_literals):
        value = eval(_value_literals, {'__builtins__': None}, locals_dict)
        return value
    
    keys = [key for key, value in config.items() if isinstance(key, str) and isinstance(value, str)]
    
    lazy_configs = {key: evaluate(value[1:]) for key, value in config.items()
                    if key in keys and value.startswith('@')}
    config.ClassificationMetricUpdater(lazy_configs)


def get_default_exp_dir(dataset_name=None, model_name=None):
    exp_dir = os.path.join(dataset_name, model_name) if None not in [model_name, dataset_name] else 'untitled'
    return os.path.expanduser(os.path.join("~", "tensorflow_logs", exp_dir))


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def get_urid():
    now = datetime.datetime.now()
    urid = "{0:04d}{1:02d}{2:02d}-{3:02d}{4:02d}".format(now.year, now.month, now.day, now.hour, now.minute)
    return urid


def plot_to_tf_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


@contextlib.contextmanager
def redirect_stdout(file: Union[os.PathLike, bytes, str]):
    log_file = open(file, 'a')
    std_out = sys.stdout
    try:
        sys.stdout.flush()
        sys.stdout = log_file
        yield log_file
    finally:
        sys.stdout.flush()
        sys.stdout = std_out
        log_file.close()


if __name__ == '__main__':
    logs_dir = os.path.expanduser(os.path.join("~", "tensorflow_logs"))
    experiment_dir = os.path.join(logs_dir, 'cifar10', 'convnet_small', '20200413-1145')
    io_specs = get_io_specs_tflite(os.path.join(experiment_dir, 'export'))
    print(io_specs['input'], '\n', io_specs['output'])
