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
import json
import logging
import inspect
import tensorflow as tf
from functools import wraps
from typing import Dict, Any
from enum import Enum

from nervox.utils.types import *
from .serializers import SerializerRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _validate_parametrization(params):
    """This function validates the automatic parameterization of a class.
    The `params` captured must be a dictionary of type `Dict[str, JSONSerializable]`
    and must contain `__class__` and `__module__` keys.

    Args:
        params (Dict[str, JSONSerializable]): The captured parameterization of the
                                               object to be validated.

    Raises:
        TypeError:  Raised when the `params` attribute is not of type
                    `Dict[str, JSONSerializable]`/
        ValueError: Raised when the `params` attribute does not contain
                    `__class__` and `__module__` keys.
        TypeError:  Raised when the `params` attribute contains keys
                    that are not of type `str`
        TypeError:  Raised when the `params` attribute contains values
                     that are not of type `JSONSerializable.
    """
    if not isinstance(params, dict):
        raise TypeError(
            "The `params` attribute must be a dictionary of type `Dict[str, JSONSerializable]`"
        )

    elif ["__class__", "__module__"] not in list(params.keys()):
        raise ValueError(
            "The `params` attribute must contain `__class__` and `__module__` keys."
        )

    else:
        for key, value in params.items():
            if not isinstance(key, str):
                raise TypeError(
                    "The `params` attribute may only contain keys of type `str`\n"
                    f"received: {type(key).__name__} as key"
                )
            if not is_jsonable(value):
                raise TypeError(
                    "The `params` values may only be of type `JSONSerializable`\n"
                    f"received: {type(value).__name__}"
                )


def serialize_to_json(obj):
    def default_serializer(_obj):
        if isinstance(_obj, tuple(SerializerRegistry.list())):
            # If the object is an instance of a class whose serialization
            # method is known in the SerializerRegistry, we can use the
            # registered serializer to serialize.
            serializer = SerializerRegistry.get(type(_obj))
            _config = serializer(_obj)

        else:
            # if no custom serializer is available, we check if the object
            #  has an attribute `params` if so, we use it to serialize.
            _config = getattr(_obj, "params", None)

        if _config is None:
            raise TypeError(
                "Serialization to JSON failed!\n"
                "The serialization of a custom object can be enabled in two ways:\n\n"
                "1) The object provides `params` attribute, a dictionary with, at minimum,\n"
                "the keys: `__init__`, `__class__` and `__module__` to enable reproduction.\n"
                "If you  implemented a custom class, you may want to decorate your class\n"
                "`__init__` method with `@capture_params to automatically achieve this.\n"
                "2) You can register a custom serializer for the object type by using the\n"
                "`SerializerRegistry` mechanism, see the docs for details.\n"
                f" The concerned class is : `{type(_obj).__name__}`"
            )
        return _config

    try:
        json_serialization = json.dumps(
            obj, default=default_serializer, sort_keys=True, indent=4
        )
    except TypeError as e:
        logger.error(str(e))
        raise TypeError(
            "\n\nSerialization Failure:\n"
            f"Failed while attempting to serialize an object of class `{type(obj).__name__}`\n"
            f" {str(e)}"
        )
    return json_serialization


def is_jsonable(x):
    try:
        serialize_to_json(x)
        return True
    except (TypeError, OverflowError):
        return False


def _expand_params(config: Dict[str, Any]) -> Dict[str, Any]:
    def _get_parameterization(obj: Any) -> Dict[str, Any]:
        parameterization = None

        if isinstance(obj, (tuple, list, set)):
            parameterization = []
            for item in obj:
                parameterization.append(_get_parameterization(item))
            parameterization = tuple(parameterization)

        elif isinstance(obj, dict):
            parameterization = {}
            for key, item in obj.items():
                parameterization[key] = _get_parameterization(item)
            parameterization = dict(parameterization)

        else:
            parameterization = serialize_to_json(obj)

        return parameterization

    expanded_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            expanded_config[key] = _expand_params(value)
        elif not is_jsonable(value):
            parameterization = _get_parameterization(value)
            if parameterization is None:
                raise ValueError(
                    f"{key}: Object `{type(value)}` is not JSON serializable"
                    " and does not provide parameterization."
                )
        else:
            expanded_config[key] = value

    return expanded_config


def capture_params(*args_outer, **kwargs_outer):
    ignore_list = kwargs_outer.pop("ignore", [])
    apply_local_updates = kwargs_outer.pop("apply_local_updates", False)

    def _set_attribute_value(obj, attribute_name, attribute_value):
        if isinstance(obj, tf.Module):
            restore_tracking = obj._setattr_tracking
            obj._setattr_tracking = False
            setattr(obj, attribute_name, attribute_value)
            obj._setattr_tracking = restore_tracking
        else:
            setattr(obj, attribute_name, attribute_value)

    def _capture_params(_callable):
        if not callable(_callable):
            raise TypeError(
                "The `capture_params` decorator must be used on a callable type,\n"
                "for example, a function, a class method or a functor etc.\n"
                f"expected: `Callable`\n"
                f"received: `{type(_callable).__name__}` -> `{_callable.__name__}`"
            )

        def is_bounded():
            # check if the callable is a class method
            # or is bounded to an object.
            return True

        @wraps(_callable)
        def _wrapped_callable_(*args, **kwargs):
            """This wraps the callable to capture the parameters of the function/method.
            The captured parameters are stored in the `params` attribute of the object.
            When the captured method is not a class method, the `params` attribute is
            added to the function object itself. In case of a class method, the `params`
            attribute is added to the class itself, while for built-in methods, the
            `params` attribute is added to a

            Raises:
                AttributeError: When an illegal use of the `params` attribute is detected.
            """
            _profile = sys.getprofile()
            parameters = [
                param
                for param in inspect.signature(_callable).parameters.values()
                if param.kind not in [inspect.Parameter.VAR_KEYWORD]
            ]

            obj = None
            if is_bounded():
                # ------------------------------------------------------------------------------
                # ignore the first argument when callable is a instance or class method
                #  as the first argument is always `self` or `cls` respectively.
                obj = args[0]
                _args = args[1:]
                parameters = parameters[1:]

            setattr(_wrapped_callable_, "_wrapped_with_capture_params_", True)

            # ------------------------------------------------------------------------------
            # extract the position-only and variable positional arguments
            position_only_args = [
                param.name
                for param in parameters
                if param.kind == inspect.Parameter.POSITIONAL_ONLY
            ]
            position_keyword_args = [
                param.name
                for param in parameters
                if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            ]

            # ------------------------------------------------------------------------------
            # check if the object already has a `params` attribute and verify its legal use.
            if hasattr(obj, "params"):
                try:
                    _validate_parametrization(obj.params)
                except (ValueError, TypeError) as e:
                    logger.error(str(e))
                    _typename = type(obj).__name__
                    raise AttributeError(
                        f"Automatic parameterization is setup for object {obj} of type:"
                        f" {_typename}\n, this means that `params` attribute is handled"
                        " automatically\n. However, an illegal use of the attribute is"
                        " detected\n. Please, make sure you are not using the `params`"
                        " attribute for other purposes.\n",
                    )

            # ------------------------------------------------------------------------------
            # collect parameters from the function definition
            params = {
                param.name: param.default
                for param in parameters
                if param.kind
                in [
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ]
                and param.name not in ignore_list
            }

            # Collect args and kwargs passed to the function
            # ------------------------------------------------------------------------------
            # (a) extract the position-only arguments
            args_positional_only = list(_args[: len(position_only_args)])
            args_mapped_positional = dict()

            # (b) extract the variable positional arguments
            passed_as_keyword = sum(
                1 for param in position_keyword_args if param in kwargs
            )
            varargs_start = len(position_only_args)
            varargs_start += len(position_keyword_args) - passed_as_keyword
            var_positional_params = list(_args[varargs_start:])

            if var_positional_params:
                # (c1) extend _args to include the variable positional arguments
                # as well as the mappable positional arguments. This implies all
                # arguments passed to the function are recorded `args_positional`
                # attribute of the object.
                args_positional_only = _args
                params = {
                    key: value
                    for key, value in params.items()
                    if key not in position_keyword_args
                }
            else:
                # (c2) when variable positional arguments are not spotted we can map
                # the positional arguments and update the `params` dictionary.
                # Otherwise, mappable positional arguments are passed as a list
                # to the function.
                _zipped_args = zip(
                    position_keyword_args, _args[len(position_only_args) :]
                )
                args_mapped_positional = dict(_zipped_args)
                args_mapped_positional = {
                    key: value
                    for key, value in args_mapped_positional.items()
                    if key not in ignore_list
                }

            # fmt:off
            # (d) update the params dictionary with the positional
            #     and keyword arguments passed to the function 
            if args_positional_only:
                params.update({"args_positional": args_positional_only})
            params.update(args_mapped_positional)
            params.update({key: value for key, value in kwargs.items()
                            if key not in ignore_list})
            # fmt:on

            # ------------------------------------------------------------------------------
            # collect parameters from local context
            def profiler(frame, event, _):
                if event == "return" and frame.f_back.f_code.co_name in [
                    "_wrapped_callable_"
                ]:
                    frame_locals = frame.f_locals
                    updates = {
                        key: value
                        for key, value in frame_locals.items()
                        if key in params
                    }
                    params.update(updates) if apply_local_updates else None

            # ------------------------------------------------------------------------------
            try:
                sys.setprofile(profiler)
                _callable(*args, **kwargs)

                # fill in the params attribute
                params = _expand_params(params)

                # make `params` attribute for the callable object
                #  if it does exist.
                if obj is not None:
                    if not hasattr(obj, "params"):
                        _set_attribute_value(obj, "params", {})
                    obj.params[_callable.__name__] = params
                    obj.params["__module__"] = type(obj).__module__
                    obj.params["__class__"] = type(obj).__name__

                # record the params attribute as an attribute of
                #  the callable of the closure for the callable.
                else:
                    # set the params attribute for the closure
                    _wrapped_callable_.params[_callable.__name__] = params
                    _wrapped_callable_.params["__module__"] = type(_callable).__module__
                    _wrapped_callable_.params["__class__"] = type(_callable).__name__

            finally:
                sys.setprofile(_profile)

        return _wrapped_callable_

    if args_outer and callable(args_outer[0]):
        return _capture_params(args_outer[0])
    else:
        return _capture_params
