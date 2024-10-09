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
This snippet demonstrates how to `capture_params` decorator captures 
parametrizes an object instantiation and store them as a jsonable
object, accessible through `params` attribute of the object once
it is initialized.
"""
from functools import wraps
import inspect


def capture_params(*args_outer):
    def _capture_params(_callable):
        """
        Decorator that captures parameters of a method.
        """

        @wraps(_callable)
        def wrapper(*args, **kwargs):
            obj = args[0]
            _args = args[1:]

            if obj is not None:
                setattr(obj, "_wrapped_with_capture_params_", True)

            parameters = [
                param
                for param in inspect.signature(_callable).parameters.values()
                if param.kind not in [inspect.Parameter.VAR_KEYWORD]
            ][1:]

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

            # collect parameters from the function definition
            params = {
                param.name: param.default
                for param in parameters
                if param.kind
                in [
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ]
            }

            # Collect args and kwargs passed to the function
            # -----------------------------------------------
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

            # (d) update the params dictionary with the positional
            #     and keyword arguments passed to the function
            params.update({"args_positional": args_positional_only})
            params.update(args_mapped_positional)
            params.update({key: value for key, value in kwargs.items()})

            # update the params attribute of the object
            if not hasattr(obj, "params"):
                obj.params = params
            else:
                obj.params.update(params)

            return _callable(*args, **kwargs)

        wrapper._wrapper_capture_params_ = True
        return wrapper

    if args_outer and callable(args_outer[0]):
        return _capture_params(args_outer[0])
    else:
        return _capture_params


class Transform:
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        __user_init__ = getattr(cls, "__init__")

        if not hasattr(__user_init__, "wrapped_with_capture_params"):
            __user_init__ = capture_params(cls.__init__, **kwargs)

        def _wrapped_init(self, *args, **kwargs):
            __user_init__(self, *args, **kwargs)

        cls.__init__ = _wrapped_init


class MyTransform(Transform):
    def __init__(
        self,
        pos_arg_1,
        pos_arg2,
        /,
        param1,
        param2,
        *args,
        kwarg1=10,
        kwarg2=None,
        **kwargs,
    ):
        self.pos_arg_1 = pos_arg_1
        self.pos_arg2 = pos_arg2
        self.param1 = param1
        self.param2 = param2
        self.kwarg1 = kwarg1
        self.kwarg2 = kwarg2
        self.args = args
        self.kwargs = kwargs


if __name__ == "__main__":
    mt = MyTransform(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, param_a="a", param_b="b")
    print(mt.params)
    mt = MyTransform(1, 2, param1=3, param2=4, param_a="a", param_b="b")
    print(mt.params)
