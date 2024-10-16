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
    This module contains the various serializers for the Nervox library.
    The serializers are used to serialize the objects to JSON format.   
"""
import inspect
import tensorflow as tf


def serializer_dtype(obj: tf.DType):
    if not isinstance(obj, tf.DType):
        raise TypeError(
            f"Invalid type spotted:"
            "\nExpected: tf.DType"
            "\nReceived: {type(obj).__name__}"
        )
    return {
        "__class__": "DType",
        "__module__": "tensorflow",
        "__init__": {"_args": [obj.as_datatype_enum]},
        "__repr__": repr(obj),
    }


def serializer_slice(obj: slice):
    if not isinstance(obj, slice):
        raise TypeError(
            f"Invalid type spotted:"
            "\nExpected: slice"
            "\nReceived: {type(obj).__name__}"
        )
    return {
        "__class__": "slice",
        "__module__": "builtins",
        "__init__": {"_args": (obj.start, obj.stop, obj.step)},
        "__repr__": repr(obj),
    }


class SerializerRegistry:
    """A registry for serializers of types.

    This class is used to register serializers for a user-defined type or any
    type that does not have a default serialization mechanism in place.
    The registered serializers are invoked by the framework to serialize the
    objects to JSON format, therefore make sure that the serialization routine
    provided by your custom serializer is jsonable.

    Users must also ensure that the serialization is able to reconstruct the
    object correctly later through either a default deserialization mechanism
    or by providing a custom deserialization routine. A custom deserializer is
    handled by the `DeserializerRegistry`.

    Example:
    ```python

    from nervox.utils import SerializerRegistry

    MyCustomClass:
      def __init__(x, y)
        self.x = x
        self.y = y

    def my_custom_serializer(obj:MyCustomClass):
        # SerializerRegistry expects a function with a single
        # argument and an annotation for the obj_type.
        return {'__init__': {'x': obj.x, 'y':obj.y),
                '__class__': type(obj).__name__,',
                '__module__': obj.__module__}
                }

    SerializerRegistry.register(my_custom_serializer)\n"
    """

    _serializers = {tf.DType: serializer_dtype, slice: serializer_slice}

    @classmethod
    def register(cls, serializer: callable):
        """Register a serializer for a given type.
        Args:
            serializer: The serializer function to register.
        """
        if not inspect.isfunction(serializer):
            raise ValueError("The serializer must be a function")

        if len(inspect.signature(serializer).parameters) != 1:
            raise ValueError(
                "The serializer function must accept a single argument"
                ", i.e. the object to be serialized"
            )

        obj_type = serializer.__annotations__["obj_type"]

        if obj_type is None:
            raise ValueError(
                "The serializer function must have an"
                " annotation for the obj_type argument"
            )
        cls._serializers[serializer.__annotations__["obj"]] = serializer

    @classmethod
    def get(cls, obj_type, default=None):
        """Retrieves a serializer for a given object type.
        Args:
            obj_type: The object type to retrieve the serializer for.
        Returns:
            The serializer function for the given object type, or None if not found.
        """
        return cls._serializers.get(obj_type, default)

    @classmethod
    def list(cls):
        """List all the registered serializers.
        Returns:
            A list of all the registered serializers.
        """
        return list(cls._serializers.keys())
