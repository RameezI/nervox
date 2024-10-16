# Copyright © 2023 Rameez Ismail - All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
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
#
# This code is adapted from Sonnet by DeepMind:
# https://github.com/deepmind/sonnet
# The project is licensed under the Apache-2.0.
# You may obtain a copy of the license at:
# http://www.apache.org/licenses/LICENSE-2.0

"""
Initializers for neural network layers.
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import abc
import collections
import numpy as np
import tensorflow as tf
from nervox.utils import types
from nervox.utils import capture_params
from typing import Iterable, Mapping, Optional, Union


class Initializer:
    def __init_subclass__(cls, *args, **kwargs):
        """The __init_subclass__ method is called when a subclass of Initializer is defined.
        This method wraps the user `__init__` method with the capture_params, which enables
        automatic capturing of the objects parameterization. The parameterization is stored
        in the `params` attribute of the object.
        """
        super().__init_subclass__(*args, **kwargs)
        __user_init = getattr(cls, "__init__")

        # if the user has not already decorated the __init__ method, decorate it.
        if not hasattr(__user_init, "_wrapper_capture_params_"):
            __user_init = capture_params(__user_init, **kwargs)

        def __wrapped_init(self, *args, **kwargs):
            # call the user's __init__ method
            # we call the __init__ method of the base class already,
            # so if the user forgets to call super().__init__ , a
            # default super().__init__ is in place.
            if cls.__base__ is Initializer:
                super(cls, self).__init__()
            __user_init(self, *args, **kwargs)

        cls.__init__ = __wrapped_init

    def __init__(self) -> None: ...

    @abc.abstractmethod
    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor: ...


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0."""

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        dtype = _as_numerical_dtype(dtype)
        return tf.zeros(shape, dtype)


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1."""
    
    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        dtype = _as_numerical_dtype(dtype)
        return tf.ones(shape, dtype)


class Constant(Initializer):
    """Initializer that generates tensors initialized to the given value."""

    def __init__(self, value: Union[float, int]):
        if not np.isscalar(value):
            raise TypeError(
                "Invalid type for value: {} (expected scalar).".format(type(value))
            )
        self.value = value

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        dtype = _as_numerical_dtype(dtype)
        value = tf.convert_to_tensor(self.value, dtype)
        return tf.fill(value=value, dims=shape)


class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.

    The generated values follow a uniform distribution in the range
    ``[minval, maxval)``.
    """

    def __init__(
        self,
        minval: float = 0.0,
        maxval: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Constructs a random uniform initializer.

        Args:
          minval: A python scalar or a scalar tensor. Lower bound of the range of
            random values to generate. Defaults to ``0``.
          maxval: A python scalar or a scalar tensor. Upper bound of the range of
            random values to generate. Defaults to ``1``.
          seed: The seed used in the generation of random numbers.
        """
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType):
        dtype = _as_numerical_dtype(dtype)
        return tf.random.uniform(
            shape=shape,
            minval=self.minval,
            maxval=self.maxval,
            dtype=dtype,
            seed=self.seed,
        )


class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution."""

    def __init__(
        self,
        mean: float = 0.0,
        stddev: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Constructs a random normal initializer.

        Args:
          mean: A python scalar or a scalar tensor. Mean of the random values to
            generate.
          stddev: A python scalar or a scalar tensor. Standard deviation of the
            random values to generate.
          seed: The seed used in the generation of random numbers.
        """
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        dtype = _as_floating_dtype(dtype)
        return tf.random.normal(
            shape=shape, mean=self.mean, stddev=self.stddev, dtype=dtype, seed=self.seed
        )


class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.

    These values follow a normal distribution except that values more than two
    standard deviations from the mean are discarded and re-drawn. This is the
    recommended initializer for neural network weights and filters.
    """

    def __init__(
        self,
        mean: float = 0.0,
        stddev: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Constructs a truncated normal initializer.

        Args:
          mean(float):              A python scalar or a scalar tensor. Mean of the random values
                                    to generate (default: 0.0).
          stddev(Union[float]):     A python scalar or a scalar tensor. Standard deviation of the
                                    random values to generate. Can also be one of the following
                                    (default: 1.0).
          seed(float, optional):    The seed used in the generation of random numbers.
        """
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType):
        dtype = _as_floating_dtype(dtype)
        return tf.random.truncated_normal(
            shape=shape, mean=self.mean, stddev=self.stddev, dtype=dtype, seed=self.seed
        )


class Identity(Initializer):
    """Initializer that generates the identity matrix.

    Constructs a 2D identity matrix or batches of these.
    """

    def __init__(self, gain: float = 1.0):
        """Constructs an identity initializer.

        Args:
          gain: Multiplicative factor to apply to the identity matrix.
        """
        self.gain = gain

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        dtype = _as_numerical_dtype(dtype)
        rank = shape.shape[0] if isinstance(shape, tf.Tensor) else len(shape)
        if rank < 2:
            raise ValueError(
                "The tensor to initialize must be " "at least two-dimensional"
            )
        elif rank == 2:
            initializer = tf.eye(num_rows=shape[0], num_columns=shape[1], dtype=dtype)
        else:  # rank > 2
            initializer = tf.eye(
                num_rows=shape[-2],
                num_columns=shape[-1],
                batch_shape=shape[:-2],
                dtype=dtype,
            )
        return self.gain * initializer


class Orthogonal(Initializer):
    """Initializer that generates an orthogonal matrix.

    NOTE: Does not support 1D tensors.

    The implementation is based on :cite:`saxe2013exact`.

    If the shape of the tensor to initialize is two-dimensional, it is initialized
    with an orthogonal matrix obtained from the QR decomposition of a matrix of
    random numbers drawn from a normal distribution.
    If the matrix has fewer rows than columns then the output will have orthogonal
    rows. Otherwise, the output will have orthogonal columns.

    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape ``(shape[0] * ... * shape[n - 2], shape[n - 1])``
    is initialized, where ``n`` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.
    """

    def __init__(self, gain: float = 1.0, seed: Optional[int] = None):
        """Constructs an orthogonal initializer.

        Args:
          gain: Multiplicative factor to apply to the orthogonal matrix
          seed: ``int``, the seed used in the generation of random numbers.
        """
        self.gain = gain
        self.seed = seed

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        dtype = _as_floating_dtype(dtype)
        if len(shape) < 2:
            raise ValueError(
                "The tensor to initialize must be " "at least two-dimensional"
            )
        # Flatten the input shape with the last dimension remaining
        # its original shape so it works for conv2d
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = [tf.maximum(num_cols, num_rows), tf.minimum(num_cols, num_rows)]

        # Generate a random matrix
        a = tf.random.normal(flat_shape, dtype=dtype, seed=self.seed)
        # Compute the qr factorization
        q, r = tf.linalg.qr(a, full_matrices=False)
        # Make Q uniform
        d = tf.linalg.tensor_diag_part(r)
        q *= tf.sign(d)
        if num_rows < num_cols:
            q = tf.linalg.matrix_transpose(q)
        return self.gain * tf.reshape(q, shape)


# -----------------------------------------------------------------------------
# Generalized Xavier/Glorot initializer
class VarianceScaling(Initializer):
    """Initializer capable of adapting its scale to the shape of weights tensors.

    With ``distribution="truncated_normal" or "normal"``,
    samples are drawn from a distribution with a mean of zero and a standard
    deviation (after truncation, if used) ``stddev = sqrt(scale / n)``
    where ``n`` is:

      - Number of input units in the weight tensor, if ``mode = fan_in``.
      - Number of output units, if ``mode = fan_out``.
      - Average of the numbers of input and output units, if ``mode = fan_avg``.

    Note that for transposed convolution the mode selected should be reversed. For
    number of input units use ``fan_out`` and for number of output units
    ``fan_in``.

    With ``distribution=uniform``, samples are drawn from a uniform distribution
    within ``[-limit, limit]``, with ``limit = sqrt(3 * scale / n)``.

    The variance scaling initializer can be configured to generate other standard
    initializers using the scale, mode and distribution arguments. Here are some
    example configurations:

    ==============  ==============================================================
    Name            Parameters
    ==============  ==============================================================
    glorot_uniform  scale=1.0, mode=``fan_avg``, distribution=``uniform``
    glorot_normal   scale=1.0, mode=``fan_avg``, distribution=``truncated_normal``
    lecun_uniform   scale=1.0, mode=``fan_in``,  distribution=``uniform``
    lecun_normal    scale=1.0, mode=``fan_in``,  distribution=``truncated_normal``
    he_uniform      scale=2.0, mode=``fan_in``,  distribution=``uniform``
    he_normal       scale=2.0, mode=``fan_in``,  distribution=``truncated_normal``
    ==============  ==============================================================
    """

    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_in",
        distribution: str = "truncated_normal",
        seed: Optional[int] = None,
    ):
        """Constructs a variance scaling initializer.

        Args:
          scale: Scaling factor (positive ``float``).
          mode: One of ``fan_in``, ``fan_out``, ``fan_avg``.
          distribution: Random distribution to use. One of ``truncated_normal``,
            ``untruncated_normal`` and  ``uniform``.
          seed: ``int``, the seed used in the generation of random numbers.

        Raises:
          ValueError: In case of an invalid value for the ``scale``, ``mode`` or
            ``distribution`` arguments.
        """
        if scale <= 0.0:
            raise ValueError("`scale` must be positive float.")
        if mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise ValueError("Invalid `mode` argument:", mode)
        distribution = distribution.lower()
        if distribution not in {"uniform", "truncated_normal", "normal"}:
            raise ValueError("Invalid `distribution` argument:", distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        dtype = _as_floating_dtype(dtype)
        scale = self.scale

        # get shape slices for fan_in and fan_out
        fan_in, fan_out = _compute_fans(shape)
        fan_in = tf.cast(fan_in, dtype)
        fan_out = tf.cast(fan_out, dtype)

        # normalize the scale as per the mode
        if self.mode == "fan_in":
            scale /= fan_in
        elif self.mode == "fan_out":
            scale /= fan_out
        else:
            scale /= tf.maximum(1.0, (fan_in + fan_out) / 2.0)

        if self.distribution == "truncated_normal":
            # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            distribution_stddev = 0.87962566103423978
            stddev = tf.sqrt(scale) / distribution_stddev
            return tf.random.truncated_normal(
                shape=shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed
            )
        elif self.distribution == "normal":
            stddev = tf.sqrt(scale)
            return tf.random.normal(
                shape=shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed
            )
        else:  # self.distribution == "uniform"
            limit = tf.sqrt(3.0 * scale)
            return tf.random.uniform(
                shape=shape, minval=-limit, maxval=limit, dtype=dtype, seed=self.seed
            )


# -----------------------------------------------------------------------------
# specializations of VarianceScaling
class GlorotUniform(VarianceScaling):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(
            scale=1.0,
            mode="fan_avg",
            distribution="uniform",
            seed=seed,
        )

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        return super().__call__(shape, dtype)


class GlorotNormal(VarianceScaling):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(
            scale=1.0,
            mode="fan_avg",
            distribution="truncated_normal",
            seed=seed,
        )

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        return super().__call__(shape, dtype)


class LecunUniform(VarianceScaling):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(
            scale=1.0,
            mode="fan_in",
            distribution="uniform",
            seed=seed,
        )

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        return super().__call__(shape, dtype)


class LecunNormal(VarianceScaling):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(
            scale=1.0,
            mode="fan_in",
            distribution="truncated_normal",
            seed=seed,
        )

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        return super().__call__(shape, dtype)


class HeUniform(VarianceScaling):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(
            scale=2.0,
            mode="fan_in",
            distribution="uniform",
            seed=seed,
        )

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        return super().__call__(shape, dtype)


class HeNormal(VarianceScaling):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(
            scale=2.0,
            mode="fan_in",
            distribution="truncated_normal",
            seed=seed,
        )

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        return super().__call__(shape, dtype)


class KaimingUniform(VarianceScaling):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(
            scale=2.0,
            mode="fan_in",
            distribution="uniform",
            seed=seed,
        )

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        return super().__call__(shape, dtype)


class KaimingNormal(VarianceScaling):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(
            scale=2.0,
            mode="fan_in",
            distribution="truncated_normal",
            seed=seed,
        )

    def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
        return super().__call__(shape, dtype)


# -----------------------------------------------------------------------------


def check_initializers(
    initializers: Mapping[str, Initializer], expected_keys: Iterable[str]
):
    """Checks a dictionary of initializers only contains the given keys."""
    if initializers is None:
        return {}

    if not isinstance(initializers, collections.abc.Mapping):
        raise TypeError("Initializers must be a dict-like object.")

    extra_keys = frozenset(initializers) - frozenset(expected_keys)
    if extra_keys:
        raise KeyError(
            "Invalid initializer keys {}, initializers can only "
            "be provided for {}".format(
                ", ".join(map(repr, extra_keys)), ", ".join(map(repr, expected_keys))
            )
        )
    return initializers


def _compute_fans(shape: types.ShapeLike):
    """Computes the number of input and output units for a weight shape.

    Args:
      shape: Integer shape tuple or `tf.TensorShape`. This is the shape of the
            weight tensor whose input and output units are to be computed

    Returns:
      A tuple of scalars `(fan_in, fan_out)`.
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        fan_in_shape = shape[:-1]
        fan_in = np.prod(fan_in_shape)
        fan_out = shape[-1]
    return fan_in, fan_out


def _as_floating_dtype(dtype: tf.DType) -> tf.DType:
    dtype = tf.as_dtype(dtype)
    if dtype.is_floating:
        return dtype
    raise ValueError("Expected floating point type, got {}".format(dtype))


def _as_numerical_dtype(dtype: tf.DType) -> tf.DType:
    dtype = tf.as_dtype(dtype)
    if dtype.is_floating or dtype.is_integer:
        return dtype
    raise ValueError("Expected integer or floating point type, got {}".format(dtype))
