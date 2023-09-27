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

import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from typing import Any
from typing import Tuple, Optional, Union, Collection
from nervox.utils.types import TensorLike


class Metric(tf.Module, metaclass=ABCMeta):
    """
    This is the base class for all Metrics defined in the nervox framework.
    """

    def __init__(self, name: str, dtype: tf.DType = tf.float32):
        super().__init__(name)
        self.dtype = dtype
        self._built = False

    @property
    def built(self):
        return self._built

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the metric to it's initial state.
        Usually, this is called at the start of each epoch.
        """
        pass

    def update_state(self, *args, **kwargs) -> None:
        """
        Updates the metrics state using the passed batch output.
         This wrapper for the `update` method ensures compatibility with keras Metrics.
        """
        self.update(*args, **kwargs)

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Updates the metrics state using the passed batch output.
        Usually, this is called once for each batch.
        """
        pass

    @property
    def value(self):
        return self.result()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.update(*args, **kwds)
        return self.value

    @abstractmethod
    def result(self) -> Any:
        """
        Provides the accumulated/collected result for the metric.
        Usually, this is called at the end of each epoch.
        Returns:
            Any: the actual quantity of interest.
        """
        pass


# class Mean(Metric):
#     """
#     Maintains a cumulative running  mean for a value.

#     This module computes the mean of a batch of elements along specified axes, by
#     maintaining two internal variables, 1) sum: the sum of all the values along the
#     specified axes 2) count: the total count of the elements summed.The mean at any
#     instant is simply the ratio of the values of the two internal variables.
#         TODO: Add maths here.

#        Examples:
#         >>> from nervox.metrics import Mean
#         >>> mean = Mean()
#         >>> mean.update(1.0)
#         >>> mean.update(2.0)
#         >>> mean.result()
#         1.5
#         >>> mean.reset()
#         >>> mean.result()
#         0.0
#     """

#     def __init__(
#         self,
#         *,
#         axis=None,
#         keep_dims: bool = False,
#         name: Optional[str] = "mean",
#         dtype=tf.float32
#     ):
#         super().__init__(name, dtype=dtype)
#         self.axis = axis
#         self._value_sum = None
#         self._count = None
#         self.keep_dims = keep_dims

#     @staticmethod
#     def _create_variables(shape: Tuple, dtype:tf.DType):
#         value_sum = tf.Variable(tf.zeros(shape), trainable=False, shape=shape, dtype=dtype)
#         count = tf.Variable(0.0, trainable=False, shape=tuple(), dtype=dtype)
#         return value_sum, count

#     def reset(self) -> None:
#         self._value_sum.assign(
#             tf.zeros(self._value_sum.shape, self.dtype)
#         ) if self._value_sum is not None else None

#         self._count.assign(
#             tf.zeros(self._count.shape, self.dtype)
#         ) if self._value_sum is not None else None

#     def _variables_shape(self, input_shape):
#         if self.keep_dims:
#             shape = input_shape
#             shape = tf.tensor_scatter_nd_update(shape, [[self.axis]], [1])
#         elif self.axis is None:
#             shape = tuple()
#         else:
#             axis = self.axis if self.axis >= 0 else (len(input_shape) + self.axis)
#             axes = [_axis for _axis in range(len(input_shape)) if _axis != axis]
#             shape = tf.gather(input_shape, axes)
#         return shape

#     def update(self, values: TensorLike) -> None:
#         value_sum = tf.reduce_sum(values, axis=self.axis, keepdims=self.keep_dims)
#         count = tf.cast(
#             tf.size(values) if self.axis is None else tf.shape(values)[self.axis],
#             dtype=self.dtype,
#         )

#         if self._value_sum is None or self._count is None:
#             shape = self._variables_shape(tf.shape(values))
#             self._value_sum, self._count = self._create_variables(shape, self.dtype)

#         self._value_sum.assign_add(value_sum)
#         self._count.assign_add(count)

#     def result(self) -> TensorLike:
#         mean = tf.constant(np.nan)
#         mean = self._value_sum / self._count if self._count is not None else mean
#         return mean


# Alias
ShapeLike = Union[Tuple, tf.TensorShape, TensorLike]


class Mean(Metric):
    """Maintains a moving average for values along specified dimensions.
    Two modes are supported, CumulativeRunningAverage (default) that computes
    a running arithmetic mean and ExponentialRunningAverage, which calculates
    the running mean that is biased either towards the history or towards the
    more immediate evidence.

    CumulativeRunningAverage:
    When  momentum is set to `None` this mode is used to compute the running mean.
    In this mode each sample is weighted equally towards the average resulting in
    arithmetic mean of the values over the specified axes.

    Algorithm:
        Initially:
            state_0 =0

        Then iteratively:
            count +=1
            average_i += (value - state{i-1}*(1/count)

    ExponentialRunningAverage
    When momentum is set, an exponential running average is computed.
    The new samples are weighted with exponentially decaying weights, the decay factor,
    which is calculated as (1-momentum). A high momentum value implies that once we
    have gained  enough evidence (observations), it will be harder for new samples to
    influence the average, offering increased resistance to outliers.
    Reference (https://arxiv.org/pdf/1412.6980.pdf).

    Algorithm:
        Initially:
            state_0 = 0
            count = 0

        Then iteratively:
            count += 1
            state_i += (value - state{i-1}) * (1 - momentum)
            average_i = hidden_i / (1 - decay^count)


    Attributes:
      average: Variable holding average. Note that this is None until the first
        value is passed.
    """

    def __init__(
        self,
        axis: Optional[Union[int, Collection[int]]] = None,
        keepdims: bool = False,
        momentum: Optional[float] = None,
        name: Optional[str] = None,
        dtype=tf.float32,
    ):
        """Creates a de-biased moving average module.

        Args:

          axis [optional]:      Axis or axes along which to average. The default, axis=None,
                                will average over all of the elements of the input array.
                                If axis is negative it counts from the last to the first axis.

          keepdims: bool        Weather to keep dimensions or to decimate the dimension being
                                reduced by the averaging. When `True` the decimated dimensions
                                are kept and is represented in the shape by a value of `1`.

          momentum [optional]:  Defines the decay rate of an exponentially  moving average.
                                A higher momentum means a lower decay rate, decay=(1-momentum);
                                where; 0<=momentum<1. Higher momentum implies more dependence on the
                                history then on the immediate evidence.

                                Note values close to 1 result in a slow decay whereas values close to
                                0 result in faster decay, thus relying more on immediate values in
                                contrast to values too-far in future.

                                When `None`, the default, cumulative running average is computed
                                instead of a exponential decaying average, whereby each sample
                                has equal contribution.
          name:                 Name of the module.
        """
        super().__init__(name=name, dtype=dtype)
        self.keepdims = keepdims
        self.axis = tuple([axis]) if isinstance(axis, int) else axis
        self._momentum = momentum
        self._latent = None
        self._count = None

    def build(self, input_shape: Union[tf.TensorShape, TensorLike]):
        # expand axes
        axes = None
        rank = tf.size(input_shape)
        if self.axis is not None:
            axes = [ax if ax >= 0 else (rank - ax) for ax in self.axis]

        mean = tf.reduce_mean(
            tf.zeros(input_shape, dtype=self.dtype),
            axis=axes,
            keepdims=self.keepdims,
        )
        # create variables
        with tf.name_scope(self.name):
            self._latent = tf.Variable(
                mean,
                trainable=False,
                dtype=self.dtype,
                name="latent",
            )
            self._count = tf.Variable(
                0,
                trainable=False,
                dtype=tf.int64,
                name="count",
            )
        self._built = True

    def update(self, values: tf.Tensor):
        """Applies EMA to the value given."""
        if self._latent is None or self._count is None:
            self.build(tf.shape(values))

        mean = tf.reduce_mean(values, axis=self.axis, keepdims=self.keepdims)
        self._count.assign_add(1)
        delta = mean - self._latent

        if self._momentum is not None:
            # exponentially decaying average
            self._latent.assign_add(delta * (1.0 - self._momentum))
        else:
            # cumulative arithmetic mean
            count = tf.cast(self._count, self.dtype)
            self._latent.assign_add(delta * (1.0 / count))

    def result(self) -> TensorLike:
        if not self.built:
            raise AssertionError(
                "Results requested on an uninitiated object, the metric needs to be built first!\n"\
                " The metric is automatically built on first invocation, i.e. when passed a\n"\
                " tensor-like object to compute a running mean. When no such tensor is yet," \
                " provided, the metic is uninitiated and the internal state is undefined.\n\n"
            )
        if self._momentum:
            count = tf.cast(self._count, self.dtype)
            mean = self._latent / (1 - tf.pow(self._momentum, count))
        else:
            mean = self._latent
        return mean

    def reset(self):
        """Resets the EMA."""
        self._count.assign(tf.zeros_like(self._count))
        self._latent.assign(tf.zeros_like(self._latent))
