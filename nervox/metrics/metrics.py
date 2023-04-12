import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from typing import Any
from typing import Tuple
from nervox.utils.types import TensorLike


class Metric(tf.Module, metaclass=ABCMeta):
    """
    This is the base class for all Metrics defined in the nervox framework.
    """

    def __init__(self, name: str, dtype: tf.DType = tf.float32):
        super().__init__(name)
        self.dtype = dtype

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

    @abstractmethod
    def result(self) -> Any:
        """
        Provides the accumulated/collected result for the metric.
        Usually, this is called at the end of each epoch.
        Returns:
            Any: the actual quantity of interest.
        """
        pass


class Mean(Metric):
    """
    Computes the mean of the passed values.
       Examples:
        >>> from nervox.metrics import Mean
        >>> mean = Mean()
        >>> mean.update(1.0)
        >>> mean.update(2.0)
        >>> mean.result()
        1.5
        >>> mean.reset()
        >>> mean.result()
        0.0
    """

    def __init__(
        self,
        *,
        axis=None,
        keep_dims: bool = False,
        name: str = "mean",
        dtype=tf.float32
    ):
        super().__init__(name, dtype=dtype)
        self.axis = axis
        self._value_sum = None
        self._count = None
        self.keep_dims = keep_dims

    @staticmethod
    def _create_variables(shape: Tuple):
        value_sum = tf.Variable(tf.zeros(shape), trainable=False, shape=shape)
        count = tf.Variable(0.0, trainable=False, shape=tuple())
        return value_sum, count

    def reset(self) -> None:
        self._value_sum.assign(
            tf.zeros(self._value_sum.shape, self.dtype)
        ) if self._value_sum is not None else None

        self._count.assign(
            tf.zeros(self._count.shape, self.dtype)
        ) if self._value_sum is not None else None

    def _variables_shape(self, input_shape):
        if self.keep_dims:
            shape = input_shape
            shape = tf.tensor_scatter_nd_update(shape, [[self.axis]], [1])
        elif self.axis is None:
            shape = tuple()
        else:
            axis = self.axis if self.axis >= 0 else (len(input_shape) + self.axis)
            axes = [_axis for _axis in range(len(input_shape)) if _axis != axis]
            shape = tf.gather(input_shape, axes)
        return shape

    def update(self, values: TensorLike) -> None:
        value_sum = tf.reduce_sum(values, axis=self.axis, keepdims=self.keep_dims)
        count = tf.cast(
            tf.size(values) if self.axis is None else tf.shape(values)[self.axis],
            dtype=self.dtype,
        )

        if self._value_sum is None or self._count is None:
            shape = self._variables_shape(tf.shape(values))
            self._value_sum, self._count = self._create_variables(shape)

        self._value_sum.assign_add(value_sum)
        self._count.assign_add(count)

    def result(self) -> TensorLike:
        mean = tf.constant(np.nan)
        mean = self._value_sum / self._count if self._count is not None else mean
        return mean
