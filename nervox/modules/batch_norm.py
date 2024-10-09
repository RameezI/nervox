# Copyright 2019 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Batch normalization module."""

import tensorflow as tf
from typing import Optional, Tuple, Union
from nervox.modules import Module
from nervox.metrics import Mean
from nervox.utils import initializers
from nervox.utils import get_channel_index


class BatchNorm(Module):
    r"""Batch normalization module.

    This implements normalization across the batch and spatial dimensions.
    It maintains moving averages of the mean and variance which can be
    used to normalize at test time. The constructor is generic and
    requires the user to pass in objects to compute these.

    At training time we use the batch statistics for that batch and these
    are then used to update the moving averages.

    At test time we can use the moving averages of the batch statistics
    It transforms the input ``x`` into:

    .. math::

        \d{outputs} = \d{scale} \dfrac{x - \mu}{\sigma + \epsilon} + \d{offset}

    Where :math:`\mu` and :math:`\sigma` are respectively the mean and standard
    deviation of ``x``.

    There are many different variations for how users want to manage scale and
    offset if they require them at all. These are:

      - No scale/offset in which case ``create_*`` should be set to ``False`` and
        ``scale``/``offset`` aren't passed when the module is called.
      - Trainable scale/offset in which case ``create_*`` should be set to
        ``True`` and again ``scale``/``offset`` aren't passed when the module is
        called. In this case this module creates and owns the ``scale``/``offset``
        variables.
      - Externally generated ``scale``/``offset``, such as for conditional
        normalization, in which case ``create_*`` should be set to ``False`` and
        then the values fed in at call time.

    Attributes:
      scale: If ``create_scale``, a trainable :tf:`Variable` holding the current
        scale after the module is connected for the first time.
      offset: If ``create_offset``, a trainable :tf:`Variable` holding the current
        offset after the module is connected for the first time.
    """

    def __init__(
        self,
        momentum: Union[float, None] = 0.999,
        eps: float = 1e-5,
        create_scale: bool = True,
        create_offset: bool = True,
        scale_init: Optional[initializers.Initializer] = None,
        offset_init: Optional[initializers.Initializer] = None,
        data_format: str = "channels_last",
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        """Constructs a ``BaseBatchNorm`` module.

        Args:
        create_scale:     Whether to create a trainable scale per channel applied
                          after the normalization.
        create_offset:    Whether to create a trainable offset per channel applied
                          after normalization and scaling.
        momentum:         Defines the decay rate of the exponential moving averages of
                          the mean and variance. A higher momentum value means a lower
                          decay rate, decay=(1-momentum); where; 0<=momentum<1. A lower
                          decay rate or momentum implies more dependence on the history
                          then on the immediate evidence.
        eps:              A small epsilon to avoid division by zero variance.
                          Defaults to ``1e-5``.
        scale_init:       Optional initializer for the scale variable. Can only be set
                          if ``create_scale=True``. By default scale is initialized to ``1``.
        offset_init:      Optional initializer for the offset variable. Can only be set
                          if ``create_offset=True``. By default offset is initialized to ``0``.
        data_format:      The data format of the input. Can be either
                          ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``.
                          Defaults to ``channels_last``.
        dtype:            The dtype for all the variables created by this module.
        name:             Name of the module.
        """
        super().__init__(name=name, dtype=dtype)

        self._eps = eps
        self._momentum = momentum
        self._moving_mean = None
        self._moving_variance = None

        self._data_format = data_format
        self._channel_index = get_channel_index(data_format)

        self._create_scale = create_scale
        self._create_offset = create_offset

        if not self._create_scale and scale_init is not None:
            raise ValueError("Cannot set `scale_init` if `create_scale=False`")
        self._scale_init = scale_init or initializers.Ones()

        if not self._create_offset and offset_init is not None:
            raise ValueError("Cannot set `offset_init` if `create_offset=False`")
        self._offset_init = offset_init or initializers.Zeros()

    def compute(
        self,
        inputs: tf.Tensor,
        training: bool,
        scale: Optional[tf.Tensor] = None,
        offset: Optional[tf.Tensor] = None,
    ):
        """Returns normalized inputs.

        Args:
        inputs:    An n-D tensor of the data_format specified above on which the
                    transformation is performed.
        training:   Whether the module should be connected in training mode,
                    meaning the moving averages are updated.
        scale:      A tensor up to n-D. The shape of this tensor must be broadcastable
                    to the shape of ``inputs``. This is the scale applied to the normalized
                    inputs. This cannot be passed in if the module was constructed with
                    ``create_scale=True``.
        offset:     A tensor up to n-D. The shape of this tensor must be broadcastable
                    to the shape of ``inputs``. This is the offset applied to the normalized
                    inputs. This cannot be passed in if the module was constructed with
                    ``create_offset=True``.

        Returns:
          An n-d tensor of the same shape as inputs that has been normalized.
        """
        use_batch_stats = training
        if self._create_scale:
            if scale is not None:
                raise ValueError(
                    "Cannot pass `scale` at call time if `create_scale=True`."
                )

        if self._create_offset:
            if offset is not None:
                raise ValueError(
                    "Cannot pass `offset` at call time if `create_offset=True`."
                )

        if scale is None:
            scale = self.scale
        if offset is None:
            offset = self.offset

        if training:
            mean, variance = self._moments(inputs, use_batch_stats)
            self._update_statistics(mean, variance)
        else:
            mean = self._moving_mean.result()
            variance = self._moving_variance.result()

        out = tf.nn.batch_normalization(
            inputs,
            mean=mean,
            variance=variance,
            scale=scale,
            offset=offset,
            variance_epsilon=self._eps,
        )
        return out

    def build(self, input_shape):
        rank = len(input_shape)
        keepdims = False if self._channel_index == -1 else True
        axes = tuple(set(range(rank)) - set([self._channel_index]))

        with tf.name_scope(self.name):
            self._moving_mean = Mean(
                axis=axes,
                keepdims=keepdims,
                momentum=self._momentum,
                name="moving_mean",
            )
            self._moving_variance = Mean(
                axis=axes,
                keepdims=keepdims,
                momentum=self._momentum,
                name="moving_variance",
            )

            # lets build it already since we know the shape
            self._moving_mean.build(input_shape=input_shape)
            self._moving_variance.build(input_shape=input_shape)

            # Creates scale and offset parameters
            if self._channel_index == -1:
                params_shape = [input_shape[-1]]
            else:  # self._channel_index == 1
                params_shape = [input_shape[1]] + [1] * (rank - 2)

            if self._create_scale:
                self.scale = tf.Variable(
                    self._scale_init(params_shape, self.dtype),
                    name="scale",
                    trainable=True,
                )

            if self._create_offset:
                self.scale = tf.Variable(
                    self._offset_init(params_shape, self.dtype),
                    name="offset",
                    trainable=True,
                )

    def _moments(
        self, inputs: tf.Tensor, use_batch_stats: bool
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if use_batch_stats:
            mean, variance = tf.nn.moments(inputs, self._axis, keepdims=True)
        else:  # use moving stats
            mean = self._moving_mean.value
            variance = self._moving_variance.value
        return mean, variance

    def _update_statistics(self, mean, variance):
        self._moving_mean.update(mean)
        self._moving_variance.update(variance)
