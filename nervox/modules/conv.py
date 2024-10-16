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

"""Convolutional modules."""

import numpy as np
import tensorflow as tf
from nervox.modules import Module
from nervox.utils import pad
from nervox.utils import initializers
from nervox.utils import get_channel_index
from typing import Optional, Sequence, Union
import logging

# define logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConvND(Module):
    """A general N-dimensional convolutional module."""

    def __init__(
        self,
        num_spatial_dims: int,
        output_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: pad.Paddings = "SAME",
        with_bias: bool = True,
        kernel_init: Optional[initializers.Initializer] = initializers.HeNormal(),
        bias_init: Optional[initializers.Initializer] = initializers.Zeros(),
        data_format: Optional[str] = "NDHWC",
        name: Optional[str] = None,
        dtype: Optional[tf.DType] = tf.float32,
    ):
        """
        Constructs a `ConvND` module with the specified parameters. This is a generalization of the various
        convolutional modules found in most neural networks. This module creates a `N` dimensional convolutional
        kernel that is convolved (actually cross-correlated) with the input tensor to produce a output tensor. If
        `with_bias` is True a bias vector is created and added to the output as well.

        Args:

          num_spatial_dims (int):                       The number of spatial dimensions along which convolution
                                                        occurs. Must be one of {1, 2, 3}. This is `N ` in the
                                                        description above. This also represents the rank of the
                                                        kernel.

          output_channels (int):                        The number of output channels.

          kernel_size (Union[int, Sequence[int]]):      Sequence of kernel sizes (of length num_spatial_dims),
                                                        or an integer. `kernel_shape` will be expanded to define
                                                        a kernel size in all dimensions.

          stride (Union[int, Sequence[int]], optional): Sequence of strides (of length num_spatial_dims), or an
                                                        integer.`stride` will be expanded to define stride in all
                                                        dimensions. Defaults to 1.

          rate (Union[int, Sequence[int]], optional):   Sequence of dilation rates (of length num_spatial_dims),
                                                        or integer that is used to define dilation rate in all
                                                        dimensions. A value of 1 means standard ND convolution,
                                                        whereas `rate > 1` corresponds to dilated convolution.
                                                        Defaults to 1.

          padding (Union[str, Paddings], optional):     Padding to apply to the input. This can either "SAME",
                                                        "VALID" or a callable or sequence of callables up to N.
                                                        Any callables must take a single integer argument equal
                                                        to the effective kernel size and return a list of two
                                                        integers representing the padding before and after.
                                                        See nervox.utils.pad.* for more details and example
                                                        functions. Defaults to "SAME".

          with_bias (bool, optional):                   Whether to include bias parameters. Defaults to True.

          kernel_init (Initializer, optional):          Optional initializer for the convolutional kernel. By
                                                        default the kernel is initialized truncated random normal
                                                        values with a standard deviation of:
                                                        `1/sqrt(input_feature_size)`,
                                                        which is commonly used when the inputs are zero centered
                                                        (see https://arxiv.org/abs/1502.03167v3).


          bias_init (Initializer, optional):           Optional initializer for the bias. By default the bias is
                                                        initialized to zero.

          data_format (str, optional):                  The data format of the input. Defaults to
          name (str, optional):                         Name of the module. Defaults to None.
          dtype (tf.DType, optional):                   The dtype of the input. Defaults to tf.float32.

        Raises:
            ValueError: When `num_spatial_dims` is not one of [1, 2, 3].
            ValueError: When the `kernel_size` is not a single integer or
                        a sequence of integers of length `num_spatial_dims`.
            ValueError: When `padding` is not a string or callable.
        """
        super().__init__(name=name, dtype=dtype)

        if not 1 <= num_spatial_dims <= 3:
            raise ValueError(
                f"supported convolution operations for num_spatial_dims=1, 2 or "
                "3, received num_spatial_dims={num_spatial_dims}."
            )

        self.output_channels = output_channels

        try:
            kernel_size = tuple(np.broadcast_to(kernel_size, (num_spatial_dims,)))
        except ValueError as e:
            logger.error(str(e))
            raise ValueError(
                "kernel_size must be a single integer or a sequence of integers"
                f"of length {num_spatial_dims}.\n"
            )

        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate

        if isinstance(padding, str):
            self.conv_padding = padding.upper()
            self.padding_func = None
        else:
            self.conv_padding = "VALID"
            self.padding_func = padding

        self.data_format = data_format
        self._channel_index = get_channel_index(data_format)
        self._num_spatial_dims = num_spatial_dims
        self.with_bias = with_bias

        self.kernel_init = kernel_init
        self.bias_init = bias_init

        # Generate the kernel and bias variables using the build function.
        # The kernel and bias variables are initialized  through the
        # `self.kernel_init` and `self.bias_init` respectively.
        self.kernel = None
        self.bias = None

        # Whether bias will be built.
        self.with_bias = with_bias

    def build(self, input_shape: tf.TensorShape):
        """Creates the variables for the convolutional kernel and bias (if applicable).
        The input shape is used to create a variable for the kernel. The kernel is
        initialized using the `kernel_init` passed to the constructor. If `with_bias`
        is True, a bias variable is created as well. The bias is initialized using
        the `bias_init` passed to the constructor.

        Args:
            input_shape (tf.TensorShape):  The input shape to the module `call` function.
                                           This is used to create the kernel and bias variables,
                                           which will be updated during the training process.
                                           The first dimension is the batch dimension.
        Raises:
            ValueError:   When the input shape has an invalid rank. The expected rank
                          is `num_spatial_dims + 2`. The additional two dimensions are
                          the batch dimension and the channel dimension.
            ValueError:   When the number of input channels in `input_shape` is not known.
        """

        if not (input_shape.rank == self._num_spatial_dims + 2):
            raise ValueError(
                f"Input shape must have rank {self._num_spatial_dims + 2}.\n"
                f"Received a shape: {input_shape} with rank {input_shape.rank}."
            )

        self.input_channels = input_shape[self._channel_index]
        if self.input_channels is None:
            raise ValueError("The number of input channels must be known.")

        # kernel shape is [*kernel_size, input_channels, output_channels]
        kernel_shape = self.kernel_size + (self.input_channels, self.output_channels)
        self.kernel = tf.Variable(
            self.kernel_init(kernel_shape, self._dtype),
            name="kernel",
        )

        # bias shape is [output_channels]
        if self.with_bias:
            self.bias = tf.Variable(
                self.bias_init((self.output_channels,), self._dtype),
                name="bias",
            )

        if self.padding_func:
            self._padding = pad.create(
                padding=self.padding_func,
                kernel=self.kernel_size,
                rate=self.rate,
                n=self._num_spatial_dims,
                channel_index=self._channel_index,
            )

    def compute(self, inputs: tf.Tensor) -> tf.Tensor:
        """Applies the defined convolution to the inputs.
        Args:
            inputs (tf.Tensor): An ``N + 2`` rank :tf:`Tensor` of dtype :tf:`float16`, `tf.bfloat16`
                                or `tf.float32` to which the convolution is applied.
        Returns:
            tf.Tensor: An ``N + 2`` dimensional :tf:`Tensor` of shape:
              ``[batch_size, output_dim_1, output_dim_2, ..., output_channels]``.

        """
        if self.padding_func:
            inputs = tf.pad(inputs, self._padding)

        outputs = tf.nn.convolution(
            inputs,
            self.kernel,
            strides=self.stride,
            padding=self.conv_padding,
            dilations=self.rate,
            data_format=self.data_format,
        )
        if self.with_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=self.data_format)

        return outputs


# ---------------------------------------------------------------------------------------------------------------------
# specializations of ConvND


class Conv3D(ConvND):
    def __init__(
        self,
        output_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        rate: int | Sequence[int] = 1,
        padding: pad.Paddings = "SAME",
        with_bias: bool = True,
        kernel_init: Optional[initializers.Initializer] = initializers.HeNormal(),
        bias_init: Optional[initializers.Initializer] = initializers.Zeros(),
        data_format: Optional[str] = "NDHWC",
        name: Optional[str] = None,
        dtype: Optional[tf.DType] = tf.float32,
    ):
        """
        Constructs a `Conv2D` module. This is a specialization of `ConvND` for `num_spatial_dims=3`. See `ConvND`
        for details.  This module creates a  three dimensional convolutional kernel that is convolved (actually
        cross-correlated) with the input tensor to produce a output tensor. If `with_bias` is True a bias vector
        is created and added to the output as well.

          Args:

            output_channels (int):                        The number of output channels.

            kernel_size (Union[int, Sequence[int]]):      Sequence of kernel sizes (of length num_spatial_dims),
                                                          or an integer. `kernel_shape` will be expanded to define
                                                          a kernel size in all dimensions.

            stride (Union[int, Sequence[int]], optional): Sequence of strides (of length num_spatial_dims), or an
                                                          integer.`stride` will be expanded to define stride in all
                                                          dimensions. Defaults to 1.

            rate (Union[int, Sequence[int]], optional):   Sequence of dilation rates (of length num_spatial_dims),
                                                          or integer that is used to define dilation rate in all
                                                          dimensions. A value of 1 means standard ND convolution,
                                                          whereas `rate > 1` corresponds to dilated convolution.
                                                          Defaults to 1.

            padding (Union[str, Paddings], optional):     Padding to apply to the input. This can either "SAME",
                                                          "VALID" or a callable or sequence of callables up to N.
                                                          Any callables must take a single integer argument equal
                                                          to the effective kernel size and return a list of two
                                                          integers representing the padding before and after.
                                                          See nervox.utils.pad.* for more details and example
                                                          functions. Defaults to "SAME".

            with_bias (bool, optional):                   Whether to include bias parameters. Defaults to True.

            kernel_init (Initializer, optional):          Optional initializer for the convolutional kernel. By
                                                          default the kernel is initialized truncated random normal
                                                          values with a standard deviation of:
                                                          `1/sqrt(input_feature_size)`,
                                                          which is commonly used when the inputs are zero centered
                                                          (see https://arxiv.org/abs/1502.03167v3).


            bias_init (Initializer, optional):           Optional initializer for the bias. By default the bias is
                                                          initialized to zero.

            data_format (str, optional):                  The data format of the input. Defaults to
            name (str, optional):                         Name of the module. Defaults to None.
            dtype (tf.DType, optional):                   The dtype of the input. Defaults to tf.float32.

          Raises:
              ValueError:   When the `kernel_size` is not a single integer or
                            a sequence of integers of length `num_spatial_dims`.
              ValueError:   When `padding` is not a string or callable.
        """
        super().__init__(
            3,
            output_channels,
            kernel_size,
            stride,
            rate,
            padding,
            with_bias,
            kernel_init,
            bias_init,
            data_format,
            name,
            dtype,
        )


class Conv2D(ConvND):
    def __init__(
        self,
        output_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        rate: int | Sequence[int] = 1,
        padding: pad.Paddings = "SAME",
        with_bias: bool = True,
        kernel_init: Optional[initializers.Initializer] = initializers.HeNormal(),
        bias_init: Optional[initializers.Initializer] = initializers.Zeros(),
        data_format: Optional[str] = "NHWC",
        name: Optional[str] = None,
        dtype: Optional[tf.DType] = tf.float32,
    ):
        """
        Constructs a `Conv2D` module. This is a specialization of `ConvND` for `num_spatial_dims=2`. See `ConvND`
        for details.  This module creates a two dimensional convolutional kernel that is convolved (actually
        cross-correlated) with the input tensor to produce a output tensor. If `with_bias` is True a bias vector
        is created and added to the output as well.

          Args:

            output_channels (int):                        The number of output channels.

            kernel_size (Union[int, Sequence[int]]):      Sequence of kernel sizes (of length num_spatial_dims),
                                                          or an integer. `kernel_shape` will be expanded to define
                                                          a kernel size in all dimensions.

            stride (Union[int, Sequence[int]], optional): Sequence of strides (of length num_spatial_dims), or an
                                                          integer.`stride` will be expanded to define stride in all
                                                          dimensions. Defaults to 1.

            rate (Union[int, Sequence[int]], optional):   Sequence of dilation rates (of length num_spatial_dims),
                                                          or integer that is used to define dilation rate in all
                                                          dimensions. A value of 1 means standard ND convolution,
                                                          whereas `rate > 1` corresponds to dilated convolution.
                                                          Defaults to 1.

            padding (Union[str, Paddings], optional):     Padding to apply to the input. This can either "SAME",
                                                          "VALID" or a callable or sequence of callables up to N.
                                                          Any callables must take a single integer argument equal
                                                          to the effective kernel size and return a list of two
                                                          integers representing the padding before and after.
                                                          See nervox.utils.pad.* for more details and example
                                                          functions. Defaults to "SAME".

            with_bias (bool, optional):                   Whether to include bias parameters. Defaults to True.

            kernel_init (Initializer, optional):          Optional initializer for the convolutional kernel. By
                                                          default the kernel is initialized truncated random normal
                                                          values with a standard deviation of:
                                                          `1/sqrt(input_feature_size)`,
                                                          which is commonly used when the inputs are zero centered
                                                          (see https://arxiv.org/abs/1502.03167v3).


            bias_init (Initializer, optional):           Optional initializer for the bias. By default the bias is
                                                          initialized to zero.

            data_format (str, optional):                  The data format of the input. Defaults to
            name (str, optional):                         Name of the module. Defaults to None.
            dtype (tf.DType, optional):                   The dtype of the input. Defaults to tf.float32.

          Raises:
              ValueError:   When the `kernel_size` is not a single integer or
                            a sequence of integers of length `num_spatial_dims`.
              ValueError:   When `padding` is not a string or callable.
        """
        super().__init__(
            2,
            output_channels,
            kernel_size,
            stride,
            rate,
            padding,
            with_bias,
            kernel_init,
            bias_init,
            data_format,
            name,
            dtype,
        )


class Conv1D(ConvND):
    def __init__(
        self,
        output_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        rate: int | Sequence[int] = 1,
        padding: pad.Paddings = "SAME",
        with_bias: bool = True,
        kernel_init: Optional[initializers.Initializer] = initializers.HeNormal(),
        bias_init: Optional[initializers.Initializer] = initializers.Zeros(),
        data_format: Optional[str] = "NWC",
        name: Optional[str] = None,
        dtype: Optional[tf.DType] = tf.float32,
    ):
        """
        Constructs a `Conv1D` module. This is a specialization of `ConvND` for `num_spatial_dims=1`. See `ConvND`
        for details.  This module creates a one dimensional convolutional kernel that is convolved (actually
        cross-correlated) with the input tensor to produce a output tensor. If `with_bias` is True a bias vector
        is created and added to the output as well.

          Args:

            output_channels (int):                        The number of output channels.

            kernel_size (Union[int, Sequence[int]]):      Sequence of kernel sizes (of length num_spatial_dims),
                                                          or an integer. `kernel_shape` will be expanded to define
                                                          a kernel size in all dimensions.

            stride (Union[int, Sequence[int]], optional): Sequence of strides (of length num_spatial_dims), or an
                                                          integer.`stride` will be expanded to define stride in all
                                                          dimensions. Defaults to 1.

            rate (Union[int, Sequence[int]], optional):   Sequence of dilation rates (of length num_spatial_dims),
                                                          or integer that is used to define dilation rate in all
                                                          dimensions. A value of 1 means standard ND convolution,
                                                          whereas `rate > 1` corresponds to dilated convolution.
                                                          Defaults to 1.

            padding (Union[str, Paddings], optional):     Padding to apply to the input. This can either "SAME",
                                                          "VALID" or a callable or sequence of callables up to N.
                                                          Any callables must take a single integer argument equal
                                                          to the effective kernel size and return a list of two
                                                          integers representing the padding before and after.
                                                          See nervox.utils.pad.* for more details and example
                                                          functions. Defaults to "SAME".

            with_bias (bool, optional):                   Whether to include bias parameters. Defaults to True.

            kernel_init (Initializer, optional):          Optional initializer for the convolutional kernel. By
                                                          default the kernel is initialized truncated random normal
                                                          values with a standard deviation of:
                                                          `1/sqrt(input_feature_size)`,
                                                          which is commonly used when the inputs are zero centered
                                                          (see https://arxiv.org/abs/1502.03167v3).


            bias_init (Initializer, optional):           Optional initializer for the bias. By default the bias is
                                                          initialized to zero.

            data_format (str, optional):                  The data format of the input. Defaults to
            name (str, optional):                         Name of the module. Defaults to None.
            dtype (tf.DType, optional):                   The dtype of the input. Defaults to tf.float32.

          Raises:
              ValueError:   When the `kernel_size` is not a single integer or
                            a sequence of integers of length `num_spatial_dims`.
              ValueError:   When `padding` is not a string or callable.
        """
        super().__init__(
            1,
            output_channels,
            kernel_size,
            stride,
            rate,
            padding,
            with_bias,
            kernel_init,
            bias_init,
            data_format,
            name,
            dtype,
        )


if __name__ == "__main__":
    N, D, H, W, C = 1, 10, 16, 16, 3
    x = tf.random.normal([N, D, H, W, C], dtype=tf.float32)
    conv3d = Conv3D(output_channels=16, kernel_size=3)
    conv3d_output = conv3d(x)
    print(conv3d_output)
    print(conv3d.params)
