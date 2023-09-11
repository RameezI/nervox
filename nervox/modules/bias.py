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
A bias module that supports broadcasting.
"""

from typing import Optional, Sequence
import tensorflow as tf
from nervox.modules import Module
from nervox.utils import initializers
from nervox.utils import types


class Bias(Module):
    """Bias module.
    Example Usage:

        >>> N, H, W, C = 1, 2, 3, 4
        >>> x = tf.random.normal([N, H, W, C])
        >>> from nervox.modules import Bias

        >>> scalar_bias = Bias(dims=[])
        >>> scalar_bias_output = scalar_bias(x)
        >>> assert scalar_bias.b.shape == []

    create a bias over all non-minibatch dimensions:

        >>> all_bias = Bias()
        >>> all_bias_output = all_bias(x)
        >>> assert all_bias.b.shape == [H, W, C]

    create a bias over the last non-minibatch dimension:

        >>> last_bias = Bias(dims=[-1])
        >>> last_bias_output = last_bias(x)
        >>> assert last_bias.b.shape == [C]

    create a bias over the first non-minibatch dimension:

        >>> first_bias = Bias(dims=[1])
        >>> first_bias_output = first_bias(x)
        >>> assert first_bias.b.shape == [H, 1, 1]

    subtract and later add the same learned bias:

        >>> bias = Bias()
        >>> h1 = bias(x, multiplier=-1)
        >>> h2 = bias(x)
        >>> h3 = bias(x, multiplier=-1)
        >>> reconstructed_x = bias(h3)
        >>> assert tf.reduce_all(tf.equal(x, reconstructed_x))
    """

    def __init__(
        self,
        dims: Optional[Sequence[int]] = None,
        initializer: Optional[initializers.Initializer] = None,
        dtype: Optional[tf.DType] = tf.float32,
        name: Optional[str] = None,
    ):
        """Constructs a `Bias` module that supports broadcasting. This modules creates & applies
        a bias variable to the input, the bias is applied only to the specified dims.

        Args:
          dims:         Sequence of which dimensions in the input the bias is applied to.
                        The leading dimensions are emitted, while intermediate and trailing
                        dimensions  are broadcasted over (given a dimension of 1).
          initializer:  Optional initializer for the bias. Default to zeros.
          name:         Name of the module.
        """

        # fmt: off
        super().__init__(name=name, dtype=dtype)
        self.dims = dims
        self.initializer = initializers.Zeros() \
          if initializer is None else initializer
        
        # variables to be created
        self._bias = None
        # fmt: on

    def build(self, input_shape):
        with tf.name_scope(self.name):
            bias_shape = calculate_bias_shape(input_shape, self.dims)
            input_size = input_shape[1:]
            self.input_size = input_size
            self._bias = tf.Variable(self.initializer(bias_shape, self.dtype), name="bias")

    def compute(self, inputs: tf.Tensor, multiplier: Optional[float] = None):
        """Adds bias to `inputs` and optionally multiplies by `multiplier`.

        Args:
          inputs: A Tensor of size `[batch_size, input_size1, ...]`.
          multiplier: A scalar or Tensor which the bias term is multiplied by before
          adding it to `inputs`. Anything which works in the expression `bias *
          multiplier` is acceptable here. This may be useful if you want to add a
          bias in one place and subtract the same bias in another place via
          `multiplier=-1`.

        Returns:
          A Tensor of size `[batch_size, input_size1, ...]`.
        """
        if multiplier is not None:
            return inputs + (self._bias * multiplier)
        else:
            return inputs + self._bias


def calculate_bias_shape(input_shape: types.ShapeLike, bias_dims: Sequence[int]):
    """Calculate `bias_shape` based on the `input_shape` and `bias_dims`.

    Args:
      input_shape: Shape of the input being passed into the module. The leading
        dimension is the mini-batch size.
      bias_dims: The dimensions that bias should be applied over. The remaining
        dimensions will be broadcast over.

    Returns:
      bias_shape: Tuple corresponding to the shape of bias Variable to create.

    Raises:
      ValueError: If the user attempts to add bias over the mini-batch dimension,
          e.g. `bias_dims=[0]`.
    """
    input_rank = len(input_shape)
    if bias_dims is None:
        # If None, default is to use all dimensions.
        return input_shape[1:]

    elif not bias_dims:
        # If empty list, use a scalar bias.
        return ()

    else:
        # Otherwise, calculate bias_shape from bias_dims.
        bias_shape = [1] * input_rank
        # Populate bias dimensions.
        for dim in bias_dims:
            if dim < 0:
                dim %= input_rank

            if dim == 0:
                raise ValueError("Cannot apply bias across the minibatch dimension.")
            elif dim >= input_rank:
                raise ValueError(
                    "Dimension %d (bias_dims=%r) out of range for input of rank %r."
                    % (dim, tuple(bias_dims), input_rank)
                )

            bias_shape[dim] = input_shape[dim]
        # Strip leading unit dimensions.
        start = input_rank
        for dim in range(1, input_rank):
            if bias_shape[dim] != 1:
                start = dim
                break
        return tuple(bias_shape[start:])  # Do not apply across minibatch dimension.


if __name__ == "__main__":
    N, H, W, C = 1, 2, 3, 4
    x = tf.random.normal([N, H, W, C], dtype=tf.float32)
    bias = Bias(dims=[])
    bias_output = bias(x)
    print(bias_output)
    print(bias.params)