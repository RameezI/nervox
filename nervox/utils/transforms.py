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

"""A module that provides pure functional tensor transforms."""

import tensorflow as tf
from nervox.utils.types import TensorLike


def onehot_transform(x: TensorLike, axis=-1) -> tf.Tensor:
    """
    This function transforms a tensor into a onehot encoding.
    Args:
        x (TensorLike): A tensor with arbitrary shape (..., num_classes, ...)
                        to be transformed into a onehot encoding at a specified
                        axis. For the given axis the class with the highest score
                        is set to `1` and the rest are set to `0`.

        axis (int):     The axis that contains the class prediction scores.
                        Defaults to -1.

    Returns:
        tf.Tensor:      The onehot encoding of the input tensor. The class with the
                        highest score is set to `1` and the rest are set to `0`.
    """
    indices = tf.argmax(x, axis=axis)
    onehot_predictions = tf.one_hot(indices, depth=x.shape[axis])
    return onehot_predictions


def multihot_transform(x: TensorLike, threshold=0.5) -> tf.Tensor:
    """
    This function transforms a tensor into a multi-hot encoding.

    Args:
        x (TensorLike):     A tensor with arbitrary shape (..., num_classes, ...)
                            to be transformed into a multi-hot encoding. All values
                            above a threshold is set to 1 and the rest are set to 0.

        threshold (float):  A threshold to decide if the one-hot label is 0 or 1.
                            Defaults to 0.5.

    Returns:
        tf.Tensor:          The multi-hot encoding of the input tensor. The class with the
                            score above threshold is set to 1 and the rest are set to 0.
    """

    multihot_encoding = tf.cast(x > threshold, x.dtype)
    return multihot_encoding
