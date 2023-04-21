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