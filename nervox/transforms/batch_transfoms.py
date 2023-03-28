import tensorflow as tf
from nervox.utils.types import TensorLike

def onehot_transform(x: TensorLike, axis=-1) -> tf.Tensor:
    indices = tf.argmax(x, axis=axis)
    onehot_predictions = tf.one_hot(indices, depth=x.shape[axis])
    return onehot_predictions

def multihot_transform(x: TensorLike, threshold=0.5) -> tf.Tensor:
    multihot_encoding = tf.cast(x > threshold, x.dtype)
    return multihot_encoding