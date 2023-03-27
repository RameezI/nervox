"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import tensorflow as tf
import numpy as np
from nervox.utils.types import TensorLike
from typing import Union
from nervox.utils import capture_params

# Aliases
Loss = tf.keras.losses.Loss
Reduction = tf.keras.losses.Reduction


def reduce_weighted_loss(weighted_losses, reduction=Reduction.SUM_OVER_BATCH_SIZE):
    """Reduces the individual weighted loss measurements."""
    if reduction == Reduction.NONE:
        loss = weighted_losses
    else:
        loss = tf.reduce_sum(weighted_losses)
        if reduction == Reduction.SUM_OVER_BATCH_SIZE:
            loss = tf.reduce_mean(weighted_losses)
    return loss


@tf.__internal__.dispatch.add_dispatch_support
def binary_cross_entropy(y_true: TensorLike, y_pred: TensorLike,
                         weights: Union[None, TensorLike] = None,
                         label_smoothing: float = 0.0,
                         gamma_neg: float = 0.0,
                         gamma_pos: float = 0.0,
                         focus_credit_pos: float = 0.0,
                         focus_credit_neg: float = 0.0,
                         from_logits=True,
                         ):
    """Computes the binary cross entropy loss.
    Standalone usage:
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = binary_cross_entropy(y_true, y_pred)
    >>> loss.numpy()
    array([0.916 , 0.714], dtype=float32)
    Args:
      y_true:               Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred:               The predicted values. shape = `[batch_size, d0, .. dN]`.

      weights:              The relative weight for each logit/prediction; the shape must allow
                            broadcasting to the y_true and y_pred
      from_logits:          Whether `y_pred` is expected to be a logits tensor.

      label_smoothing:      Float in [0, 1]. If > `0` then smooth the labels by
                            squeezing them towards 0.5 That is, using `1. - 0.5 * label_smoothing`
                            for the target class and `0.5 * label_smoothing` for the non-target class.

      gamma_pos:            Controls the down-weighing of the penality for positive detection with a
                            propability p+=(0, 1); based on the current focus of the network
                             == (1 - p+)^gamma

                            BOUNDARY CASES AND FOCUS:
                                            p+==0;  => (1-0)^gamma >=1   :: increase the penality
                                            P+==1;  => (1-1)^gamma ==0   :: penality term is zero

     gamma_neg:            Controls the down-weighing of the penality for negative detection with a
                           propability p+=(0, 1); based on the current focus of the network
                            == (p+)^gamma

                           BOUNDARY CASES AND FOCUS:
                                    p+==0;  => (0)^gamma ==0   :: penality term is zero
                                    P+==1;  => (1)^gamma ==0   :: increase the penality

    focus_credit_pos:     Adds some constant slack to the p- . This is done to ensure an increased focus,
                          by a constant value,  for positive class detection.

    focus_credit_neg:     Adds some constant slack to the p+ . This is done to ensure an increased focus,
                          by a constant value,  for negative class detection.

    Returns:
      Binary cross entropy loss value. shape = `[batch_size, d0, .. dN-1]`.
    """
    
    if label_smoothing:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    
    if from_logits:
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    
    else:
        epsilon_ = tf.constant(np.finfo(float).eps, y_pred.dtype)
        y_true = tf.clip_by_value(y_true, epsilon_, 1. - epsilon_)
        bce = y_true * tf.math.log(y_pred + epsilon_)
        bce += (1 - y_true) * tf.math.log(1 - y_true + epsilon_)
        bce = -bce
    
    if weights is not None:
        bce = bce * weights
    
    bce = tf.reduce_mean(bce, -1)
    return bce


@tf.__internal__.dispatch.add_dispatch_support
def cross_entropy(y_true: TensorLike, y_pred: TensorLike,
                  weights: Union[None, TensorLike] = None,
                  label_smoothing: float = 0.0,
                  from_logits=True,
                  axis=-1
                  ):
    """Computes the binary cross entropy loss.
    Standalone usage:
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = cross_entropy(y_true, y_pred)
    >>> loss.numpy()
    array([0.916 , 0.714], dtype=float32)
    Args:
      y_true:               Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred:               The predicted values. shape = `[batch_size, d0, .. dN]`.
      weights:              The relative weight for each logit/prediction; the shape must allow
                            broadcasting to the y_true and y_pred
      from_logits:          Whether `y_pred` is expected to be a logits tensor.
      label_smoothing:      Float in [0, 1]. If > `0` then smooth the labels by
                            squeezing them towards 0.5 That is, using `1. - 0.5 * label_smoothing`
                            for the target class and `0.5 * label_smoothing` for the non-target class.
      axis:                 The axis over which the cross entropy is calculated; default value is -1
    Returns:
      Binary cross entropy loss value. shape = `[batch_size, d0, .. dN-1]`.
    """
    
    if label_smoothing:
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
    
    if from_logits:
        xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred,
                                                           axis=axis)
    
    else:
        epsilon_ = tf.constant(np.finfo(float).eps, y_pred.dtype)
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=axis, keepdims=True)
        output = tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_)
        xentropy = -tf.reduce_sum(y_true * tf.math.log(output), axis)
    
    if weights is not None:
        xentropy = xentropy * weights
    xentropy = tf.reduce_mean(xentropy, -1)
    
    return xentropy


class BinaryCrossEntropy(Loss):
    def __init__(self, from_logits: bool = True, label_smoothing=0.0,
                 reduction=Reduction.AUTO, name='binary_cross_entropy'):
        """Initializes `BinaryCrossEntropy` instance.
          Args:
            from_logits:        Whether to interpret `y_pred` as a tensor of logits or probabilities.
                                Default value for `from_logits` is `True`

            label_smoothing:    Float in [0, 1]. When 0, no smoothing occurs. When > 0,
                                we compute the loss between the predicted labels and a smoothed version
                                of the true labels, where the smoothing squeezes the labels towards 0.5.
                                Larger values of `label_smoothing` correspond to heavier smoothing.

            reduction:          Type of `tf.keras.losses.Reduction` to apply to
                                loss. Default value is `AUTO`. `AUTO` indicates that the reduction
                                option will be determined by the usage context. For almost all cases
                                this defaults to `SUM_OVER_BATCH_SIZE`.

            name:               Name for the op. Defaults to 'binary_cross_entropy'.
          """
        super().__init__(reduction, name)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
    
    def __call__(self, y_true: TensorLike, y_pred: TensorLike,
                 class_weights: TensorLike = None,
                 sample_weights: TensorLike = None):
        losses = binary_cross_entropy(y_true, y_pred,
                                      weights=class_weights,
                                      from_logits=self.from_logits,
                                      label_smoothing=self.label_smoothing)
        
        weighted_losses = losses * sample_weights \
            if sample_weights is not None else losses
        
        reduced_loss = reduce_weighted_loss(weighted_losses)
        return reduced_loss


class CrossEntropy(Loss):
    def __init__(self, from_logits: bool = True, label_smoothing=0.0, sparse_labels=False,
                 reduction=Reduction.AUTO, name='cross_entropy'):
        """Initializes `CrossEntropy` instance.
          Args:
            from_logits:        Whether to interpret `y_pred` as a tensor of logits or probabilities.
                                Default value for `from_logits` is `True`

            label_smoothing:    Float in [0, 1]. When 0, no smoothing occurs. When > 0,
                                we compute the loss between the predicted labels and a smoothed version
                                of the true labels, where the smoothing squeezes the labels towards 0.5.
                                Larger values of `label_smoothing` correspond to heavier smoothing.

            reduction:          Type of `tf.keras.losses.Reduction` to apply to
                                loss. Default value is `AUTO`. `AUTO` indicates that the reduction
                                option will be determined by the usage context. For almost all cases
                                this defaults to `SUM_OVER_BATCH_SIZE`.

            name:               Name for the op. Defaults to 'binary_cross_entropy'.
          """
        super().__init__(reduction, name)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.sparse_labels = sparse_labels
    
    def __call__(self, y_true: TensorLike, y_pred: TensorLike,
                 class_weights: TensorLike = None,
                 sample_weights: TensorLike = None):
        if self.sparse_labels:
            raise NotImplementedError
        
        losses = cross_entropy(y_true, y_pred, weights=class_weights,
                               from_logits=self.from_logits,
                               label_smoothing=self.label_smoothing,
                               axis=-1)
        
        weighted_losses = losses * sample_weights \
            if sample_weights is not None else losses
        
        reduced_loss = reduce_weighted_loss(weighted_losses)
        return reduced_loss


def hamming_distance(labels: TensorLike, predictions: TensorLike) -> tf.Tensor:
    """Computes hamming distance.
    Hamming distance is for comparing two binary strings.
    It is the number of bit positions in which two bits
    are different.
    Args:
        labels: target values.
        predictions: predicted values.
    Returns:
        hamming distance: float.
    Usage:
    >>> y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=np.int32)
    >>> y_pred = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1], dtype=np.int32)
    >>> hamming_distance(y_true, y_pred).numpy()
    0.3
    """
    result = tf.not_equal(labels, predictions)
    not_eq = tf.reduce_sum(tf.cast(result, tf.float32))
    ham_distance = tf.math.divide_no_nan(not_eq, len(result))
    return ham_distance


class HammingLoss(Loss):
    def call(self, y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
        """Computes hamming loss.
        Hamming loss is the fraction of wrong labels to the total number
        of labels.In a multi-label classification, hamming loss penalizes
        only the individual labels.
        Args:
            y_true: actual target value.
            y_pred: predicted target value.
        Returns:
            hamming loss: float.
        """
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        nonzero = tf.cast(tf.math.count_nonzero(y_true - y_pred, axis=-1), tf.float32)
        return nonzero / y_true.get_shape()[-1]


class CumulativeMAE(tf.keras.losses.Loss):
    
    def get_config(self):
        return getattr(self, 'params', dict())
    
    @capture_params
    def __init__(self, accumulation_axis=-1, **kwargs):
        super().__init__(name=kwargs.pop('name', None))
        self.axis = accumulation_axis
        self.tf2_mae_loss = tf.keras.losses.MeanAbsoluteError(**kwargs)
    
    def call(self, y_true: TensorLike, y_pred: TensorLike,
             sample_weight: Union[None, TensorLike] = None) -> tf.Tensor:
        """Computes cumulative mae loss.
        Args:
            y_true: actual target value.
            y_pred: predicted target value.
            sample_weight: Wight for each sample, when None all samples carry an equal weight.
        Returns:
            cumulative mae loss: float.
        """
        y_true = tf.reduce_sum(y_true, axis=self.axis)
        y_pred = tf.reduce_sum(y_pred, axis=self.axis)
        return self.tf2_mae_loss(y_true, y_pred, sample_weight)
