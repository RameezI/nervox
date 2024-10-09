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

import tensorflow as tf
from enum import Enum

# Aliases
Reduction = tf.keras.losses.Reduction


def scaled_loss(loss_value):
    """Scales and returns the given loss value by the number of replicas.
    Sum_over_batch size reduction over replicas:
        losses = [sum(loss_k)/batch_size_k for loss_k in replicas]
     :: actual_batch_size = k*batch_size
     => correction:  loss = loss/k
    """
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    if num_replicas > 1:
        loss_value *= 1.0 / num_replicas
    return loss_value


class DistributedLossWrapper:
    """A container that wraps the loss object to correctly
    perform reduction while training on multiple replicas in sync"""

    def __init__(self, loss: tf.keras.losses.Loss):
        super(DistributedLossWrapper, self).__init__()
        loss._allow_sum_over_batch_size = True
        self._loss = loss

    def __call__(self, *args, **kwargs):
        if self._loss.reduction in [Reduction.SUM_OVER_BATCH_SIZE]:
            loss = scaled_loss(self._loss(*args, **kwargs))
        else:
            loss = self._loss(*args, **kwargs)
        return loss


# class AverageLoss(Metric):
#     """
#     Use this AverageSum metric when the loss reduction is set to `sum` or when loss is already scaled
#     correctly for each replica. This metric tracks/maintain mean= (sum/count) over the batches,
#     while computes `sum` over the replicas.
#     """
#
#     def __init__(self, *args, **kwargs):
#         super(AverageLoss, self).__init__(**kwargs)
#         self.mean = tf.keras.metrics.Mean(name='tf2_mean')
#
#     def update(self, values, sample_weight=None):
#         # scale values by the ins-sync replica count to correct for
#         # the 'Mean' loss metrics counting each replica as a batch
#         values *= tf.distribute.get_strategy().num_replicas_in_sync
#         self.mean(values, sample_weight)
#
#     def result(self):
#         return self.mean.result()
