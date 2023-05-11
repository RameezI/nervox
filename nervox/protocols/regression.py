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

"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import tensorflow as tf
from nervox.core.strategy import Protocol
from typing import List
from .classification import Classification
from nervox.losses import CumulativeMAE

#
# class Regression(Classification):
#     def __init__(self, supervised_keys, ):
#         """
#         Args:
#             supervised_keys:    The supervised keys for regression.
#         """
#         super(Regression, self).__init__(supervised_keys)
#         self.supervised_keys = supervised_keys
#
#     def configure(self, model,
#                   optimizer: tf.keras.optimizers.Optimizer = None,
#                   losses: List[tf.keras.losses.Loss] = None,
#                   metrics: List[tf.keras.metrics.Metric] = None):
#
#         """Compile model"""
#         if optimizer is None:
#             optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#         else:
#             optimizer = type(optimizer)(**optimizer.get_config())
#
#         if losses is None:
#             mean_absolute_error = tf.keras.metrics.MeanAbsoluteError(name='mae')
#             losses = [mean_absolute_error]
#         else:
#             losses = [type(loss)(**loss.get_config())
#                       for loss in losses]
#
#         if metrics is None:
#             # metrics
#             mae = tf.keras.metrics.MeanAbsoluteError(name='mae')
#             metrics = [mae]
#
#         else:
#             metrics = [type(metric)(**metric.get_config())
#                        for metric in metrics]
#
#         model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
#
#
# class RegressionSIMO(Strategy):
#
#     def __init__(self, input_key='image',
#                  regressands=('weights/ingredient',
#                               'calories/ingredient',
#                               'macros/ingredient')):
#         super(RegressionSIMO, self).__init__()
#         self.input_key = input_key
#         self.regressands = regressands
#
#     def configure(self, model,
#                   optimizer: tf.keras.optimizers.Optimizer = None,
#                   losses: List[tf.keras.losses.Loss] = None):
#
#         if optimizer is None:
#             optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#         else:
#             optimizer = type(optimizer)(**optimizer.get_config())
#
#         if losses is None:
#             losses = [tf.keras.losses.MeanAbsoluteError()]
#         else:
#             losses = [type(loss)(**loss.get_config()) for loss in losses]
#
#         # metrics
#         metrics = tf.keras.metrics.MeanAbsoluteError(name='mae')
#         model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
#
#     def train_step(self, batch):
#         pass
#
#     def validation_step(self, batch):
#         pass
#
#     def test_step(self, batch):
#         pass
#
#     def call(self, batch):
#         pass
#
#
# if __name__ == '__main__':
#     reg = ('weights/ingredients', 'calories/ingredients', 'macros/ingredients')
