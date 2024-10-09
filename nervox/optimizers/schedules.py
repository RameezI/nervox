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

import math
import tensorflow as tf
from typing import Optional
from nervox.utils import capture_params


class LinearWarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule."""

    @capture_params
    def __init__(
        self,
        initial_learning_rate: float,
        warmup_steps: int,
        decay_steps: int,
        alpha: float = 0.0,
        name: Optional[str] = None,
        dtype: tf.DType = tf.float32,
    ):

        super().__init__()

        self.initial_learning_rate = tf.convert_to_tensor(
            initial_learning_rate, dtype=dtype, name="initial_learning_rate"
        )
        self.warmup_steps = tf.cast(warmup_steps, dtype)
        self.decay_steps = tf.cast(decay_steps, dtype)
        self.alpha = tf.cast(alpha, dtype)
        self.name = name
        self.dtype = dtype

    @tf.function
    def __call__(self, step: tf.constant):

        global_step = tf.cast(step, self.dtype)

        if global_step <= self.warmup_steps:
            linear_ramp = global_step / self.warmup_steps
            factor = (1 - self.alpha) * linear_ramp + self.alpha

        else:
            decay_step = tf.minimum(global_step - self.warmup_steps, self.decay_steps)
            progress = decay_step / self.decay_steps
            cosine_decayed = 0.5 * (
                1.0 + tf.cos(tf.constant(math.pi, dtype=self.dtype) * progress)
            )
            factor = (1 - self.alpha) * cosine_decayed + self.alpha

        return tf.multiply(self.initial_learning_rate, factor)

    def get_config(self):
        params = getattr(self, "params", {})
        return params
