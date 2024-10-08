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
