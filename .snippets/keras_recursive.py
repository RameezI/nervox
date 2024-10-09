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

import tempfile
import tensorflow as tf
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format=" %(message)s")
logger = logging.getLogger("keras_recursive")
logger.setLevel(logging.INFO)


class TerminalModule(tf.keras.Model):
    def __init__(self):
        """
        A module that takes two variables as input and computes some loss.
            Args:
                a (tf.Variable): The variable a tracked by the module.
                b (tf.Variable): The variable b tracked by the module.
        """
        super().__init__()
        self.a = tf.keras.layers.Dense(1)
        self.b = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.a(x)
        x = self.b(x)
        return x


if __name__ == "__main__":
    module = TerminalModule()
    input_sample = tf.constant(6.0, shape=(1, 1))
    module(input_sample)

    with tempfile.TemporaryDirectory() as export_dir:
        tf.saved_model.save(module, export_dir)
        [logger.info(item) for item in list(Path(export_dir).rglob("*"))]
        module.save(export_dir, save_format="tf")

        restored_module = tf.saved_model.load(export_dir)

    logger.info(
        f"Forward call of the keras module {module(tf.constant(6.0, shape= (1, 1)))}"
    )
    logger.info(
        f"Forward call of the restored module {restored_module(tf.constant(6.0, shape= (1, 1)))}"
    )
