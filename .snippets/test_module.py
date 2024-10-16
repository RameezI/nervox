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
import nervox as nx


class SimpleDense(nx.Module):

    def __init__(self, units=32):
        self.units_simple = units

    # def build(self, input_shape):
    #     kernel = tf.random.normal(
    #         shape=(input_shape[-1], self.units),
    #         dtype=self.dtype
    #     )

    #     bias = tf.zeros(shape=(self.units,), dtype=self.dtype)

    #     self.kernel = tf.Variable(initial_value=kernel, trainable=True)
    #     self.bias = tf.Variable(initial_value=bias, trainable=True)

    # def compute(self, inputs):
    #     return tf.matmul(inputs, self.kernel) + self.bias


class ComplexDense(SimpleDense):
    def __init__(self, units_simple=33, units_complex=64):
        super().__init__(units_simple)
        self.units_complex = units_complex


module = ComplexDense(24, 64)
print(module.params)
print(module.units_simple, module.units_complex)
print(module.name)
print(module.dtype)
