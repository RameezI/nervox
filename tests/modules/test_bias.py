# Copyright © 2023 Rameez Ismail - All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author(s): Rameez Ismail
# Email(s):  rameez.ismaeel@gmail.com
# 
# This code is adapted from Sonnet by DeepMind: 
# https://github.com/deepmind/sonnet
# The project is licensed under the Apache-2.0. 
# You may obtain a copy of the license at:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for nervox.modules.bias."""

import tensorflow as tf
from nervox.modules.bias import Bias
from nervox.utils import initializers 


class BiasTest(tf.test.TestCase):

  def test_dims_scalar(self):
    mod = Bias(dims=())
    mod(tf.ones([1, 2, 3, 4]))
    self.assertEmpty(mod._bias.shape)

  def test_dims_custom(self):
    b, d1, d2, d3 = range(1, 5)
    mod = Bias(dims=[1, 3])
    out = mod(tf.ones([b, d1, d2, d3]))
    self.assertEqual(mod._bias.shape, [d1, 1, d3])
    self.assertEqual(out.shape, [b, d1, d2, d3])

  def test_dims_negative_out_of_order(self):
    mod = Bias(dims=[-1, -2])
    mod(tf.ones([1, 2, 3]))
    self.assertEqual(mod._bias.shape, [2, 3])

  def test_dims_invalid(self):
    mod = Bias(dims=[1, 5])
    with self.assertRaisesRegex(ValueError,
                                "5 .* out of range for input of rank 3"):
      mod(tf.ones([1, 2, 3]))

  def test_b_init_defaults_to_zeros(self):
    mod = Bias()
    mod(tf.ones([1, 1]))
    self.assertAllEqual(mod._bias.read_value(), tf.zeros_like(mod._bias))

  def test_b_init_custom(self):
    mod = Bias(initializer=initializers.Ones())
    mod(tf.ones([1, 1]))
    self.assertAllEqual(mod._bias.read_value(), tf.ones_like(mod._bias))

  def test_name(self):
    mod = Bias(name="foo")
    self.assertEqual(mod.name, "foo")
    mod(tf.ones([1, 1]))
    self.assertEqual(mod._bias.name, "foo/bias:0")

  def test_multiplier(self):
    mod = Bias(initializer=initializers.Ones())
    out = mod(tf.ones([1, 1]), multiplier=-1)
    self.assertAllEqual(tf.reduce_sum(out), 0)


if __name__ == "__main__":
  tf.test.main()