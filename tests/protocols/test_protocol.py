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

from abc import ABC
import tensorflow as tf
from nervox.transcoders import Protocol
from tensorflow.python.framework import constant_op
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import combinations


class TestStrategy(Protocol, ABC):
    def __init__(self):
        super(TestStrategy, self).__init__()


class TestStrategyBaseNotInit(Protocol, ABC):
    def __init__(self):
        super(TestStrategyBaseNotInit).__init__()


class TestSupervisedClassification(tf.test.TestCase):
    
    def test_forgot_to_init_base(self):
        strategy = TestStrategyBaseNotInit()
        with self.assertRaises(RuntimeError):
            initialized = strategy.is_initialized
            assert not initialized
    
    def test_base_initialized(self):
        strategy = TestStrategy()
        self.assertTrue(strategy.is_initialized)


@combinations.generate(
    combinations.combine(
        distributor=[tf.distribute.MirroredStrategy(devices=["/cpu:0", "/cpu:1"]),
                     ],
        mode=['graph', 'eager']))
class TestMirroredStrategy(tf.test.TestCase):
    
    def testMirrored2CPUs(self, distributor):
        with distributor.scope():
            one_per_replica = distributor.run(lambda: constant_op.constant(1))
            num_replicas = distributor.reduce(reduce_util.ReduceOp.SUM, one_per_replica, axis=None)
            self.assertEqual(2, self.evaluate(num_replicas))
    
    def testLocalResultForDictionary(self, distributor):
        def model_fn():
            return {'a': constant_op.constant(1.), 'b': constant_op.constant(2.)}
        
        with distributor.scope():
            self.assertEqual(tf.distribute.get_strategy(), distributor)
            result = distributor.run(model_fn)
            got = self.evaluate(distributor.experimental_local_results(result))
            self.assertEqual(({'a': 1., 'b': 2.}, {'a': 1., 'b': 2.}), got)


if __name__ == "__main__":
    tf.test.main()
