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
import numpy as np
from absl.testing import parameterized
from tensorflow.python.distribute import combinations
from nervox.data import DataStream

DATASETS = [
    tf.data.Dataset.from_tensor_slices(
        [np.random.rand(3, 3)] * 20,
    ),
    tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5]),
]


@combinations.generate(combinations.combine(dataset=DATASETS))
class TestDataStream(tf.test.TestCase, parameterized.TestCase):
    def test_stream_iterations(self, dataset):
        stream = DataStream(dataset)
        for (
            a,
            b,
        ) in zip(dataset, stream):
            self.assertAllClose(a, b)

    def test_stream_mutability(self, dataset):
        stream = DataStream(dataset)
        dataset = dataset.batch(5)
        with self.assertRaises(AssertionError):
            for (
                a,
                b,
            ) in zip(dataset, stream):
                self.assertAllClose(a, b)

    def test_batch_reassignment(self, dataset):
        batched_dataset = dataset.batch(5)
        batched_dataset_ref = dataset.batch(4)
        stream = DataStream(batched_dataset, batch_size=4)
        for (
            a,
            b,
        ) in zip(batched_dataset_ref, stream):
            self.assertAllClose(a, b)

    def test_stream_distribution(self, dataset):
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        dataset = dataset.with_options(options)
        stream = DataStream(dataset, batch_size=4)
        distributor = tf.distribute.MirroredStrategy()
        with distributor.scope():
            distributed_stream = stream.distribute()
        self.assertTrue(isinstance(stream.as_dataset(), tf.data.Dataset))
        self.assertTrue(
            isinstance(
                distributed_stream.as_dataset(), tf.distribute.DistributedDataset
            )
        )


if __name__ == "__main__":
    tf.test.main()
