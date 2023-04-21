"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import tensorflow as tf
from absl.testing import parameterized
from nervox.data import DataStream
from nervox.protocols import Classification
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from nervox.modules.visencoders import GlobalAvgPoolDecoder


distributors = [tf.distribute.get_strategy(),
                #strategy_combinations.one_device_strategy,
                #strategy_combinations.mirrored_strategy_with_one_cpu,
                #strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
                #strategy_combinations.one_device_strategy_gpu,
                #strategy_combinations.mirrored_strategy_with_one_gpu,
                ]


class DummyModel(tf.keras.Model):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv_layer = tf.keras.layers.Conv2D(16, 3)
        
    def call(self, x):
        x = self.conv_layer(x)
        return x


@combinations.generate(
    combinations.combine(
        run_eagerly=[True, False],
        distributor=distributors))
class TestClassification(tf.test.TestCase, parameterized.TestCase):
 
    def setUp(self):
        image = tf.expand_dims(tf.random.normal((32, 32, 3)), axis=0)
        label = tf.expand_dims(tf.one_hot(0, 10), axis=0)
        dataset_train = tf.data.Dataset.from_tensor_slices({'image': image,
                                                            'label': label})
        self.dummy_train_stream = DataStream(dataset_train, batch_size=1)
        self.protocol = Classification(supervision_keys=('image', 'label'))
    
    def test_training_loop(self, run_eagerly, distributor):
        with distributor.scope():
            self.protocol.compile({'encoder': DummyModel(),
                                   'decoder': GlobalAvgPoolDecoder(output_units=10)})
            try:
                self.protocol.train(self.dummy_train_stream, run_eagerly=run_eagerly)
            except Exception as e:
                self.fail(msg=str(e))

    # TODO: Write equivalence test, check if metrics, generated by the strategy with various
    #  distribution schemes, are equivalent
    def test_equivalence(self, run_eagerly, distributor):
        pass


if __name__ == "__main__":
    tf.test.main()
