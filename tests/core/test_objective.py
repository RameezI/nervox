"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""
import tempfile
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.distribute import combinations
from nervox.core import Protocol
from nervox.core import Objective
from nervox.data import DataStream
from nervox import Trainer

# L2 regularization method
l2 = tf.keras.regularizers.l2


class LossMismatchError(Exception):
    pass


class DummyModel(tf.keras.Model):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv_layer = tf.keras.layers.Conv2D(16, 3, kernel_regularizer=l2(1e-4))
        self.classifier = tf.keras.Sequential([tf.keras.layers.Flatten(),
                                               tf.keras.layers.Dense(10)])
    
    def call(self, x):
        x = self.conv_layer(x)
        x = self.classifier(x)
        return x


class DummyProtocol(Protocol):
    @staticmethod
    def objective_configurator(model):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss)
        return Objective(loss, optimizer)
    
    def train_step(self, batch):
        model = self.module
        images, labels = batch['image'], batch['label']
        
        predictions = model(images)
        reg_losses = model.losses
        
        loss_ref = model.compiled_loss(labels, predictions,
                                       regularization_losses=reg_losses)

        loss = self.objective.compute_loss(labels, predictions) + reg_losses
        
        tf.debugging.assert_equal(loss_ref, loss)

    def evaluate_step(self, batch):
        pass

    def predict_step(self, batch):
        pass


# TODO: Check for multi_gpu distributors as well
@combinations.generate(
    combinations.combine(run_eagerly=[True, False]))
class TestObjective(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        image = tf.expand_dims(tf.random.normal((32, 32, 3)), axis=0)
        label = tf.expand_dims(tf.one_hot(0, 10), axis=0)
        
        dataset_train = tf.data.Dataset.from_tensor_slices({'image': image,
                                                            'label': label})
        self.dummy_train_stream = DataStream(dataset_train, batch_size=1)
        self.protocol = DummyProtocol()
    
    def test_strategy_call(self, run_eagerly):
        
        with tempfile.TemporaryDirectory() as tempdir:
            trainer = Trainer(self.dummy_train_stream, logs_dir=tempdir)
            trainer.push_module(DummyModel, alias='model')
            
            try:
                trainer.spin(self.protocol, max_epochs=1, run_eagerly=run_eagerly)
            except tf.errors.InvalidArgumentError:
                self.fail('The loss from keras compiled model is'
                          'not same as computed from the objective!')


if __name__ == "__main__":
    tf.test.main()
