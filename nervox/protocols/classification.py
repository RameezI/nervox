"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import tensorflow as tf
from typing import Tuple, Dict, Union, Callable
from nervox.core.protocol import Protocol
from nervox.core import Objective

# objective configurator
from nervox.losses import CrossEntropy
from nervox.metrics.classification import AccuracyScore, AveragingMode
from nervox.transforms import onehot_transform


class Classification(Protocol):
    def __init__(self, supervision_keys: Tuple[str, str], **kwargs):
        """
        Args:
            supervision_keys:   An ordered pair of strings; where the first element represents the key for the
                                input data while the second element represents the key for the label.
        """
        super().__init__(**kwargs)
        self.supervision_keys = supervision_keys

    # @staticmethod
    # def objective_configurator():
    #     """
    #     Configure method provides a placeholder for defining/configuring the objective(s) for the strategy.
    #     The configure method can be overridden through the `configurator` argument of the constructor.
    #     The configurator is user supplied function/functor that returns objective(s) for the training.
    #     For more permissive customization, user must write a new protocol or derive from an existing one.
    #     """
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    #     xentropy = CrossEntropy(transform=tf.nn.sigmoid)
    #     accuracy = AccuracyScore(onehot_transform, averaging_mode=AveragingMode.SAMPLE)
    #     objective = Objective(xentropy, optimizer=optimizer, metrics=[accuracy])
    #     return objective

    def train_step(self, batch: Dict[str, tf.Tensor]) -> None:
        """
        Args:
            batch:  A batch of data with various features
        """

        data, labels = batch[self.supervision_keys[0]], batch[self.supervision_keys[1]]

        # aliases
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        objective = self.objective
        optimizer = self.objective.optimizer

        with tf.GradientTape() as tape:
            encoding = encoder(data, training=True)
            predictions = decoder(encoding, training=True)
            regularization_loss = encoder.losses + decoder.losses
            loss = objective.compute_loss(labels, predictions) + regularization_loss

        # apply optimization to the trainable variables
        parameters = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, parameters)
        optimizer.apply_gradients(zip(gradients, parameters))
        objective.update_metrics(labels, predictions)
        return {metric.name: metric.result() for metric in self.metrics}

    def evaluate_step(self, batch: Dict[str, tf.Tensor])-> None:
        data, labels = batch[self.supervision_keys[0]], batch[self.supervision_keys[1]]

        # aliases
        encoder = self.modules["encoder"]
        classifier = self.modules["decoder"]
        objective = self.objective

        encoding = encoder(data, training=False)
        predictions = classifier(encoding, training=False)
        regularization_loss = encoder.losses + classifier.losses

        # calculate loss and update loss metrics
        objective.compute_loss(labels, predictions) + regularization_loss
        objective.update_metrics(labels, predictions)
        return {metric.name: metric.result() for metric in self.metrics}

    # The default serving signature
    def predict_step(self, batch: Union[tf.Tensor, Dict[str, tf.Tensor]]):
        data = batch[self.supervision_keys[0]] if isinstance(batch, Dict) else batch
        encoding = self.modules["encoder"](data, training=False)
        predictions = self.modules["decoder"](encoding, training=False)
        return predictions
