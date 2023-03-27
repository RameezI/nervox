"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import tensorflow as tf
from enum import Enum
from typing import Tuple, Dict, Union, Any

from nervox.core.protocol import Protocol
from nervox.core.objective import Objective
from nervox.metrics.classification import (
    onehot_transform,
    AccuracyScore,
    AveragingMode,
)

from nervox.losses import CrossEntropy


class Classification(Protocol):
    def __init__(self, supervision_keys: Tuple[str, str]):
        """
        Args:
            supervision_keys:   An ordered pair of strings; where the first element represents the key for the
                                input data while the second element represent the key for the label.
        """

        super(Classification, self).__init__()
        self.supervision_keys = supervision_keys

    @staticmethod
    def configure():
        """
        Configure method provides a placeholder for defining/configuring the objective/objectives for the strategy.
        The configure method can be overridden through the `configure_cb` argument of the protocol constructor.
        For more permissive customization, user must write a new protocol or derive from an existing one.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        categorical_xentropy = CrossEntropy(transform=tf.nn.sigmoid)

        accuracy = AccuracyScore(onehot_transform, averaging_mode=AveragingMode.SAMPLE)
        objective = Objective(
            categorical_xentropy, optimizer=optimizer, metrics=[accuracy]
        )
        return objective

    def train_step(self, batch: Dict[str, tf.Tensor]) -> None:
        """
        Args:
            batch:                      A batch of data with various features
        """

        data, labels = batch[self.supervision_keys[0]], batch[self.supervision_keys[1]]

        # aliases
        encoder = self.models["encoder"]
        decoder = self.models["decoder"]
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

    def evaluate_step(self, batch: Dict[str, tf.Tensor]):

        data, labels = batch[self.supervision_keys[0]], batch[self.supervision_keys[1]]

        # aliases
        encoder = self.models["encoder"]
        classifier = self.models["decoder"]
        objective = self.objective

        encoding = encoder(data, training=False)
        predictions = classifier(encoding, training=False)
        regularization_loss = encoder.losses + classifier.losses

        # calculate loss and update loss metrics
        objective.compute_loss(labels, predictions) + regularization_loss
        objective.update_metrics(labels, predictions)

    # The default serving signature
    def predict_step(self, batch: Union[tf.Tensor, Dict[str, tf.Tensor]]):
        data = batch[self.supervision_keys[0]] if isinstance(batch, Dict) else batch
        encoding = self.models["encoder"](data, training=False)
        predictions = self.models["decoder"](encoding, training=False)
        return predictions
