"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import tensorflow as tf
from typing import Union
from nervox.utils import capture_params
from nervox.modules.ml_decoder import MLDecoderAttention, StackedDense

l2 = tf.keras.regularizers.l2
GlobalAveragePooling = tf.keras.layers.GlobalAveragePooling2D


class GlobalAvgPoolDecoder(tf.keras.Model):
    """A classification head with each output unit representing a specific class/label,
    This classifier flattens the incoming feature maps and feed it to a dense read-out layer
    """

    @capture_params
    def __init__(
        self,
        output_units: int,
        fcn_units: Union[None, int] = None,
        dropout_rate=0.5,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.global_avg_pooling = GlobalAveragePooling(
            data_format=data_format, keepdims=True
        )
        self.flat_layer = tf.keras.layers.Flatten(data_format=data_format)
        self.top_fcn = (
            tf.keras.layers.Dense(fcn_units) if fcn_units is not None else None
        )
        self.dropout = (
            tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        )
        self.classifier = tf.keras.layers.Dense(
            output_units, kernel_regularizer=l2(1e-4)
        )

    def call(self, x, training=False):
        x = self.global_avg_pooling(x) if self.global_avg_pooling is not None else x
        x = self.flat_layer(x)
        x = tf.nn.relu(self.top_fcn(x)) if self.top_fcn is not None else x
        x = self.dropout(x, training=training)
        return self.classifier(x)


class MLDecoder(tf.keras.Model):
    @capture_params
    def __init__(
        self,
        output_units: int,
        num_of_groups: int = 111,
        embedding_dim: int = 768,
        n_heads: int = 8,
        dim_feedforward=2048,
        dropout_rate_attention: float = 0.1,
        dropout_readout: float = 0.5,
        query_embeddings_trainable: bool = False,
        num_layers=1,
        data_format="channels_last",
    ) -> None:
        super().__init__()

        self.queries = tf.constant(tf.range(0, num_of_groups))

        self.embedding_spatial = tf.keras.layers.Dense(embedding_dim)

        # learnable queries
        self.embedding_queries = tf.keras.layers.Embedding(
            num_of_groups,
            embedding_dim,
            input_length=num_of_groups,
            trainable=query_embeddings_trainable,
        )

        self.attention_layers = [
            MLDecoderAttention(
                embedding_dim,
                num_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout_rate=dropout_rate_attention,
                data_format=data_format,
            )
            for _ in range(num_layers)
        ]

        group_factor = int((output_units / num_of_groups) + 0.999)
        groups = int(output_units // group_factor)
        self.dropout_readout = tf.keras.layers.Dropout(dropout_readout)
        self.read_out_layer = StackedDense(
            group_factor, groups_count=groups, kernel_regularizer=l2(1e-4)
        )

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
        """
        Args:
            x:              An embedding tensor of shape (B x H x W xD)
            training:       A boolean representing if the model is invoked in training mode,
                            when the value is true, or inference mode otherwise.

        Returns:            Logits tensor

        """
        x_shape = tf.shape(x)
        x = tf.reshape(x, (x_shape[0], -1, x_shape[-1]))  # (B x K x D)
        x = tf.nn.relu(self.embedding_spatial(x))

        queries = tf.tile(
            tf.expand_dims(self.queries, axis=0), multiples=[x_shape[0], 1]
        )
        queries = self.embedding_queries(queries)

        for attention_layer in self.attention_layers:
            x = attention_layer(queries, x, training=training)

        x = self.dropout_readout(x, training=training)
        logits = self.read_out_layer(x)
        return logits
