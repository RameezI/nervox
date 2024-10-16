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
    
    
class MLDecoderAttention(tf.keras.Model):
    @capture_params
    def __init__(self, d_model: int, num_heads=8, dim_feedforward=2048, dropout_rate=0.1,
                 data_format='channels_last', layer_norm_eps=1e-5) -> None:
        super().__init__()
        axis = -1 if data_format == 'channels_last' else 1
        self.multihead_attn = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=d_model//num_heads,
                                                                 dropout=dropout_rate)
        # Implementation of Feedforward model
        self.dense_1 = tf.keras.layers.Dense(dim_feedforward)
        self.dense_2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(axis=axis, epsilon=layer_norm_eps)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(axis=axis, epsilon=layer_norm_eps)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(axis=axis, epsilon=layer_norm_eps)
        
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_3 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, targets: tf.Tensor, memory: tf.Tensor,
             attention_mask: Optional[tf.Tensor] = None,
             training: bool = None) -> tf.Tensor:
        x_a = self.layer_norm_1(targets + self.dropout_1(targets, training=training))
        x_b = self.multihead_attn(targets, memory, attention_mask=attention_mask, training=training)
        x_a = self.layer_norm_2(x_a + self.dropout_2(x_b, training=training))
        x_b = self.dense_2(self.dropout(tf.nn.relu(self.dense_1(x_a)), training=training))
        out = self.layer_norm_3(x_a + self.dropout_3(x_b, training=training))
        return out
    
    def to_json(self):
        params = getattr(self, 'params', {})
        return {'module': self.__module__,
                'class': type(self).__name__,
                'config': dict(params)}
