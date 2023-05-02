import tensorflow as tf
from typing import Optional
from nervox.utils import capture_params

__all__ = []

class StackedDense(tf.keras.Model):
    def __init__(self, output_units_per_group: int, groups_count:int, **kwargs) -> None:
        super().__init__()
        self.dense_layers = []
        for k in range(groups_count):
            self.dense_layers.append(tf.keras.layers.Dense(output_units_per_group, **kwargs))
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        out_list = []
        for k, dene_layer in enumerate(self.dense_layers):
            out_k = dene_layer(x[:, k, :])
            out_list.append(out_k)
        out = tf.stack(out_list, axis=1)
        out = tf.reshape(out, (tf.shape(out)[0], -1))
        # flattened output (Batch, output_units_per_group x group_count)
        return out
    
    def to_json(self):
        params = getattr(self, 'params', {})
        return {'module': self.__module__,
                'class': type(self).__name__,
                'config': dict(params)}
    

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
