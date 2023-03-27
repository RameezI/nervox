"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import tensorflow as tf
from nervox.utils import capture_params
from itertools import zip_longest
l2 = tf.keras.regularizers.l2


class BottleneckLayers(tf.keras.Model):
    """A collection of 1x1 Conv layers followed by 3x3 Conv.
        Arguments:
        num_filters: number of filters passed to each convolution layer.
        data_format: "channels_first" or "channels_last"
    """
    def get_config(self):
        return getattr(self, 'params', dict())

    @capture_params
    def __init__(self, filters_count, data_format='channels_last'):
        super(BottleneckLayers, self).__init__(name="bottle_neck_layers")
        axis = -1 if data_format == "channels_last" else 1
        config_convolutions = {'padding': "same", 'use_bias': False, 'data_format': data_format,
                               'kernel_initializer': 'he_normal', 'kernel_regularizer': l2(1e-4)
                               }
        self.batchnorm_0 = tf.keras.layers.BatchNormalization(axis=axis, epsilon=1.001e-5)
        self.convolution_0 = tf.keras.layers.Conv2D(4*filters_count, 1, **config_convolutions)
        self.batchnorm_1 = tf.keras.layers.BatchNormalization(axis=axis, epsilon=1.001e-5)
        self.convolution_1 = tf.keras.layers.Conv2D(filters_count, 3, **config_convolutions)
    
    def call(self, x, training=True, mask=None):
        x = self.batchnorm_0(x, training=training)
        x = self.convolution_0(tf.nn.relu(x))
        x = self.batchnorm_1(x, training=training)
        x = self.convolution_1(tf.nn.relu(x))
        return x


class TransitionLayers(tf.keras.Model):
    """Transition Layers to reduce the number of features.
    Arguments:
    num_filters: number of filters passed to a convolution layer.
    data_format: "channels_first" or "channels_last"
    weight_decay: weight decay
    dropout_rate: dropout rate.
    """
    def get_config(self):
        return getattr(self, 'params', dict())
    
    @capture_params
    def __init__(self, num_filters, data_format='channels_last'):
        super(TransitionLayers, self).__init__(name="transition_layers")
        axis = -1 if data_format == "channels_last" else 1
        config_convolutions = {'padding': "same", 'use_bias': False, 'data_format': data_format,
                               'kernel_initializer': 'he_normal', 'kernel_regularizer': l2(1e-4)
                               }
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=axis, epsilon=1.001e-5)
        self.convolution = tf.keras.layers.Conv2D(num_filters, 1, **config_convolutions)
        self.avg_pool = tf.keras.layers.AveragePooling2D(data_format=data_format)
    
    def call(self, x, training=True, mask=None):
        output = self.batchnorm(x, training=training)
        output = self.convolution(tf.nn.relu(output))
        output = self.avg_pool(output)
        return output


class DenseBlock(tf.keras.Model):
    """Dense Block consisting of BottleNeck Layers where each block's output is concatenated with its input.
    Arguments:
    num_layers: Number of layers in each block.
    growth_rate: number of filters to add per conv block.
    data_format: "channels_first" or "channels_last"
    weight_decay: weight decay
    dropout_rate: dropout rate.
    """
    def get_config(self):
        return getattr(self, 'params', dict())
    
    @capture_params
    def __init__(self, cardinality, growth_rate, data_format='channels_last', dropout_rate=0.0):
        super(DenseBlock, self).__init__(name="DenseBlock")
        axis = -1 if data_format == "channels_last" else 1
        self.bottleneck_layers = []
        for _ in range(int(cardinality)):
            self.bottleneck_layers.append(BottleneckLayers(growth_rate, data_format))
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.concatenate = tf.keras.layers.Concatenate(axis=axis)
    
    def call(self, x, training=True, mask=None):
        for layer in self.bottleneck_layers:
            y = layer(x, training=training)
            x = self.concatenate([x, y])
        x = self.dropout(x, training=training)
        return x


class DenseNet(tf.keras.Model):
    """Creating the Densenet Architecture.
    Arguments:
    num_of_blocks:              number of dense blocks.
    num_layers_in_each_block:   number of convolution layers in each dense-block
    growth_rate:                number of filters to add per convolution layer.
    output_classes:             number of output classes.
    data_format:                "channels_first" or "channels_last"
    weight_decay:               weight decay
    rate:                       dropout rate.
    """
    def get_config(self):
        return getattr(self, 'params', dict())
    
    @capture_params
    def __init__(self,
                 filters_stem=64,
                 blocks_configuration=(6, 12, 24, 16),
                 growth_rate=32,
                 reduction_factor=0.5,
                 output_classes=101,
                 include_top=True,
                 dropout_rate=0.0,
                 data_format='channels_last',
                 ):
        """
        Args:
            stem_filters            number of 3x3, stride=2 filters, applied to the input at the stem of the network
            blocks_configuration:   provides configuration and count of blocks; the size of tuple indicates the block
                                    count while the elements corresponds to the block parametrization i.e. number of
                                    layers in each block.
            growth_rate:
            output_classes:
            data_format:
            reduction_factor:
            dropout_rate:           The dropout rate for the top layer only
        """
        super(DenseNet, self).__init__()
        
        axis = -1 if data_format == "channels_last" else 1
        config_convolutions = {'padding': "same", 'use_bias': False, 'data_format': data_format,
                               'kernel_initializer': 'he_normal', 'kernel_regularizer': l2(1e-4)
                               }
        
        self.include_top = include_top
        
        # stem convolution
        self.convolution_stem = tf.keras.layers.Conv2D(filters_stem, (7, 7), strides=2, **config_convolutions)
        self.batchnorm_stem = tf.keras.layers.BatchNormalization(axis=axis, epsilon=1.001e-5)
        self.pooling_stem = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        
        # dense block initialization
        self.dense_blocks = []
        self.transition_blocks = []
        
        channels_count = filters_stem
        for k, block_cardinality in enumerate(blocks_configuration):
            self.dense_blocks.append(DenseBlock(block_cardinality, growth_rate, data_format=data_format))
            channels_count = channels_count + growth_rate * block_cardinality
            if k < len(blocks_configuration) - 1:
                transition_filters = max(round(channels_count * reduction_factor), 8)
                self.transition_blocks.append(TransitionLayers(transition_filters, data_format=data_format))
                channels_count = transition_filters
        self.batchnorm_top = tf.keras.layers.BatchNormalization(axis=axis)
        if self.include_top:
            self.global_pooling_top = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)
            self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0.0 else None
            self.classifier = tf.keras.layers.Dense(output_classes, kernel_regularizer=l2(1e-4))
    
    def call(self, x, training=True, mask=None):
        x = self.convolution_stem(x)
        x = self.batchnorm_stem(x, training=training)
        x = self.pooling_stem(tf.nn.relu(x))
        for dense, transition in zip_longest(self.dense_blocks, self.transition_blocks, fillvalue=None):
            x = dense(x, training=training)
            x = transition(x, training=training) if transition is not None else x
        x = self.batchnorm_top(x, training=training)
        x = tf.nn.relu(x)
        if self.include_top:
            x = self.global_pooling_top(x)
            x = self.dropout(x, training=training) if self.dropout is not None else x
            x = self.classifier(x)
        return x
