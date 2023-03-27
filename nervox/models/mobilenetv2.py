"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import tensorflow as tf
from nervox.utils import capture_params

l2 = tf.keras.regularizers.l2

class BottleneckLayers(tf.keras.Model):
    """A collection of 1x1 convolution [expansion] --> 3x3 channel-wise convolution --> 1x1 convolution[squeeze]
    """
    def get_config(self):
        return getattr(self, 'params', dict())
    
    def __init__(self,
                 expanded_filters_count,
                 projection_filters_count,
                 stride=1.0,
                 data_format='channels_last',
                 activation=tf.nn.relu6,
                 weight_decay=4e-5):
        """
        Args:
            expanded_filters_count:     number of filters applied to the input.
            projection_filters_count:   number of filters that are projected out
            stride:                     stride for all convolution operations
            data_format:                number of filters applied to expanded volume
            activation:                 activation function for all convolution operations
            dropout_rate:               dropout_rate for the complete block, applied to output only
            weight_decay:               weight magnitude penalty for all convolution operations
        """
        
        super(BottleneckLayers, self).__init__(name="bottleneck_layers")
        
        axis = -1 if data_format == "channels_last" else 1
        self.activation_fcn = activation
        
        config_convolutions = {'padding': "same", 'use_bias': False, 'data_format': data_format,
                               'kernel_initializer': 'he_normal', 'kernel_regularizer': l2(weight_decay)
                               }
        
        # Expansion layer
        self.expansion = tf.keras.layers.Conv2D(expanded_filters_count, kernel_size=1, **config_convolutions)
        self.batchnorm_expansion = tf.keras.layers.BatchNormalization(axis=axis)
        
        # Depthwise Convolution layer [Spatial Combinations per channel]
        config_depthwise = config_convolutions.copy()
        config_depthwise['depthwise_regularizer'] = config_depthwise.pop('kernel_regularizer')
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, **config_depthwise)
        self.batchnorm_depthwise = tf.keras.layers.BatchNormalization(axis=axis)
        
        # Projection layer
        self.projection = tf.keras.layers.Conv2D(projection_filters_count, kernel_size=1, **config_convolutions)
        self.batchnorm_projection = tf.keras.layers.BatchNormalization(axis=axis)
    
    def call(self, x, training=True, mask=None):
        x = self.expansion(x)
        x = self.batchnorm_expansion(x, training=training)
        x = self.activation_fcn(x)
        
        x = self.depthwise_conv(x)
        x = self.batchnorm_depthwise(x, training=training)
        x = self.activation_fcn(x)
        
        x = self.projection(x)
        x = self.batchnorm_projection(x, training=training)
        return x


class InvertedResidualBlocks(tf.keras.Model):
    """ Inverted Residual Blocks consisting of BottleNeck Layers """

    def get_config(self):
        return getattr(self, 'params', dict())

    @staticmethod
    def expand(filter_count: int, expansion_coefficient: float = 1.0):
        filter_count = max(int(round(filter_count * expansion_coefficient)), filter_count)
        return filter_count

    def __init__(self,
                 input_channels,
                 output_channels,
                 expansion_coefficient,
                 cardinality,
                 stride=1.0,
                 activation=tf.nn.relu6,
                 data_format='channels_last'):
        """
        Args:
            input_channels:         count of input channels processed by this block.
            output_channels:        count of channels emitted by the block
            expansion_coefficient:  expansion constant (expansion of input volume by the point-wise filters)
            cardinality:            number of layers in this block
            stride:                 stride for all layers in the underlying bottleneck module
            data_format:            data-format == channels_first or channels_last
        """
        
        super(InvertedResidualBlocks, self).__init__(name="Inverted_Residual_Block")
        stride = stride
        self.bottleneck_layers = []

        channels_count = input_channels
        for _ in range(cardinality):
            expanded_filter_count = self.expand(channels_count, expansion_coefficient)
            kwargs_group = {'stride': stride, 'activation': activation, 'data_format': data_format}
            layer_group = BottleneckLayers(expanded_filter_count, output_channels, **kwargs_group)
            self.bottleneck_layers.append(layer_group)
            is_symmetric = (stride == 1) and (input_channels == output_channels)
            self.Add = tf.keras.layers.Add() if is_symmetric else None
            channels_count = output_channels
            stride = 1
        
    def call(self, x, training=True, mask=None):
        for layer_group in self.bottleneck_layers:
            xk = layer_group(x, training=training)
            x = self.Add([x, xk]) if self.Add is not None else xk
        return x


class Mobilenetv2(tf.keras.Model):
    """ Inverted Residual Blocks consisting of BottleNeck Layers.
      Arguments:
      stem_filter_count:        Number of filters in initial convolution layer.
      block_params:             list of quadruples (t, c, n, s)
                                    t: expansion factor
                                    c: filters
                                    n: cardinality/repetition
                                    s: stride
                                    For more information on these parameters [https://arxiv.org/pdf/1801.04381.pdf]
      alpha:                    width multiplier
      top_filters:
      data_format:              "channels_first" or "channels_last"
      """
    def get_config(self):
        return getattr(self, 'params', dict())
    
    @staticmethod
    def scale(filter_count: int, scale_coefficient: float = 1.0, hard_minimum: int = 8):
        filter_count = max(int(round(filter_count * scale_coefficient)), hard_minimum)
        return filter_count
    
    @capture_params
    def __init__(self,
                 filters_stem=32,
                 blocks_configuration=((1, 16, 1, 1),
                                       (6, 24, 2, 1),
                                       (6, 32, 3, 2),
                                       (6, 64, 4, 2),
                                       (6, 96, 3, 1),
                                       (6, 160, 3, 2),
                                       (6, 320, 1, 1),
                                       ),
                 alpha=1.0,
                 filters_top=1280,
                 output_classes=10,
                 include_top=True,
                 dropout_rate=0.0,
                 data_format="channels_last",
                 ):
        """
        Args:
            filters_stem:  Number of filters in the first convolution layer.
            blocks_configuration:       list of quadruples (t, c, n, s)
                                        t: expansion factor
                                        c: filters
                                        n: cardinality/repetition
                                        s: stride
                                        For more information on these parameters [https://arxiv.org/pdf/1801.04381.pdf]
            alpha:                      width/depth compression multiplier [0, 1)
            filters_top:                number of filters in final convolution layer
            output_classes:             number of neurons in the final layer
            data_format:                "channels_first" or "channels_last"
            weight_decay:               The L2 normalization constant for each convolution layer
        """
        
        super(Mobilenetv2, self).__init__(name="mobilenet_v2")
        
        axis = -1 if data_format == 'channels_last' else 1
        config_convolutions = {'padding': "same", 'use_bias': False, 'data_format': data_format,
                               'kernel_initializer': 'he_normal', 'kernel_regularizer': l2(1e-4)
                               }
        filters_stem = self.scale(filters_stem, alpha)

        self.activation_fcn = tf.nn.relu6
        self.include_top = include_top
        self.convolution_stem = tf.keras.layers.Conv2D(filters_stem, kernel_size=3, strides=2, **config_convolutions)
        self.batchnorm_stem = tf.keras.layers.BatchNormalization(axis=axis)
        
        channels_count = filters_stem
        self.inverted_residual_blocks = []
        for expansion, filters, cardinality, stride in blocks_configuration:
            filters_count = self.scale(filters, alpha)
            kwargs_block = {'stride': stride, 'activation': self.activation_fcn, 'data_format': data_format}
            block = InvertedResidualBlocks(channels_count, filters_count, expansion, cardinality, **kwargs_block)
            self.inverted_residual_blocks.append(block)
            channels_count = filters_count
        
        # Final 1x1 convolution layer
        # The original paper does not scale down the filters for the final convolution
        filters_top = self.scale(filters_top, alpha) if alpha > 1.0 else filters_top
        self.convolution_top = tf.keras.layers.Conv2D(filters_top, kernel_size=1, strides=1, **config_convolutions)
        self.batchnorm_top = tf.keras.layers.BatchNormalization(axis=axis)
        
        if self.include_top:
            self.global_pooling_top = tf.keras.layers.GlobalAveragePooling2D()
            self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0.0 else None
            self.classification = tf.keras.layers.Dense(output_classes, kernel_regularizer=l2(1e-4))
    
    def call(self, x, training=True, mask=None):
        # stem operations
        x = self.convolution_stem(x)
        x = self.batchnorm_stem(x, training=training)
        x = self.activation_fcn(x)
        # inverted residual blocks
        for block in self.inverted_residual_blocks:
            x = block(x, training=training)
        # top convolution
        x = self.convolution_top(x)
        x = self.batchnorm_top(x, training=training)
        x = self.activation_fcn(x)
        # classifier
        if self.include_top:
            x = self.global_pooling_top(x)
            x = self.dropout(x, training=training) if self.dropout is not None else x
            x = self.classification(x)
        return x
