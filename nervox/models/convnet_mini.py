"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import tensorflow as tf
from nervox.utils import capture_params


class ConvnetMini(tf.keras.Model):
    """ This class creates a fully connected network.
        Arguments:
        conv_layers:            A list of k elements providing number of filters for k convolution layers
        neurons_per_layer:      A list of k elements providing number of neurons for k dense layers
        output_classes:         number of output classes.
        data_format:            "channels_first" or "channels_last"
       """
    def get_config(self):
        return getattr(self, 'params', dict())
    
    @capture_params
    def __init__(self,
                 conv_layers=(32, 32, 32),
                 output_classes=10,
                 data_format='channels_last'
                 ):
        super(ConvnetMini, self).__init__(name="convnet")
        self.convolution_layers = []
        self.pool_layers = []

        for i in range(len(conv_layers)):
            self.convolution_layers.append(tf.keras.layers.Conv2D(conv_layers[i],
                                                                  kernel_size=(3, 3),
                                                                  padding='same',
                                                                  data_format=data_format
                                                                  ))
            self.pool_layers.append(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), data_format=data_format))

        self.flat_layer = tf.keras.layers.Flatten(data_format=data_format)
        self.dropout_layer = tf.keras.layers.Dropout(0.25)
        self.classification_layer = tf.keras.layers.Dense(output_classes)

    def call(self, x, training=True, mask=None):
        for i in range(len(self.convolution_layers)):
            x = self.convolution_layers[i](x)
            x = self.pool_layers[i](tf.nn.relu(x))
        x = self.dropout_layer(x, training=training)
        x = self.flat_layer(x)
        y = self.classification_layer(x)
        return y

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], self.params['output_classes']))

    def compute_output_signature(self, input_signature):
        dtype = input_signature.dtype
        shape = self.compute_output_shape(input_signature.shape)
        return tf.TensorSpec(shape, dtype=dtype)
