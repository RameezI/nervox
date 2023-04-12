"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""
import json

import tensorflow as tf
from nervox.utils import capture_params


class Convnet(tf.keras.Model):
    """ This class creates a convolution network which should achieve ~85% accuracy on CIFAR-10 dataset when learning
        the default classification objective in the nervox. This model is mainly used to test and verify the
        health of the nervox. This script also serves as a canonical form of defining a neural network model
        within the nervox.
        
        The training configuration leading to 85% test accuracy :
             'epochs': 50,
             'batch_size': 64,
             'strategy': 'explicit_supervision',
             'learning_rates': {0: 1e-2, 30: 1e-3},
            
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
                 conv_layers=(64, 128, 128, 128),
                 data_format='channels_last',
                 dropout_rate=0.25
                 ):
        super().__init__()
        axis = -1 if data_format == 'channels_last' else 1
        self.convolution_layers = list()
        self.pooling_layers = list()
        self.dense_layers = list()
        self.dropout_layers = list()
        self.batch_norm_layers = list()
        
        assert len(conv_layers) >= 2
        
        self.convolution_layers.append(tf.keras.layers.Conv2D(conv_layers[0],
                                                              kernel_size=(3, 3),
                                                              padding='same',
                                                              data_format=data_format
                                                              ))
        
        self.convolution_layers.append(tf.keras.layers.Conv2D(conv_layers[1],
                                                              kernel_size=(3, 3),
                                                              padding='same',
                                                              data_format=data_format
                                                              ))
        
        self.pooling_layers.append(tf.keras.layers.Conv2D(conv_layers[1],
                                                          kernel_size=(2, 2),
                                                          strides=(2, 2),
                                                          data_format=data_format
                                                          ))
        
        self.dropout_layers.append(tf.keras.layers.Dropout(dropout_rate))
        
        for i in range(2, len(conv_layers)):
            self.convolution_layers.append(tf.keras.layers.Conv2D(conv_layers[i],
                                                                  kernel_size=(3, 3),
                                                                  padding='same',
                                                                  data_format=data_format,
                                                                  use_bias=False,
                                                                  ))
            
            self.batch_norm_layers.append(tf.keras.layers.BatchNormalization(axis=axis))
            
            self.pooling_layers.append(tf.keras.layers.Conv2D(conv_layers[i],
                                                              kernel_size=(2, 2),
                                                              strides=(2, 2),
                                                              data_format=data_format
                                                              ))
            self.dropout_layers.append(tf.keras.layers.Dropout(dropout_rate))
    
    def call(self, x, training=True):
        x = self.convolution_layers[0](x)
        x = self.convolution_layers[1](tf.nn.relu(x))
        x = self.pooling_layers[0](tf.nn.relu(x))
        x = self.dropout_layers[0](x, training=training)
        # zip the remaining layers
        zipped_layers = zip(self.convolution_layers[2:],
                            self.batch_norm_layers,
                            self.pooling_layers[1:],
                            self.dropout_layers[1:]
                            )
        # Apply convolution layer tuples
        for convolution, batch_norm, pooling, dropout in zipped_layers:
            x = convolution(x)
            x = batch_norm(tf.nn.relu(x), training=training)
            x = pooling(x)
            x = dropout(tf.nn.relu(x), training=training)
        return x