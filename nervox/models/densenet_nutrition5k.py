"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import tensorflow as tf
from nervox.utils import capture_params

l2 = tf.keras.regularizers.l2


class RegressionHead(tf.keras.Model):
    def get_config(self):
        return getattr(self, 'params', dict())
    
    @capture_params
    def __init__(self, units=1, hidden_units=(256,),
                 dropout_rate=0.1, **kwargs):
        super(RegressionHead, self).__init__(**kwargs)
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.fcn_layers = [tf.keras.layers.Dense(latent)
                           for latent in hidden_units]
        self.output_layer = tf.keras.layers.Dense(units, kernel_regularizer=l2(1e-4))
    
    def call(self, x, training=True):
        x = self.dropout_layer(x, training=training)
        for layer in self.fcn_layers:
            x = tf.nn.relu(layer(x))
        x = self.output_layer(x)
        return x


class ClassificationHead(tf.keras.Model):
    def get_config(self):
        return getattr(self, 'params', dict())
    
    @capture_params
    def __init__(self, units=555, hidden_units=None, dropout_rate=0.0, **kwargs):
        super(ClassificationHead, self).__init__(**kwargs)
        self.fcn_layer = tf.keras.layers.Dense(hidden_units) \
            if hidden_units is not None else None
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.output_layer = tf.keras.layers.Dense(units, kernel_regularizer=l2(1e-4))
    
    def call(self, x, training=True):
        x = tf.nn.relu(self.fcn_layer(x)) \
            if self.fcn_layer is not None else x
        x = self.dropout_layer(x, training=training)
        x = self.output_layer(x)
        return x


class DensenetNutrition5k(tf.keras.Model):
    """Creates DenseNet121 using Keras.Application, optionally pre-trained weights can be loaded
    For more information visit: [Keras Applications Densenet121]
    (https://keras.io/api/applications/densenet/#densenet121-function)
    """
    
    def get_config(self):
        return getattr(self, 'params', dict())
    
    @capture_params
    def __init__(self,
                 weights='imagenet',
                 base_trainable=True,
                 data_format="channels_last",
                 output_units=555,
                 ):
        super(DensenetNutrition5k, self).__init__()
        assert data_format == "channels_last", \
            'The pre-trained model only supports channels_last format'
        
        # axis = -1 if data_format == "channels_last" else 1
        self.base_model = tf.keras.applications.densenet.DenseNet121(include_top=False,
                                                                     weights=weights)
        self.base_trainable = base_trainable
        
        # freeze the layers or make them trainable
        self.base_model.trainable = base_trainable
        for layer in self.base_model.layers:
            layer.trainable = self.base_trainable
        
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)
        self.classifier = ClassificationHead(output_units, name='ingredients')
        self.regression = RegressionHead(output_units, name='calories')
    
    def call(self, x, training=True):
        stem_training = training if self.base_trainable else False
        x = self.base_model(x, training=stem_training)
        x = self.global_pool(x)
        # Attach head/heads to the base model
        ingredients = self.classifier(x, training=training)
        return ingredients
