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

import tensorflow as tf
from nervox.utils import capture_params
from typing import Union, Callable

""" This module contains a set of pre-trained encoders for image feature extraction.
These encoders are based on the Keras implementation of the models. They are wrapped
as nervox `Module` object and can be used as any other module if the  nervox 
framework. The encoders are trainable by default, ..


Example:
    >>> from nervox.modules import DenseNet121

"""

__all__ = [
    "DenseNet121Encoder",
    "DenseNet169Encoder",
    "DenseNet201Encoder",
    "MobileNetV1Encoder",
    "MobileNetV2Encoder",
    "XceptionEncoder",
    "InceptionV3Encoder",
    "EfficientNetB0Encoder",
    "EfficientNetB1Encoder",
    "EfficientNetB2Encoder",
    "EfficientNetB3Encoder",
    "EfficientNetB4Encoder",
]


class KerasEncoder(tf.keras.Model):

    def get_config(self):
        return getattr(self, "params", dict())

    def __init__(
        self,
        pre_processor: Union[Callable, None],
        model: tf.keras.Model,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.preprocessor = pre_processor
        self._model = model

        self._model.trainable = trainable
        for layer in self._model.layers:
            layer.trainable = trainable

    def call(self, x, training=True):
        x = self.preprocessor(tf.cast(x, tf.float32)) if self.preprocessor else x
        training = training if self._model.trainable else False
        x = self._model(x, training=training)
        return x


class DenseNet121Encoder(KerasEncoder):
    @capture_params
    def __init__(
        self,
        weights: Union[str, None] = "imagenet",
        trainable: bool = True,
        pre_process: bool = True,
    ):
        preprocessor = (
            tf.keras.applications.densenet.preprocess_input if pre_process else None
        )
        model = tf.keras.applications.densenet.DenseNet121(
            include_top=False, weights=weights
        )
        super().__init__(preprocessor, model, trainable=trainable)


class DenseNet169Encoder(KerasEncoder):
    @capture_params
    def __init__(
        self,
        weights: Union[str, None] = "imagenet",
        trainable: bool = True,
        pre_process: bool = True,
    ):
        preprocessor = (
            tf.keras.applications.densenet.preprocess_input if pre_process else None
        )
        model = tf.keras.applications.densenet.DenseNet169(
            include_top=False, weights=weights
        )
        super().__init__(preprocessor, model, trainable=trainable)


class DenseNet201Encoder(KerasEncoder):
    @capture_params
    def __init__(
        self,
        weights: Union[str, None] = "imagenet",
        trainable: bool = True,
        pre_process: bool = True,
    ):
        preprocessor = (
            tf.keras.applications.densenet.preprocess_input if pre_process else None
        )
        model = tf.keras.applications.densenet.DenseNet201(
            include_top=False, weights=weights
        )
        super().__init__(preprocessor, model, trainable=trainable)


class MobileNetV1Encoder(KerasEncoder):

    @capture_params
    def __init__(
        self,
        weights: Union[str, None] = "imagenet",
        trainable: bool = True,
        pre_process: bool = True,
    ):

        preprocessor = (
            tf.keras.applications.mobilenet.preprocess_input if pre_process else None
        )

        model = tf.keras.applications.MobileNet(include_top=False, weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class MobileNetV2Encoder(KerasEncoder):

    @capture_params
    def __init__(
        self,
        weights: Union[str, None] = "imagenet",
        trainable: bool = True,
        pre_process: bool = True,
    ):
        preprocessor = (
            tf.keras.applications.mobilenet.preprocess_input if pre_process else None
        )

        model = tf.keras.applications.MobileNetV2(include_top=False, weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class XceptionEncoder(KerasEncoder):

    @capture_params
    def __init__(
        self,
        weights: Union[str, None] = "imagenet",
        trainable: bool = True,
        pre_process: bool = True,
    ):
        preprocessor = (
            tf.keras.applications.xception.preprocess_input if pre_process else None
        )

        model = tf.keras.applications.Xception(include_top=False, weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class InceptionV3Encoder(KerasEncoder):

    @capture_params
    def __init__(
        self,
        weights: Union[str, None] = "imagenet",
        trainable: bool = True,
        pre_process: bool = True,
    ):
        preprocessor = (
            tf.keras.applications.inception_v3.preprocess_input if pre_process else None
        )

        model = tf.keras.applications.InceptionV3(include_top=False, weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB0Encoder(KerasEncoder):

    @capture_params
    def __init__(
        self,
        weights: Union[str, None] = "imagenet",
        trainable: bool = True,
        pre_process: bool = True,
    ):
        preprocessor = (
            tf.keras.applications.efficientnet.preprocess_input if pre_process else None
        )

        model = tf.keras.applications.EfficientNetB0(include_top=False, weights=weights)

        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB1Encoder(KerasEncoder):

    @capture_params
    def __init__(
        self,
        weights: Union[str, None] = "imagenet",
        trainable: bool = True,
        pre_process: bool = True,
    ):
        preprocessor = (
            tf.keras.applications.efficientnet.preprocess_input if pre_process else None
        )

        model = tf.keras.applications.EfficientNetB1(include_top=False, weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB2Encoder(KerasEncoder):
    @capture_params
    def __init__(
        self,
        weights: Union[str, None] = "imagenet",
        trainable: bool = True,
        pre_process: bool = True,
    ):
        preprocessor = (
            tf.keras.applications.efficientnet.preprocess_input if pre_process else None
        )

        model = tf.keras.applications.EfficientNetB2(include_top=False, weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB3Encoder(KerasEncoder):
    @capture_params
    def __init__(
        self,
        weights: Union[str, None] = "imagenet",
        trainable: bool = True,
        pre_process: bool = True,
    ):
        preprocessor = (
            tf.keras.applications.efficientnet.preprocess_input if pre_process else None
        )

        model = tf.keras.applications.EfficientNetB3(include_top=False, weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB4Encoder(KerasEncoder):

    @capture_params
    def __init__(
        self,
        weights: Union[str, None] = "imagenet",
        trainable: bool = True,
        pre_process: bool = True,
    ):
        preprocessor = (
            tf.keras.applications.efficientnet.preprocess_input if pre_process else None
        )

        model = tf.keras.applications.EfficientNetB4(include_top=False, weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)
