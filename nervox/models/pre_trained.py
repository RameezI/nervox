import tensorflow as tf
from nervox.utils import capture_params
from typing import Union, Callable


class FeatureExtractorKeras(tf.keras.Model):
    
    def get_config(self):
        return getattr(self, 'params', dict())
    
    def __init__(self, pre_processor: Union[Callable, None], model: tf.keras.Model,
                 trainable: bool = True) -> None:
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


class DenseNet121(FeatureExtractorKeras):
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.densenet.preprocess_input \
            if pre_process else None
        model = tf.keras.applications.densenet.DenseNet121(include_top=False,
                                                           weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class DenseNet169(FeatureExtractorKeras):
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.densenet.preprocess_input \
            if pre_process else None
        model = tf.keras.applications.densenet.DenseNet169(include_top=False,
                                                           weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class DenseNet201(FeatureExtractorKeras):
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.densenet.preprocess_input \
            if pre_process else None
        model = tf.keras.applications.densenet.DenseNet201(include_top=False,
                                                           weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class MobileNetV1(FeatureExtractorKeras):

    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):

        preprocessor = tf.keras.applications.mobilenet.preprocess_input \
            if pre_process else None

        model = tf.keras.applications.MobileNet(include_top=False,
                                                weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class MobileNetV2(FeatureExtractorKeras):
    
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.mobilenet.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.MobileNetV2(include_top=False,
                                                  weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class Xception(FeatureExtractorKeras):
    
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.xception.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.Xception(include_top=False,
                                               weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class InceptionV3(FeatureExtractorKeras):
    
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.inception_v3.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.InceptionV3(include_top=False,
                                                  weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB0(FeatureExtractorKeras):
    
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.efficientnet.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.EfficientNetB0(include_top=False,
                                                     weights=weights)
        
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB1(FeatureExtractorKeras):
    
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.efficientnet.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.EfficientNetB1(include_top=False,
                                                     weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB2(FeatureExtractorKeras):
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.efficientnet.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.EfficientNetB2(include_top=False,
                                                     weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB3(FeatureExtractorKeras):
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.efficientnet.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.EfficientNetB3(include_top=False,
                                                     weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)


class EfficientNetB4(FeatureExtractorKeras):
    
    @capture_params
    def __init__(self, weights: Union[str, None] = 'imagenet',
                 trainable: bool = True,
                 pre_process: bool = True):
        preprocessor = tf.keras.applications.efficientnet.preprocess_input \
            if pre_process else None
        
        model = tf.keras.applications.EfficientNetB4(include_top=False,
                                                     weights=weights)
        super().__init__(preprocessor, model, trainable=trainable)
