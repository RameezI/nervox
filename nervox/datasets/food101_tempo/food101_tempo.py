"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

#tfds.download.add_checksums_dir(os.path.join(os.path.dirname(__file__), 'url_checksums'))

_DESCRIPTION = ''''
The dataset is composed of challenging meal images from 101 food categories with a total of 101'000 images.
The data is split into a train and test set. The train set consists of 750 images per class. For each class,
there are 250 manually reviewed test images. The image are scaled to a desired resolution.
The following variants of the dataset can be loaded:
- standard_complete: Images from 101 categories in 224x224 format.
- standard_mini: Images from only three categories, 'apple_pie', 'baby_back_ribs' and 'baklava', in 224x224 format.
'''

_URL_BASE = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'

_CITATION = '''
@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}
'''

# corrupt images in the dataset
_SKIP_IMAGES = [
    'bread_pudding/1375816',
    'lasagna/3782507',
    'lasagna/1142842',
    'lasagna/3787908',
    'steak/1340977',
]

_LABELS_FILE_NAME = os.path.join(os.path.dirname(__file__), "food-101_classes.txt")
with tf.io.gfile.GFile(_LABELS_FILE_NAME) as f_labels:
    _LABELS = [line.strip() for line in f_labels]


class Food101TempoConfig(tfds.core.BuilderConfig):
    def __init__(self, resolution=(224, 224), channels=3, labels=None, **kwargs):
        super(Food101TempoConfig, self).__init__(**kwargs)
        self.resolution = resolution
        self.channels = channels
        self.labels = labels
        
        
class Food101Tempo(tfds.core.GeneratorBasedBuilder):
    
    BUILDER_CONFIGS = [
        Food101TempoConfig(
            name='standard',
            description=(
                ''''
                The dataset is composed of challenging meal images from 101 food categories with a total of 101'000
                images. The data is split into a train and test set. The train set consists of 750 images per class.
                For each class, there are 250 manually reviewed test images.
                The image are scaled to have shape (224, 224, 3))
                '''
            ),
            version=tfds.core.Version("1.0.0"),
            resolution=(224, 224),
            labels=_LABELS
        ),
        Food101TempoConfig(
            name='mini',
            description=(
                 ''''
                The dataset is composed of challenging meal images from 3 food categories, 'apple_pie', 'baby_back_ribs'
                and 'baklava' with a total of 3'000 images. The data is split into a train and test set. The train set
                consists of 750 images per class. For each class, there are 250 manually reviewed test images.
                The image are scaled to have shape (224, 224, 3))
                '''
            ),
            version=tfds.core.Version("1.0.0"),
            resolution=(224, 224),
            labels=('apple_pie', 'baby_back_ribs', 'baklava')
        )
    ]
    
    def _info(self):
        self.input_shape = self.builder_config.resolution + (self.builder_config.channels,)
        self.num_classes = len(self.builder_config.labels)
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=self.input_shape, encoding_format='jpeg'),
                "label": tfds.features.ClassLabel(names=_LABELS[:self.num_classes]),
            }),
            supervised_keys=('image', 'label'),
            homepage="https://www.vision.ee.ethz.ch/datasets_extra/food-101",
            citation=_CITATION
        )
    
    def _split_generators(self, dl_manager):
        # dl_manager._register_checksums = True  # remove after first run
        extract_path = dl_manager.download_and_extract(_URL_BASE)
        food101_base_path = os.path.join(extract_path, 'food-101')
        return [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN,
                                     gen_kwargs={'images_dir_path': os.path.join(food101_base_path, "images"),
                                                 'labels_file': os.path.join(food101_base_path, 'meta', 'train.txt')}
                                     ),
            
            tfds.core.SplitGenerator(name=tfds.Split.TEST,
                                     gen_kwargs={'images_dir_path': os.path.join(food101_base_path, "images"),
                                                 'labels_file': os.path.join(food101_base_path, 'meta', 'test.txt')}
                                     )
        ]
    
    def _generate_examples(self, images_dir_path, labels_file):
        with tf.io.gfile.GFile(labels_file) as f:
            img_ids = [line.strip() for line in f]
            img_ids = list(filter(lambda x: x.split("/")[0] in self.builder_config.labels, img_ids))
            labels = [name.split("/")[0] for name in img_ids]
            
        for image_id, label in zip(img_ids, labels):
            image_path = os.path.join(images_dir_path, "{0:s}.jpg".format(image_id))
            
            if image_id in _SKIP_IMAGES:
                continue
                
            with tf.io.gfile.GFile(image_path, "rb") as f:
                decoded_image = tf.io.decode_image(f.read())
                resized_image = tf.image.resize(decoded_image, self.builder_config.resolution)
                resized_image = tf.cast(resized_image, tf.uint8).numpy()
            
            if resized_image.shape != self.input_shape:
                print(image_path)
    
            yield image_id, {
                "image": resized_image,
                "label": label
            }
