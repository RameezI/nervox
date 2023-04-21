"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import os
import uuid
from pathlib import Path
import tensorflow as tf
import tempfile
from nervox import Trainer
from nervox.data import DataStream
from nervox.protocols import Classification
from tensorflow.python.distribute import combinations
from nervox.modules.vision_decoders import GlobalAvgPoolDecoder

MODELS = ['convnet']


@combinations.generate(combinations.combine(model=MODELS))
class TestExporter(tf.test.TestCase):
    def setUp(self):
        image = tf.expand_dims(tf.random.normal((32, 32, 3)), axis=0)
        label = tf.expand_dims(tf.one_hot(0, 10), axis=0)

        dataset_train = tf.data.Dataset.from_tensor_slices({'image': image,
                                                            'label': label})
        self.dummy_train_stream = DataStream(dataset_train, batch_size=1)
    
    def create_checkpoint(self, model, logs_dir, run_id):
        trainer = Trainer(self.dummy_train_stream, logs_dir=logs_dir, run_id=run_id)
        trainer.push_module(model, alias='encoder')
        trainer.push_module(GlobalAvgPoolDecoder, alias='decoder',
                           config={'output_units': 10})
        strategy = Classification(supervision_keys=('image', 'label'))
        trainer.spin(strategy, max_epochs=1)
        return trainer.checkpoint_dir
    
    def resume_from_checkpoint(self, model, logs_dir, run_id):
        trainer = Trainer(self.dummy_train_stream, logs_dir=logs_dir, run_id=run_id)
        trainer.push_module(model, alias='encoder')
        trainer.push_module(GlobalAvgPoolDecoder, alias='decoder',
                           config={'output_units': 10})
        strategy = Classification(supervision_keys=('image', 'label'))
        trainer.spin(strategy, max_epochs=2)
        return trainer.checkpoint_dir
    
    def test_checkpoint_creation(self, model):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_id = str(uuid.uuid4())
            ckpt_dir = self.create_checkpoint(model, temp_dir, run_id=run_id)
            ckpt_file = Path(ckpt_dir, 'checkpoint')
            self.assertTrue(os.path.isfile(ckpt_file))
    
    def test_checkpoint_resume(self, model):
        run_id = str(uuid.uuid4())
        tmp_dir = tempfile.TemporaryDirectory()
        with tmp_dir as logs_dir:
            self.create_checkpoint(model, logs_dir, run_id=run_id)
            self.resume_from_checkpoint(model, logs_dir, run_id=run_id)


if __name__ == "__main__":
    tf.test.main()
