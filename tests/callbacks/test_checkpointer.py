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
from nervox.utils import VerbosityLevel
from nervox.protocols import Classification
from typing import Tuple, Dict


MODELS = ['convnet']
OPTIMIZERS = [tf.keras.optimizers.SGD,
              ]

DictModles= Dict[str, tf.keras.Model]
Optimizer = tf.keras.optimizers.Optimizer


class TestProtocol(Classification):
    @staticmethod
    def objective_configurator(optimizer: Optimizer):
        return (Optimizer)
        
        


@combinations.generate(combinations.combine(model=MODELS))
class TestCheckPointer(tf.test.TestCase):

    def setUp(self):
        image = tf.expand_dims(tf.random.normal((32, 32, 3)), axis=0)
        label = tf.expand_dims(tf.one_hot(0, 10), axis=0)
        
        dataset_train = tf.data.Dataset.from_tensor_slices({'image': image,
                                                            'label': label})
        self.dummy_train_stream = DataStream(dataset_train, batch_size=1)
    
    def create_trainer(self, model_type, logs_dir, run_id)\
                       ->Tuple[os.PathLike, Tuple[int, DictModles, Optimizer]]:
        """
        Creates a trainer instance and spin it for a single epoch.
        """
        trainer = Trainer(self.dummy_train_stream,
                          logs_dir=logs_dir,
                          run_id=run_id)
        
        trainer.push_module(model_type, alias='encoder')
        trainer.push_module(GlobalAvgPoolDecoder, alias='decoder',
                           config={'output_units': 10})
        
        strategy = Classification(supervision_keys=('image', 'label'))
        trainer.spin(strategy, max_epochs=1, verbose=VerbosityLevel.KEEP_SILENT,
                     skip_initial_evaluation=True)
        
        epoch = trainer.epoch
        models = trainer.models
        optimizer = trainer._protocol.objective.optimizer
        return trainer.checkpoint_dir, (epoch, models, optimizer)
    
    def test_checkpoint_creation(self, model):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_id = str(uuid.uuid4())
            ckpt_dir, _ = self.create_trainer(model, temp_dir, run_id=run_id)
            ckpt_file = Path(ckpt_dir, 'checkpoint')
            self.assertTrue(os.path.isfile(ckpt_file))

    def test_models_uniqueness(self, model):
        """
        Tests that the created models, when the trainer has a unique `run_id`
        are all always unique.
        """

        # model-A
        with tempfile.TemporaryDirectory() as temp_dir:
            _, (_, models_A, _) = self.create_trainer(model, temp_dir,
                                                     run_id=str(uuid.uuid4()))

        # model-B
        with tempfile.TemporaryDirectory() as temp_dir:
            _, (_, models_B, _) = self.create_trainer(model, temp_dir,
                                                     run_id=str(uuid.uuid4()))

        for modelB_key, modelA_key in zip(models_B, models_A):
            self.assertEqual(modelB_key, modelA_key)
            variables = [var.value() for var in models_B[modelB_key].variables]
            variables_ref = [var.value() for var in models_A[modelB_key].variables]
            self.assertEqual(len(variables), len(variables_ref))
            self.assertNotAllClose(variables[0], variables_ref[0])

    
    def test_epoch_returnability(self, model):
        """
        Tests that under a single `run_id` the epoch is restored from the previous run, when
         `warm_restart` is set to `True`, a default configuration of the trainer. 
        """
        run_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as temp_dir:
            _, (saved_epoch, *_) = self.create_trainer(model, temp_dir, run_id=run_id)
            _, (epoch, *_) = self.create_trainer(model, temp_dir, run_id=run_id)
            self.assertEqual(epoch, saved_epoch)
    
    def test_model_returnability(self, model):
        run_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as temp_dir:
            _, (_, saved_models, _) = self.create_trainer(model, temp_dir, run_id=run_id)
            _, (_, models, _) = self.create_trainer(model, temp_dir, run_id=run_id)

        for model_key, saved_model_key in zip(models, saved_models):
            self.assertEqual(model_key, saved_model_key)
            variables = [var.value() for var in models[model_key].variables]
            variables_ref = [var.value() for var in saved_models[model_key].variables]
            self._assertAllCloseRecursive(variables_ref, variables)

    def test_optimizer_returnability(self, model):
        run_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as temp_dir:
            _, (_, saved_models, saved_optimizer) = self.create_trainer(model, temp_dir, run_id=run_id)
            _, (_, models, optimizer) = self.create_trainer(model, temp_dir, run_id=run_id)

            # Test that all attributes are restored
            trackables = [child for child in saved_optimizer._trackable_children()
                           if getattr(saved_optimizer, child, None) is not None]

            variables_ref = [getattr(saved_optimizer, attribute)
                              for attribute in trackables]

            variables = [getattr(optimizer,attribute)
                          for attribute in trackables]

            self._assertAllCloseRecursive(variables_ref, variables)

            # Test that all slot variables are restored
            for slot in saved_optimizer.get_slot_names():
                slot_variables_ref = [saved_optimizer.get_slot(var, slot)
                                       for _, m in saved_models.items()
                                       for var in m.trainable_variables]

                slot_variables = [optimizer.get_slot(var, slot)
                                   for _, m in models.items()
                                   for var in m.trainable_variables]
                self._assertAllCloseRecursive(slot_variables_ref, slot_variables)

            # Test an exceptional variable (#TODO: investigate why!)
            # iterations attribute is serialized/restored as iter trackable
            self.assertEqual(optimizer.iterations,
                             saved_optimizer.iterations) 


if __name__ == "__main__":
    tf.test.main()