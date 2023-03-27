"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""
import tempfile
from pathlib import Path
import tensorflow as tf
from nervox import Trainer
from nervox.data import DataStream
from nervox.protocols import Classification
from nervox.models.terminals import GlobalAvgPoolDecoder
from nervox.models import DenseNet121


class TestTrainer(tf.test.TestCase):
    
    def setUp(self):
        image = tf.expand_dims(tf.random.normal((224, 224, 3)), axis=0)
        label = tf.expand_dims(tf.one_hot(0, 101), axis=0)
        
        dataset_train = tf.data.Dataset.from_tensor_slices({'image': image,
                                                            'label': label})
        dataset_test = tf.data.Dataset.from_tensor_slices({'image': image,
                                                           'label': label})
        
        self.dummy_train_stream = DataStream(dataset_train, batch_size=1)
        self.dummy_test_stream = DataStream(dataset_test, batch_size=1)

    def test_trainer_naming_logic(self):
        # data stream for training
        train_data = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
        trainer = Trainer(train_data)
        self.assertTrue(str(trainer) == 'unnamed')
        trainer = Trainer(train_data, name='test_name')
        self.assertEqual(str(trainer), 'test_name')
    
    def test_stream_connections(self):
        
        self.assertTrue(self.dummy_train_stream)
        # trainer has access to the streams
        trainer = Trainer(self.dummy_train_stream,
                          eval_stream=self.dummy_test_stream)
        self.assertTrue(trainer._data_streams['train'])
        self.assertTrue(trainer._data_streams['evaluate'])
        
        # test train stream connection, assert identical repetition
        self.assertEqual(self.dummy_train_stream, trainer._data_streams['train'])
        self.assertEqual(self.dummy_train_stream.as_dataset(),
                         trainer._data_streams['train'].as_dataset())
        
        # # test eval stream connection, assert identical repetition
        self.assertEqual(self.dummy_test_stream, trainer._data_streams['evaluate'])
        self.assertEqual(self.dummy_test_stream.as_dataset(),
                         trainer._data_streams['evaluate'].as_dataset())
    
    def test_model_addition(self):
        dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
        stream = DataStream(dataset)
        trainer = Trainer(stream)
        trainer.push_model('convnet', alias='model')
        self.assertTrue(isinstance(trainer.models['model'], tf.keras.Model))
    
    def test_multiple_model_addition(self):
        dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
        stream = DataStream(dataset)
        trainer = Trainer(stream)
        trainer.push_model('convnet', alias='encoder')
        trainer.push_model('convnet', alias='decoder')
        self.assertTrue(isinstance(trainer.models['encoder'], tf.keras.Model))
        self.assertTrue(isinstance(trainer.models['decoder'], tf.keras.Model))
    
    def test_nameless_model_addition(self):
        dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
        stream = DataStream(dataset)
        trainer = Trainer(stream)
        trainer.push_model('convnet')
        self.assertTrue(all(key in trainer.models.keys() for key in ['model']))
        trainer.push_model('convnet')
        self.assertTrue(all(key in trainer.models.keys() for key in ['model', 'model_1']))
    
    def test_spinning_invocation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(self.dummy_train_stream, logs_dir=temp_dir)
            trainer.push_model('convnet', alias='encoder')
            trainer.push_model(GlobalAvgPoolDecoder, alias='decoder',
                               config={'output_units': 101})
            strategy = Classification(supervision_keys=('image', 'label'))
            trainer.spin(strategy, max_epochs=1)
    
    def test_transfer_learning_keras(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(self.dummy_train_stream, logs_dir=temp_dir)
            
            # push models to the trainer and test with classification strategy
            trainer.push_model(DenseNet121, alias='encoder',
                               config={'trainable': False,
                                       'weights': 'imagenet'})
            trainer.push_model(GlobalAvgPoolDecoder, alias='decoder',
                               config={'output_units': 101})
            
            # record weights when initialized with `imagenet`
            model = trainer.models['encoder']._model
            weights_pre_load = [tf.constant(var)
                                for var in model.variables]
    
            # record weight after explicit load
            model_name \
                = 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_file = Path('~', '.keras', 'models', model_name).expanduser()
            model.load_weights(weights_file)
            weights_post_load = [tf.constant(var)
                                 for var in model.variables]
            
            # Assert the two weight records are the same
            for a, b in zip(weights_pre_load, weights_post_load):
                self.assertAllClose(a, b)
            
            # run for a single epoch
            strategy = Classification(supervision_keys=('image', 'label'))
            trainer.spin(strategy, max_epochs=1)
            weights_post_spin = [tf.constant(var)
                                 for var in model.variables]
            
            # assert that base model is not trained during the epoch
            for a, b in zip(weights_pre_load, weights_post_spin):
                self.assertAllClose(a, b)

    def test_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(self.dummy_train_stream, logs_dir=temp_dir)

       
if __name__ == "__main__":
    tf.test.main()
