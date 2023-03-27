"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

# Set debug level for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Environment(tf.test.TestCase):
    def setUp(self):
        super(Environment, self).setUp()
    
    def test_gpu_availability(self):
        list_devices = tf.config.list_physical_devices('GPU')
        print(list_devices)
        self.assertTrue(len(list_devices) > 0, msg='gpu is not available')
        self.assertTrue(tf.test.is_built_with_cuda(), msg='The current tensorflow is not built with cuda support')


if __name__ == "__main__":
    tf.test.main()
