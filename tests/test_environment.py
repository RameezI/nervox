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
