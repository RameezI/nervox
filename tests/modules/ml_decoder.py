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
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from nervox.data import DataStream
from nervox.modules.vision_decoders import MLDecoder


def convert_to_tensors(*args):
    results = tuple()
    for arg in args:
        results += (tf.convert_to_tensor(arg, dtype=tf.float32),)
    return results


class TestMlDecoder(tf.test.TestCase):
    
    def test_invocation(self):
        x = tf.random.normal((32, 14, 14, 1280))
        ml_decoder = MLDecoder(output_units=555, num_of_groups=111, embedding_dim=768,
                               n_heads=8, dim_feedforward=2048, dropout_rate_attention=0.1, num_layers=1,
                               data_format='channels_last')
        out = ml_decoder(x)
        print(out.shape)

    # def test_mha_keras(self):
    #     x = tf.random.normal((1, 196, 768))
    #     y = tf.random.normal((1, 111, 768))
    #     num_heads = 8
    #     mha_layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=768//num_heads)
    #     out = mha_layer(y, x)
    #     print(mha_layer.count_params())


if __name__ == "__main__":
    tf.test.main()
