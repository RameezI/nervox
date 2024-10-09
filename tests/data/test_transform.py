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

import unittest
from nervox.utils import capture_params
from nervox.data import Transform


class TestTransform(unittest.TestCase):

    class MockTransform(Transform):
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

        def transform(self, batch):
            pass

        def __call__(self, dataset):
            pass

    def test_mock_transform(self):
        mt = self.MockTransform(1, 2)
        self.assertEqual(mt.param1, 1)
        self.assertEqual(mt.param2, 2)

    def test_capture_params_explicit(self):
        class MockTransformWithCapture(Transform):
            @capture_params
            def __init__(self, param1, param2):
                self.param1 = param1
                self.param2 = param2

            def transform(self, batch):
                pass

            def __call__(self, dataset):
                pass

        mt = MockTransformWithCapture(1, 2)
        self.assertEqual(mt.params["__init__"], {"param1": 1, "param2": 2})
        self.assertEqual(mt.param1, 1)
        self.assertEqual(mt.param2, 2)

    def test_capture_params_implicit(self):
        class MockTransformWithCapture(Transform):
            def __init__(self, param1, param2):
                self.param1 = param1
                self.param2 = param2

            def transform(self, batch):
                pass

            def __call__(self, dataset):
                pass

        mt = MockTransformWithCapture(1, 2)
        self.assertEqual(mt.params["__init__"], {"param1": 1, "param2": 2})
        self.assertEqual(mt.param1, 1)
        self.assertEqual(mt.param2, 2)

    def test_illegal_decoration_syntax(self):
        with self.assertRaises(TypeError):

            @capture_params
            class MockTransform(Transform):
                def __init__(self, param1, param2):
                    self.param1 = param1
                    self.param2 = param2

                def transform(self, batch):
                    pass

                def __call__(self, dataset):
                    pass

            mt = MockTransform(1, 2)


if __name__ == "__main__":
    unittest.main()
