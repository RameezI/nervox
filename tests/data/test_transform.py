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
