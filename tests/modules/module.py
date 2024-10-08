import numpy as np
import tensorflow as tf
import nervox as nx


class SimpleDense(nx.Module):

    def __init__(self, units=32):
        # super().__init__()
        self.units = units

    def build(self, input_shape):
        kernel = tf.random.normal(shape=(input_shape[-1], self.units), dtype=self.dtype)

        bias = tf.zeros(shape=(self.units,), dtype=self.dtype)

        self.kernel = tf.Variable(initial_value=kernel, trainable=True)
        self.bias = tf.Variable(initial_value=bias, trainable=True)

    def compute(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias


class TestSimpleDenseModule(tf.test.TestCase):

    def test_SimpleDense(self):
        dense = SimpleDense(4)

        # test the shape of output
        x = tf.ones((2, 2))
        y = dense(x)
        assert y.shape == (2, 4)

        # test the number of state variables
        assert len(dense.state) == 2

        # test trainable variables
        assert len(dense.trainable_variables) == 2

        # test non-trainable variables
        assert len(dense.non_trainable_variables) == 0

        # test all variables
        assert len(dense.variables) == 2

        # test the dtype of the state variables
        assert dense.kernel.dtype == tf.float32
        assert dense.bias.dtype == tf.float32

        # test the dtype of the module
        assert dense.dtype == tf.float32

        # test get_config method
        config = dense.params["__init__"]
        assert isinstance(config, dict)
        assert config == {"units": 4}

        # test from_config method
        dense2 = SimpleDense(**config)
        assert dense2.units == dense.units

        # test compute method
        assert np.allclose(dense(x), dense.compute(x))


if __name__ == "__main__":
    tf.test.main()
