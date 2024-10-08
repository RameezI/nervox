import tensorflow as tf
import nervox as nx


class SimpleDense(nx.Module):

    def __init__(self, units=32):
        self.units_simple = units

    # def build(self, input_shape):
    #     kernel = tf.random.normal(
    #         shape=(input_shape[-1], self.units),
    #         dtype=self.dtype
    #     )

    #     bias = tf.zeros(shape=(self.units,), dtype=self.dtype)

    #     self.kernel = tf.Variable(initial_value=kernel, trainable=True)
    #     self.bias = tf.Variable(initial_value=bias, trainable=True)

    # def compute(self, inputs):
    #     return tf.matmul(inputs, self.kernel) + self.bias


class ComplexDense(SimpleDense):
    def __init__(self, units_simple=33, units_complex=64):
        super().__init__(units_simple)
        self.units_complex = units_complex


module = ComplexDense(24, 64)
print(module.params)
print(module.units_simple, module.units_complex)
print(module.name)
print(module.dtype)
