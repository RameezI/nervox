import tensorflow as tf
from nervox.metrics import Mean


class TestMeanMetrics(tf.test.TestCase):
    def test_mean_metric(self):
        values = tf.constant([[2.0, 8.0, 8.0, 2.0]])
        mean = Mean()
        expectation = 5.0
        mean.update(values)
        self.assertAllEqual(expectation, mean.result())

        expectation = 10.0
        values = tf.constant([[11.0, 17.4, 22.6, 9.0]])
        mean.update(values)
        self.assertAllEqual(expectation, mean.result())

        mean.reset()
        expectation = 60 / 4.0
        values = tf.constant([[11.0, 17.4, 22.6, 9.0]])
        mean.update(values)
        self.assertAllClose(expectation, mean.result())

    def test_mean_metric_multi_axis(self):
        values = tf.constant(
            [
                [
                    2.0,
                    8,
                    8.0,
                    2,
                ],
                [2.0, 8.0, 8.0, 2.0],
            ]
        )
        mean = Mean(axis=-1)
        expectation = [5.0, 5.0]
        mean.update(values)
        self.assertAllEqual(expectation, mean.result())

        expectation = [5.0, 10.0]
        values = tf.constant(
            [
                [
                    2.0,
                    8,
                    8.0,
                    2,
                ],
                [11.0, 17.4, 22.6, 9.0],
            ]
        )
        mean.update(values)
        self.assertAllEqual(expectation, mean.result())

        mean.reset()
        expectation = [60.0 / 4, 60.0 / 4]
        values = tf.constant([[11.0, 17.4, 22.6, 9.0], [11.0, 17.4, 22.6, 9.0]])
        mean.update(values)
        self.assertAllClose(expectation, mean.result())

        mean = Mean(axis=1)
        expectation = [[10.0, 20.0], [10, 20], [5, 5], [10.0, 15.0]]
        values = tf.constant(
            [
                [[11.0, 17.4], [9.0, 22.6]],
                [[9, 22.6], [11.0, 17.4]],
                [[2, 8.0], [8.0, 2.0]],
                [[15, 22], [5.0, 8.0]],
            ]
        )
        mean.update(values)
        self.assertAllClose(expectation, mean.result())

        mean = Mean(axis=1, keep_dims=True)
        expectation = [[[10.0, 20.0]], [[10, 20]], [[5, 5]], [[10.0, 15.0]]]
        values = tf.constant(
            [
                [[11.0, 17.4], [9.0, 22.6]],
                [[9, 22.6], [11.0, 17.4]],
                [[2, 8.0], [8.0, 2.0]],
                [[15, 22], [5.0, 8.0]],
            ]
        )
        mean.update(values)
        self.assertAllClose(expectation, mean.result())


if __name__ == "__main__":
    tf.test.main()
