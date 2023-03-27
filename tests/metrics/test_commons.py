import tensorflow as tf
from nervox.metrics import Mean


class TestMeanMetrics(tf.test.TestCase):
    def test_mean_metric(self):
        values = tf.constant([[2., 8., 8., 2.]])
        mean = Mean()
        expectation = 5.0
        mean.update(values)
        self.assertAllEqual(expectation, mean.result())
        
        expectation = 10.0
        values = tf.constant([[11., 17.4, 22.6, 9.]])
        mean.update(values)
        self.assertAllEqual(expectation, mean.result())
        
        mean.reset()
        expectation = 60 / 4.
        values = tf.constant([[11., 17.4, 22.6, 9.]])
        mean.update(values)
        self.assertAllClose(expectation, mean.result())
    
    def test_mean_metric_multi_axis(self):
        values = tf.constant([[2., 8, 8., 2, ], [2., 8., 8., 2.]])
        mean = Mean(axis=-1)
        expectation = [5.0, 5.0]
        mean.update(values)
        self.assertAllEqual(expectation, mean.result())
        
        expectation = [5.0, 10.0]
        values = tf.constant([[2., 8, 8., 2, ], [11., 17.4, 22.6, 9.]])
        mean.update(values)
        self.assertAllEqual(expectation, mean.result())
        
        mean.reset()
        expectation = [60. / 4, 60. / 4]
        values = tf.constant([[11., 17.4, 22.6, 9.], [11., 17.4, 22.6, 9.]])
        mean.update(values)
        self.assertAllClose(expectation, mean.result())
        
        mean = Mean(axis=1)
        expectation = [[10., 20.], [10, 20], [5, 5], [10., 15.]]
        values = tf.constant([[[11., 17.4], [9., 22.6]], [[9, 22.6], [11., 17.4]],
                              [[2, 8.], [8., 2.]], [[15, 22], [5., 8.]]
                              ])
        mean.update(values)
        self.assertAllClose(expectation, mean.result())
        
        mean = Mean(axis=1, keep_dims=True)
        expectation = [[[10., 20.]], [[10, 20]], [[5, 5]], [[10., 15.]]]
        values = tf.constant([[[11., 17.4], [9., 22.6]], [[9, 22.6], [11., 17.4]],
                              [[2, 8.], [8., 2.]], [[15, 22], [5., 8.]]
                              ])
        mean.update(values)
        self.assertAllClose(expectation, mean.result())


if __name__ == '__main__':
    tf.test.main()
