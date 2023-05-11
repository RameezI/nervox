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
import numpy as np
import tensorflow as tf
from nervox.metrics import compute_confusion_matrix, AveragingMode, \
    AccuracyScore, PrecisionScore, AveragePrecisionScore, RecallScore
from nervox.transforms import onehot_transform
from sklearn.metrics import average_precision_score


class TestConfusionVariables(tf.test.TestCase):
    def test_single_sample_unweighted(self):
        y_true = tf.constant([[0., 1., 0., 1.]])
        y_pred = tf.constant([[.3, .2, .6, .9]])
        
        confusion_matrix, thresholds \
            = compute_confusion_matrix(y_true, y_pred, 1)
        self.assertAllClose([0.0, 0.5, 1.0], thresholds)
        
        # ture positives
        expectations = [[0, 1, 0, 1.],  # @threshold =  0.0
                        [0, 0, 0, 1.],  # @thresholds = 0.5
                        [0, 0, 0, 0]    # @thresholds = 1.0
                        ]
        results = confusion_matrix['true_positives']
        self.assertAllEqual(expectations, results)
        
        # false positives
        expectations = [[1, 0, 1, 0.],  # @threshold = 0.0
                        [0, 0, 1, 0.],  # @threshold = 0.5
                        [0, 0, 0, 0]    # @threshold = 1.0
                        ]
        results = confusion_matrix['false_positives']
        self.assertAllEqual(expectations, results)
        
        # true negatives
        expectations = [[0, 0, 0, 0.],  # @threshold =  0.0
                        [1, 0, 0, 0.],  # @thresholds = 0.5
                        [1, 0, 1, 0]    # @thresholds = 0.5
                        ]
        results = confusion_matrix['true_negatives']
        self.assertAllEqual(expectations, results)
        
        # false negatives
        expectations = [[0, 0, 0, 0.],  # @threshold =  0.0
                        [0, 1, 0, 0.],  # @thresholds = 0.5
                        [0, 1, 0, 1]    # @thresholds = 1.0
                        ]
        results = confusion_matrix['false_negatives']
        self.assertAllEqual(expectations, results)
    
    def test_single_sample_classWeights(self):
        y_true = tf.constant([[0., 1., 0., 1.]])
        y_pred = tf.constant([[.3, .2, .6, .9]])
        weights = tf.constant([3.0, 1.0, 10.0, 2.0])
        
        confusion_matrix, thresholds \
            = compute_confusion_matrix(y_true, y_pred, bifurcators=1,
                                       weights=weights)
        self.assertAllClose([0.0, 0.5, 1.0], thresholds)
        
        # ture positives
        expectations = [[0, 1, 0, 2.],  # @threshold =  0.0
                        [0, 0, 0, 2.],  # @thresholds = 0.5
                        [0, 0, 0, 0.],  # @thresholds = 1.0
                        ]
        results = confusion_matrix['true_positives']
        self.assertAllEqual(expectations, results)
        
        # false positives
        expectations = [[3.0, 0, 10.0, 0.],  # @threshold  =  0.0
                        [0, 0, 10.0, 0.],    # @thresholds = 0.5
                        [0, 0, 0, 0.],       # @thresholds = 1.0
                        ]
        results = confusion_matrix['false_positives']
        self.assertAllEqual(expectations, results)
        
        # true negatives
        expectations = [[0., 0., 0., 0.],  # @threshold =  0.0
                        [3., 0., 0., 0.],  # @thresholds = 0.5
                        [3., 0., 10., 0]   # @thresholds = 1.0
                        ]
        results = confusion_matrix['true_negatives']
        self.assertAllEqual(expectations, results)
        
        # false negatives
        expectations = [[0, 0., 0., 0.],  # @threshold =  0.0
                        [0, 1., 0., 0.],  # @thresholds = 0.5
                        [0, 1., 0., 2.],  # @thresholds = 1.0
                        ]
        results = confusion_matrix['false_negatives']
        self.assertAllEqual(expectations, results)
    
    def test_multi_sample_unweighted(self):
        y_pred = tf.stack([tf.constant([0.1, 0.5, 0.6, 1.0]),
                           tf.constant([0.3, 0.2, 0.6, 0.9]),
                           tf.constant([0.0, 0.6, 0.3, 0.4]),
                           tf.constant([0., 1.0, 0.0, 1.0]),
                           tf.constant([1, 0., 0., 0.]),
                           ], axis=0)
        
        y_true = tf.stack([[1., 1., 1., 1.],
                           [0., 1, 0., 1.],
                           tf.one_hot(2, 4),
                           tf.one_hot(3, 4),
                           tf.one_hot(1, 4),
                           ], axis=0)
        
        confusion_matrix, thresholds \
            = compute_confusion_matrix(y_true, y_pred, bifurcators=1)
        
        # true positives
        expectations = [[1, 3, 2, 3.],  # @threshold =  0.0
                        [0, 0, 1, 3.],  # @thresholds = 0.5
                        [0, 0, 0, 0]    # @thresholds = 1.0
                        ]
        results = confusion_matrix['true_positives']
        self.assertAllEqual(expectations, results)
        
        # false positives
        expectations = [[4, 2, 3, 2.],  # @threshold =  0.0
                        [1, 2, 1, 0],   # @thresholds = 0.5
                        [0, 0, 0, 0]    # @thresholds = 1.0
                        ]
        
        results = confusion_matrix['false_positives']
        self.assertAllEqual(expectations, results)


class TestOnehotTransform(tf.test.TestCase):
    def test_transform(self):
        scores = tf.stack([tf.constant([0.7, 0.4, -9.0, 0.2]),
                           tf.constant([0.3, 4.0, -9.0, 0.7]),
                           tf.constant([0.3, 4.0, 9.0, 0.7]),
                           tf.constant([0.3, 4.0, 0.7, 9.0]),
                           tf.constant([0.3, 8.0, 6.0, 12]),
                           ], axis=0)
        
        expectation = tf.stack([tf.one_hot(0, 4),
                                tf.one_hot(1, 4),
                                tf.one_hot(2, 4),
                                tf.one_hot(3, 4),
                                tf.one_hot(3, 4),
                                ], axis=0)
        
        self.assertAllEqual(expectation, onehot_transform(scores))


class TestAccuracyScore(tf.test.TestCase):
    def setUp(self):
        self.labels = tf.stack([tf.one_hot(0, 4),
                                tf.constant([1.0, 0, 0.0, 1.0]),
                                tf.one_hot(1, 4),
                                tf.constant([1.0, 1.0, 0, 0.])
                                ], axis=0)
        self.scores = tf.stack([tf.one_hot(0, 4),
                                tf.constant([1.0, 0, 0.0, 0.]),
                                tf.one_hot(2, 4),
                                tf.constant([1.0, 1.0, 0, 0.])
                                ], axis=0)
    
    def test_samplesAverage_unweighted(self):
        labels, scores = self.labels, self.scores
        
        expectation = tf.constant(2.0 / 4.0)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.SAMPLE)
        accuracy_score.update(labels, scores)
        self.assertAllClose(expectation, accuracy_score.result())
        
        indices = [[2]]
        updates = [tf.one_hot(1, 4)]
        scores = tf.tensor_scatter_nd_update(scores, indices, updates)
        
        expectation = tf.constant(5.0 / 8.0)
        accuracy_score.update(labels, scores)
        self.assertAllClose(expectation, accuracy_score.result())
    
    def test_samplesAverage_with_sample_weights(self):
        labels, scores = (self.labels, self.scores)
        
        # self.labels = tf.stack([tf.one_hot(0, 4),
        #                         tf.constant([1.0, 0, 0.0, 1.0]),
        #                         tf.one_hot(1, 4),
        #                         tf.constant([1.0, 1.0, 0, 0.])
        #                         ], axis=0)
        # self.scores = tf.stack([tf.one_hot(0, 4),
        #                         tf.constant([1.0, 0, 0.0, 0.]),
        #                         tf.one_hot(2, 4),
        #                         tf.constant([1.0, 1.0, 0, 0.])
        #                         ], axis=0)
        
        # trivial test
        expectation = tf.constant(2.0 / 4.0)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.SAMPLE)
        accuracy_score.update(labels, scores, weights=[1, 1, 1, 1])
        self.assertAllClose(expectation, accuracy_score.result())
        accuracy_score.reset()
        
        expectation = tf.constant(1.0 / 3.0)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.SAMPLE)
        accuracy_score.update(labels, scores, weights=[0, 1, 1, 1])
        self.assertAllClose(expectation, accuracy_score.result())
        accuracy_score.reset()
        
        expectation = tf.constant(2.0 / 4.0)
        accuracy_score.update(labels, scores, weights=[0, 1, 1, 2])
        self.assertAllClose(expectation, accuracy_score.result())
        
        indices = [[2]]
        updates = [tf.one_hot(1, 4)]
        scores = tf.tensor_scatter_nd_update(scores, indices, updates)
        
        expectation = tf.constant(5.0 / 8.0)
        accuracy_score.update(labels, scores)
        self.assertAllClose(expectation, accuracy_score.result())
    
    def test_samplesAverage_with_class_weights(self):
        labels, scores = (self.labels, self.scores)
        # trivial test
        expectation = tf.constant(2.0 / 4.0)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.SAMPLE)
        accuracy_score.update(labels, scores, weights=[[1, 1, 1, 1]])
        self.assertAllClose(expectation, accuracy_score.result())
        accuracy_score.reset()
        
        labels, scores = (self.labels, self.scores)
        expectation = tf.constant(0)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.SAMPLE)
        accuracy_score.update(labels, scores, weights=[[0, 0, 1, 1]])
        self.assertAllClose(expectation, accuracy_score.result())
        accuracy_score.reset()
    
    def test_samplesAverage_subset_mode_disabled(self):
        # TODO: populate this test.
        labels, scores = (self.labels, self.scores)
        pass
    
    def test_microAverage_unweighted(self):
        labels, scores = self.labels, self.scores
        
        expectation = tf.constant(13 / 16.0)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.MICRO)
        accuracy_score.update(labels, scores)
        self.assertAllClose(expectation, accuracy_score.result())
        
        indices = [[2]]
        updates = [tf.one_hot(1, 4)]
        scores = tf.tensor_scatter_nd_update(scores, indices, updates)
        
        expectation = tf.constant(28.0 / 32.0)
        accuracy_score.update(labels, scores)
        self.assertAllClose(expectation, accuracy_score.result())
    
    def test_microAverage_with_sample_weights(self):
        # trivial test
        labels, scores = (self.labels, self.scores)
        expectation = tf.constant(13 / 16.)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.MICRO)
        accuracy_score.update(labels, scores, weights=[1, 1, 1, 1])
        self.assertAllClose(expectation, accuracy_score.result())
        accuracy_score.reset()
        
        labels, scores = (self.labels, self.scores)
        expectation = tf.constant(9 / 12.)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.MICRO)
        accuracy_score.update(labels, scores, weights=[0, 1, 1, 1])
        self.assertAllClose(expectation, accuracy_score.result())
        accuracy_score.reset()
        
        expectation = tf.constant(13.0 / 16.0)
        accuracy_score.update(labels, scores, weights=[0, 1, 1, 2])
        self.assertAllClose(expectation, accuracy_score.result())
        
        indices = [[2]]
        updates = [tf.one_hot(1, 4)]
        scores = tf.tensor_scatter_nd_update(scores, indices, updates)
        
        expectation = tf.constant(28.0 / 32.0)
        accuracy_score.update(labels, scores)
        self.assertAllClose(expectation, accuracy_score.result())
    
    def test_microAverage_with_class_weights(self):
        # trivial test
        labels, scores = (self.labels, self.scores)
        expectation = tf.constant(13.0 / 16.0)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.MICRO)
        accuracy_score.update(labels, scores, weights=[[1, 1, 1, 1]])
        self.assertAllClose(expectation, accuracy_score.result())
        accuracy_score.reset()
        
        labels, scores = (self.labels, self.scores)
        expectation = tf.constant(6 / 8.)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.MICRO)
        accuracy_score.update(labels, scores, weights=[[0, 0, 1, 1]])
        self.assertAllClose(expectation, accuracy_score.result())
        accuracy_score.reset()
        
        labels, scores = (self.labels, self.scores)
        expectation = tf.constant(7.0 / 8)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.MICRO)
        accuracy_score.update(labels, scores, weights=[[1, 1, 0, 0]])
        self.assertAllClose(expectation, accuracy_score.result())
    
    def test_macroAverage_unweighted(self):
        labels, scores = self.labels, self.scores
        expectation = tf.constant(13. / 16)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.MACRO)
        accuracy_score.update(labels, scores)
        self.assertAllClose(expectation, accuracy_score.result())
        
        indices = [[2]]
        updates = [tf.one_hot(1, 4)]
        scores = tf.tensor_scatter_nd_update(scores, indices, updates)
        
        expectation = tf.constant(28. / 32)
        accuracy_score.update(labels, scores)
        self.assertAllClose(expectation, accuracy_score.result())
    
    def test_macroAverage_with_sample_weights(self):
        # trivial test
        labels, scores = (self.labels, self.scores)
        expectation = tf.constant(13.0 / 16.0)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.MACRO)
        accuracy_score.update(labels, scores, weights=[1, 1, 1, 1])
        self.assertAllClose(expectation, accuracy_score.result())
        accuracy_score.reset()
        
        labels, scores = (self.labels, self.scores)
        expectation = tf.constant(9.0 / 12.0)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.MACRO)
        accuracy_score.update(labels, scores, weights=[0, 1, 1, 1])
        self.assertAllClose(expectation, accuracy_score.result())
        accuracy_score.reset()
        
        expectation = tf.constant(13.0 / 16.0)
        accuracy_score.update(labels, scores, weights=[0, 1, 1, 2])
        self.assertAllClose(expectation, accuracy_score.result())
        
        indices = [[2]]
        updates = [tf.one_hot(1, 4)]
        scores = tf.tensor_scatter_nd_update(scores, indices, updates)
        
        expectation = tf.constant(28.0 / 32)
        accuracy_score.update(labels, scores)
        self.assertAllClose(expectation, accuracy_score.result())
    
    def test_macroAverage_with_class_weights(self):
        labels, scores = (self.labels, self.scores)
        # trivial test
        expectation = tf.constant(13.0 / 16.0)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.MACRO)
        accuracy_score.update(labels, scores, weights=[[1, 1, 1, 1]])
        self.assertAllClose(expectation, accuracy_score.result())
        accuracy_score.reset()
        
        labels, scores = (self.labels, self.scores)
        expectation = tf.constant(6.0 / 16.0)
        accuracy_score = AccuracyScore(averaging_mode=AveragingMode.MACRO)
        accuracy_score.update(labels, scores, weights=[[0, 0, 1, 1]])
        self.assertAllClose(expectation, accuracy_score.result())
        accuracy_score.reset()
    # TODO: Add tests for weighted and None Averaging


class TestPrecisionScore(tf.test.TestCase):
    def setUp(self):
        self.labels = tf.stack([tf.one_hot(0, 4),
                                tf.constant([1.0, 0, 0.0, 1.0]),
                                tf.one_hot(1, 4),
                                tf.constant([1.0, 1.0, 0, 0.]),
                                tf.constant([1.0, 0.0, 1.0, 0.])
                                ], axis=0)
        self.scores = tf.stack([tf.constant([0.6, 0.3, 0.1, 0.2]),
                                tf.constant([0.9, 0.4, 0.2, 0.1]),
                                tf.one_hot(2, 4),
                                tf.constant([1.0, 1.0, 0, 0.]),
                                tf.constant([1.0, 1.0, 0, 0.])
                                ], axis=0)
    
    def test_microAverage_unweighted(self):
        labels, scores = self.labels, self.scores
        precision_score = PrecisionScore(averaging_mode=AveragingMode.MICRO)
        precision_score.update(labels, scores)
        expectation = tf.constant(5 / 7.)
        result = precision_score.result()
        self.assertAllClose(expectation, result)
        
    def test_macroAverage_unweighted(self):
        labels, scores = self.labels, self.scores
        precision_score = PrecisionScore(averaging_mode=AveragingMode.MACRO)
        precision_score.update(labels, scores)
        expectation = tf.constant(6 / 16.)
        result = precision_score.result()
        self.assertAllClose(expectation, result)


class TestRecallScore(tf.test.TestCase):
    def setUp(self):
        self.labels = tf.stack([tf.one_hot(0, 4),
                                tf.constant([1.0, 0, 0.0, 1.0]),
                                tf.one_hot(1, 4),
                                tf.constant([1.0, 1.0, 0, 0.]),
                                tf.constant([1.0, 0.0, 1.0, 0.])
                                ], axis=0)
        self.scores = tf.stack([tf.constant([0.6, 0.3, 0.1, 0.2]),
                                tf.constant([0.9, 0.4, 0.2, 0.1]),
                                tf.one_hot(2, 4),
                                tf.constant([1.0, 1.0, 0, 0.]),
                                tf.constant([1.0, 1.0, 0, 0.])
                                ], axis=0)
    
    def test_microAverage_unweighted(self):
        labels, scores = self.labels, self.scores
        recall_score = RecallScore(averaging_mode=AveragingMode.MICRO)
        recall_score.update(labels, scores)
        expectation = tf.constant(5 / 8.)
        result = recall_score.result()
        self.assertAllClose(expectation, result)
    
    def test_macroAverage_unweighted(self):
        labels, scores = self.labels, self.scores
        recall_score = RecallScore(averaging_mode=AveragingMode.MACRO)
        recall_score.update(labels, scores)
        expectation = tf.constant(6 / 16.)
        result = recall_score.result()
        self.assertAllClose(expectation, result)


class TestAveragePrecisionScore(tf.test.TestCase):
    def setUp(self):
        self.labels = tf.stack([tf.one_hot(0, 4),
                                tf.constant([1.0, 0, 0.0, 1.0]),
                                tf.one_hot(1, 4),
                                tf.constant([1.0, 1.0, 0, 0.]),
                                tf.constant([1.0, 0.0, 1.0, 0.])
                                ], axis=0)
        self.scores = tf.stack([tf.constant([0.6, 0.3, 0.1, 0.2]),
                                tf.constant([0.9, 0.4, 0.2, 0.1]),
                                tf.one_hot(2, 4),
                                tf.constant([1.0, 1.0, 0, 0.]),
                                tf.constant([1.0, 1.0, 0, 0.])
                                ], axis=0)
    
    def test_microAverage_unweighted(self):
        labels, scores = self.labels, self.scores
        mAP = AveragePrecisionScore(averaging_mode=AveragingMode.MICRO, bifurcators=500)
        mAP.update(labels, scores)
        result = mAP.result()
        expectation =\
            average_precision_score(labels, scores, average="micro")
        self.assertAllClose(expectation, result)

        mAP = AveragePrecisionScore(transform=tf.nn.sigmoid,
                                    averaging_mode=AveragingMode.MICRO,
                                    bifurcators=500)
        
        npzfile = np.load('data_00.npz')
        labels, scores = npzfile['labels'], npzfile['scores']
        #scores = tf.sigmoid(scores)
        mAP.update(labels, scores)
        result = mAP.result()
        expectation =\
            average_precision_score(labels, scores, average="micro")
        self.assertAllClose(expectation, result)
    
    def test_macroAverage_unweighted(self):
        labels, scores = self.labels, self.scores
        mAP = AveragePrecisionScore(averaging_mode=AveragingMode.MACRO, bifurcators=500)
        mAP.update(labels, scores)
        result = mAP.result()
        mAP_ref = average_precision_score(labels, scores, average='macro')
        self.assertAllClose(mAP_ref, result)
    
    def test_weightedAverage_unweighted(self):
        labels, scores = self.labels, self.scores
        mAP = AveragePrecisionScore(averaging_mode=AveragingMode.WEIGHTED, bifurcators=500)
        mAP.update(labels, scores)
        result = mAP.result()
        mAP_ref = average_precision_score(labels, scores, average='weighted')
        self.assertAllClose(mAP_ref, result)

    def test_unweighted(self):
        labels, scores = self.labels, self.scores
        mAP = AveragePrecisionScore(averaging_mode=None, bifurcators=500)
        mAP.update(labels, scores)
        result = mAP.result()
        mAP_ref = average_precision_score(labels, scores, average=None)
        self.assertAllClose(mAP_ref, result)


if __name__ == "__main__":
    tf.test.main()
