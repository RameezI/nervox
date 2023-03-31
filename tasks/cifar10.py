"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import argparse
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from nervox import Trainer
from nervox.protocols import Classification
from nervox.utils import base_parser, VerbosityLevel
from nervox.data.transforms import Normalize, OneHotLabels
from nervox.data import DataStream
from nervox.models.terminals import GlobalAvgPoolDecoder

# objective configurator
from nervox.losses import CrossEntropy
from nervox.metrics.classification import AccuracyScore, AveragingMode
from nervox.transforms import onehot_transform
from nervox.core import Objective
from nervox.utils import Signatures

def objective_configurer() -> Objective:
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    xentropy = CrossEntropy(transform=tf.nn.sigmoid)
    accuracy = AccuracyScore(onehot_transform, averaging_mode=AveragingMode.SAMPLE)
    objective = Objective(xentropy, optimizer=optimizer, metrics=[accuracy])
    return objective


def train(args: argparse.Namespace):
    # data stream for training
    train_stream = DataStream(
        "cifar10",
        version=args.dataset_version,
        split="train",
        batch_size=64,
        pkg=None,
        datasets_dir=args.datasets_dir,
        transforms=[Normalize(mean=0.0, std=1.0), OneHotLabels(num_labels=10)],
    )

    # data stream for validation
    eval_stream = DataStream(
        "cifar10",
        version=args.dataset_version,
        split="test",
        batch_size=64,
        pkg=None,
        datasets_dir=args.datasets_dir,
        transforms=[Normalize(mean=0.0, std=1.0), OneHotLabels(num_labels=10)],
    )

    export_signatures = Signatures(
        train={
            "image": tf.TensorSpec((None, 32, 32, 3)),
            "label": tf.TensorSpec((None, 10)),
        },
        evaluate={
            "image": tf.TensorSpec((None, 32, 32, 3)),
            "label": tf.TensorSpec((None, 10)),
        },
        predict={"image": tf.TensorSpec((1, 32, 32, 3))},
    )

    # Training
    # fmt: off
    trainer = Trainer( train_stream, eval_stream, name="cifar10", ckpt_opts={"monitor": "accuracy/val"},
                       export_opts={"signatures": export_signatures})
    # fmt: on

    # Add one or more models as required by the protocol
    trainer.push_module(args.encoder, alias="encoder", config={})
    trainer.push_module(
        GlobalAvgPoolDecoder, alias="decoder", config={"output_units": 10}
    )

    # training protocol
    protocol = Classification(
        supervision_keys=("image", "label"), configurator=objective_configurer
    )

    trainer.spin(
        protocol,
        max_epochs=1,
        callback_list=[],
        verbose=VerbosityLevel.UPDATE_AT_BATCH,
        # skip_initial_evaluation=True,
    )

    trainer.export(export_signatures=export_signatures)


def configuration() -> argparse.Namespace:
    """Returns the default options for the experiment (configurable through the commandline)"""
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument(
        "--dataset_version",
        required=False,
        type=str,
        default=":3.*.*",
        help="The version of the dataset to be used for training",
    )
    parser.add_argument(
        "--encoder",
        required=False,
        type=str,
        default="convnet",
        help="The version of the dataset to be used for training",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train(configuration())
