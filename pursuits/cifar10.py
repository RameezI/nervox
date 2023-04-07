"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import os
import sys
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from nervox import Trainer, Exporter
from nervox.protocols import Classification
from nervox.utils import base_parser, VerbosityLevel
from nervox.data.transforms import Normalize, OneHotLabels
from nervox.data import DataStream
from nervox.models.terminals import GlobalAvgPoolDecoder

# objective configurator
from nervox.losses import CrossEntropy
from nervox.metrics.classification import AccuracyScore, AveragingMode
from nervox.transforms import onehot_transform
from nervox.core import Objective, Protocol
from nervox.utils import Signatures

from metaflow import FlowSpec, Parameter, step
import logging

logging.basicConfig(stream=sys.stderr)

def configuration() -> argparse.Namespace:
    """Returns options for the pursuit, which are configurable through commandline interface"""
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
    args, _ = parser.parse_known_args()
    return args


class Cifar10(FlowSpec):
    @step
    def start(self):
        parameters = vars(configuration())
        for key, value in parameters.items():
            setattr(self, key, value)
        self.next(self.training)

    @step
    def training(self):
        # data stream for training
        train_stream = DataStream(
            "cifar10",
            version=self.dataset_version,
            split="train",
            batch_size=64,
            pkg=None,
            datasets_dir=self.datasets_dir,
            transforms=[Normalize(mean=0.0, std=1.0), OneHotLabels(num_labels=10)],
        )

        # data stream for validation
        eval_stream = DataStream(
            "cifar10",
            version=self.dataset_version,
            split="test",
            batch_size=64,
            pkg=None,
            datasets_dir=self.datasets_dir,
            transforms=[Normalize(mean=0.0, std=1.0), OneHotLabels(num_labels=10)],
        )

        # Training
        # fmt: off
        trainer = Trainer( train_stream, eval_stream, name="cifar10", ckpt_opts={"monitor": "accuracy/val"})
        # export_opts={"signatures": export_signatures})
        # fmt: on

        # Add one or more models as required by the protocol
        trainer.push_module(self.encoder, alias="encoder", config={})
        trainer.push_module(
            GlobalAvgPoolDecoder, alias="decoder", config={"output_units": 10}
        )

        # training protocol
        # pylint: disable=unexpected-keyword-arg
        protocol = Classification(
            supervision_keys=("image", "label"), configurator=objective_configurer
        )

        # Training
        trainer.spin(
            protocol,
            max_epochs=10,
            callback_list=[],
            verbose=VerbosityLevel.UPDATE_AT_BATCH,
            # skip_initial_evaluation=True,
        )
        self.next(self.evaluation)

    @step
    def evaluation(self):
        self.next(self.exportation)

    @step
    def exportation(self):
        self.next(self.end)

    @step
    def end(self):
        pass


def objective_configurer() -> Objective:
    import tensorflow as tf
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    xentropy = CrossEntropy(transform=tf.nn.sigmoid)
    accuracy = AccuracyScore(onehot_transform, averaging_mode=AveragingMode.SAMPLE)
    objective = Objective(xentropy, optimizer=optimizer, metrics=[accuracy])
    return objective


def exort(
    protocol: Protocol, checkopoint_dir: os.PathLike, output_path: os.PathLike = None
):
    """
    Export endpoints and the underlying resources (models and objectives)
    Args:
        protocol:           The protocol instance that contains the endpoints.
        checkpoint_dir:     The directory where the checkpoints are stored.
        output_path:        The path to the directory where the exported resources are stored.
    """

    import tensorflow as tf

    # Signatures
    export_signatures = Signatures(
        train={
            "image": tf.TensorSpec((None, 32, 32, 3), tf.float32),
            "label": tf.TensorSpec((None, 10), tf.float32),
        },
        evaluate={
            "image": tf.TensorSpec((None, 32, 32, 3), tf.float32),
            "label": tf.TensorSpec((None, 10), tf.float32),
        },
    )

    # Exporter
    exporter = Exporter(checkopoint_dir, export_signatures)
    exporter.push(protocol, output_path=output_path)


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

    # Training
    # fmt: off
    trainer = Trainer( train_stream, eval_stream, name="cifar10", ckpt_opts={"monitor": "accuracy/val"})
    # export_opts={"signatures": export_signatures})
    # fmt: on

    # Add one or more models as required by the protocol
    trainer.push_module(args.encoder, alias="encoder", config={})
    trainer.push_module(
        GlobalAvgPoolDecoder, alias="decoder", config={"output_units": 10}
    )

    # training protocol
    # pylint: disable=unexpected-keyword-arg
    protocol = Classification(
        supervision_keys=("image", "label"), configurator=objective_configurer
    )

    # Training
    trainer.spin(
        protocol,
        max_epochs=1,
        callback_list=[],
        verbose=VerbosityLevel.UPDATE_AT_BATCH,
        # skip_initial_evaluation=True,
    )


if __name__ == "__main__":
    Cifar10()
