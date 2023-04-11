"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import argparse
import logging
from nervox.core import Objective
from metaflow import FlowSpec, step
from nervox.protocols import Classification

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cliargs_parser() -> argparse.Namespace:
    from nervox.utils import base_parser

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


# Learning objective configurator
def objective_configurer() -> Objective:
    """This function return objective(s) instance for the training protocol (supervised_classification here).
    Returns: Training Objective(s). Objective is a combination of a loss function, an optimizer and metrics.
    """
    import tensorflow as tf
    from nervox.core import Objective
    from nervox.losses import CrossEntropy
    from nervox.metrics.classification import AccuracyScore, AveragingMode
    from nervox.transforms import onehot_transform

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    xentropy = CrossEntropy(transform=tf.nn.sigmoid)
    accuracy = AccuracyScore(onehot_transform, averaging_mode=AveragingMode.SAMPLE)
    return Objective(xentropy, optimizer=optimizer, metrics=[accuracy])


class Cifar10(FlowSpec):
    """This is a pursuit for a learning classification task on CIFAR10 dataset. The workflow is defined in the
    steps below. The pursuit is executed by running the following command: `python workflow_cifar10.py run`.
    Args:
        FlowSpec: The base class for all flowspecs.
    """

    @step
    def start(self):
        parameters = vars(cliargs_parser())
        for key, value in parameters.items():
            setattr(self, key, value)
        self.next(self.training)

    @step
    def training(self):
        from nervox import Trainer

        from nervox.data.transforms import Normalize, OneHotLabels
        from nervox.data import DataStream
        from nervox.utils import VerbosityLevel
        from nervox.models.terminals import GlobalAvgPoolDecoder

        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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

        # Trainer
        trainer = Trainer(
            train_stream,
            eval_stream,
            name="cifar10",
            ckpt_opts={"monitor": "accuracy/val"},
        )

        # Add modules as required by the protocol

        trainer.push_module(
            self.encoder,
            alias="encoder",
        )

        trainer.push_module(
            GlobalAvgPoolDecoder,
            alias="decoder",
            config={"output_units": 10},
        )

        protocol = Classification(
            supervision_keys=("image", "label"),
            configurator=objective_configurer,
        )

        # spin the training protocol
        trainer.spin(
            protocol,
            max_epochs=10,
            callback_list=[],
            verbose=VerbosityLevel.UPDATE_AT_BATCH,
            # skip_initial_evaluation=True,
        )
        self.checkpoint_dir = trainer.checkpoint_dir
        self.next(self.evaluation)

    @step
    def evaluation(self):
        self.next(self.exportation)

    @step
    def exportation(self):
        import tensorflow as tf
        from nervox import Exporter
        from nervox.utils import Signatures

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
        exporter = Exporter(self.checkpoint_dir, export_signatures)
        exporter.push(protocol)

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    Cifar10()
