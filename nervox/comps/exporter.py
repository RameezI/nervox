import os
import logging
import tensorflow as tf
from typing import Dict
from pathlib import Path
from nervox.core.protocol import Protocol
from nervox.utils import Signatures
from dataclasses import asdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Exporter:
    def __init__(
        self,
        checkpoint_dir: os.PathLike,
        signatures: Signatures,
        ckpt_file: str = None,
        expect_partial: bool = False,
    ):
        """
        Args:
            checkpoint_dir (os.PathLike):   Provides the directory with checkpoints, from where the variables are restored.
            protocol (Protocol):            A protocol instance that defines the training, evaluation and prediction logics.
            signatures (Signatures):        A dictionary of signatures for the training, evaluation and prediction.
            ckpt_file (str):                The checkpoint file to be loaded. Defaults to None, the latest checkpoint file is loaded.
            expect_partial (bool):          Whether to expect a partial loadind  of the variables from the checkpoints or not.
                                            Defaults to False. raises an error if not all the variables are found in the checkpoint.
        """
        self.checkpoint_dir = checkpoint_dir
        self._epoch: tf.Variable = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.signatures = signatures
        self.ckpt_file = ckpt_file
        self.expected_partial = expect_partial
        self.export_dir = None

    @property
    def export_dir(self):
        return self._export_dir

    def _load_checkpoint(self, modules: Dict[str, tf.keras.Model]):
        # TODO: add support for loading from a specific checkpoint through the ckpt_file argument
        checkpoint = tf.train.Checkpoint(epoch=self._epoch, **modules)
        latest_ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_ckpt is None:
            raise FileNotFoundError
        status = checkpoint.restore(latest_ckpt)
        status.expect_partial(self.expect_partial)
        return status

    def push(self, protocol: Protocol, output_dir: os.PathLike = None):
        """
        The export method is responsible for exporting the compiled training, evaluation and prediction graphs to a directory.
        The compiled graphs can be executed outside the python environment, for example, using the tensorflow serving.
        Exported uses SavedModel format, which is converted to other formats using converters such as, tftrt tflite, etc.
        Te compiled graphs .
        Args:
            protocol (Protocol):    The protocol instance that was used to train the nn modules. The protocol must be compiled.
                                    The export will fail if protocol is not compiled. This is because the nerural network modules are
                                    linked to the protocol only when compiled.
                                    When nervox trainer is used on the protocol, the protocol is compiled automatically. Otherwise,
                                    the protocol must be compiled manually:
                                        ``` modules ={module_name: module_instance, ...}}
                                            protocol.compile()```
            save_dir (str):             The directory where the SavedModel format, .pb serailized graphs, will be exported.
        """

        if output_dir is not None:
            output_dir = Path(Path(self.checkpoint_dir).parent, "export")
            self._export_dir = output_dir

        if not protocol.is_compiled:
            raise ValueError("Protocol must be compiled before exporting the modules!")

        ckpt_load_status = self._load_checkpoint(protocol.modules)
        ckpt_load_status.assert_existing_objects_matched()
        output_dir.mkdir(exist_ok=True)
        signatures = asdict(self.signatures)

        exports = {
            "predict": protocol.predict_step,
            "evaluate": protocol.evaluate_step,
            "train": protocol.train_step,
        }

        # fmt: off
        signatures = {
            key: tf.function(exports[key], input_signature=signatures[key])
            for key in signatures if key is not None and key in exports }
    
        # export the requested signatures
        tf.saved_model.save(self, output_dir, signatures=signatures)

    def push_tflite(self, output_dir: str):
        """_summary_

        Args:
            output_dir (str): _description_
        """
        raise NotImplementedError

    def push_tftrt(self, output_dir: str):
        """_summary_

        Args:
            output_dir (str): _description_
        """
        raise NotImplementedError

    def push_onnx(self, output_dir: str):
        """_summary_

        Args:
            output_dir (str): _description_
        """
        raise NotImplementedError
