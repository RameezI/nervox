import os
import logging
import tensorflow as tf
from typing import Dict
from pathlib import Path
from nervox.core.protocol import Protocol
from nervox.utils import Signatures
from dataclasses import asdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
            signatures (Signatures):        The input signatures for the training, evaluation and prediction step.
            ckpt_file (str):                The checkpoint file to be loaded. Defaults to None, the latest checkpoint file is loaded.
            expect_partial (bool):          Whether to expect a partial loadind  of the variables from the checkpoints or not.
                                            Defaults to False. raises an error if not all the variables are found in the checkpoint.
        """
        self.checkpoint_dir = checkpoint_dir
        self._epoch: tf.Variable = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._signatures = asdict(signatures)
        self._ckpt_file = ckpt_file
        self._expect_partial = expect_partial
        self._export_path = None
        self._endpoints = {}

    @property
    def export_path(self):
        return self._export_path

    def _load_checkpoint(self, modules: Dict[str, tf.keras.Model]):
        # TODO: add support for loading from a specific checkpoint through the ckpt_file argument
        checkpoint = tf.train.Checkpoint(epoch=self._epoch, **modules)
        latest_ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_ckpt is None:
            raise FileNotFoundError
        status = checkpoint.restore(latest_ckpt)
        status.expect_partial() if self._expect_partial else None
        return status

    def add_endpoint(
        self, name: str, endpoint: callable, signature: tf.TensorSpec = None
    ):
        """
        Register a new serving endpoint.
        Arguments:

            name:       Name of the endpoint.
            endpoint:   An additional method of the Protocol class to be exported as an endpoint.
                        It should only leverage resources (e.g. `tf.Variable` objects or `tf.lookup.StaticHashTable` objects)
                        that are already tracked by the `Protocol` instance. If needed add a resource by assigning it as
                        an attribute of the `Protocol` instance.
            signature:  Used to specify the shape and dtype of the inputs for the endpoint.

        Example:
        ```python
        protocol = Protocol()
        exporter = Exporter()
        setattr(protocol, 'additional_resource', tf.nn.Module())
        export.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
        )
        ```
        """

        if signature:
            decorated_fn = tf.function(endpoint, input_signature=signature)
            self._endpoints[name] = signature

        else:
            if isinstance(endpoint, tf.types.experimental.GenericFunction):
                if not endpoint._list_all_concrete_functions():
                    raise ValueError(
                        f"The provided tf.function '{endpoint}' "
                        "has never been called. "
                        "To specify the expected shape and dtype "
                        "of the function's arguments, "
                        "you must either provide a function that "
                        "has been called at least once, or alternatively pass "
                        "an `input_signature` argument in `add_endpoint()`."
                    )
                decorated_fn = endpoint
            else:
                raise ValueError(
                    "If the `fn` argument provided is not a `tf.function`, "
                    "you must provide an `input_signature` argument to "
                    "specify the shape and dtype of the function arguments. "
                )
        self._endpoints.update({name: decorated_fn})

    def push(self, protocol: Protocol, output_path: os.PathLike = None):
        """
        The export method is responsible for exporting the compiled training, evaluation and prediction steps through serialization.
        The compiled graphs can be executed outside the python environment, for example, using the tensorflow serving.
        Exported endpoints use SavedModel format, which can be the  converted to other formats, such as, tftrt tflite, etc.
        Args:
            protocol (Protocol):    The protocol instance that was used to train the nn modules. The protocol must be compiled.
                                    The export will fail if protocol is not compiled. This is because the nerural network modules are
                                    linked to the protocol only when compiled.
                                    When nervox trainer is used on the protocol, the protocol is compiled automatically. Otherwise,
                                    the protocol must be compiled manually:
                                        ``` modules ={module_name: module_instance, ...}}
                                            protocol.compile()```
            save_dir (str):         The directory/path where the SavedModel formatted (.pb) graphs are exported.
        """

        if output_path is None:
            output_path = Path(Path(self.checkpoint_dir).parent, "export")
            self._export_path = output_path

        if not protocol._is_compiled:
            raise ValueError("Protocol must be compiled before exporting!")

        ckpt_load_status = self._load_checkpoint(protocol.modules)
        ckpt_load_status.assert_existing_objects_matched()
        output_path.mkdir(exist_ok=True)

        endpoints = {
            "predict": protocol.predict_step_tf,
            "evaluate": protocol.evaluate_step_tf,
            "train": protocol.train_step_tf,
        }

        concrete_endpoints = {
            key: endpoints[key].get_concrete_function(self._signatures[key])
            for key in endpoints
            if key in self._signatures and self._signatures[key] is not None
        }

        # add user defined endpoints as well and piroritize them.
        concrete_endpoints.update(self._endpoints)

        tf.saved_model.save(protocol, output_path, signatures=concrete_endpoints)

    # def _push_tflite(self, saved_model_dir:os.PathLike, output_dir: os.PathLike):
    #     """_summary_

    #         Args:
    #             saved_model_dir (os.PathLike):  _description_
    #             output_dir (os.PathLike):       _description_
    #         """
    #     raise NotImplementedError

    # def _push_tftrt(self,saved_model_dir:os.PathLike, output_dir: os.PathLike):
    #     """_summary_

    #         Args:
    #             saved_model_dir (os.PathLike):  _description_
    #             output_dir (os.PathLike):       _description_
    #         """
    #     raise NotImplementedError

    # def _push_onnx(self, saved_model_dir:os.PathLike, output_dir: os.PathLike):
    #     """_summary_

    #     Args:
    #         saved_model_dir (os.PathLike):  _description_
    #         output_dir (os.PathLike):       _description_
    #     """
    #     raise NotImplementedError
