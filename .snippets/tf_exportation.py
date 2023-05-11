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

import tempfile
import numpy as np
import tensorflow as tf
from typing import Tuple
from pathlib import Path
import logging


print(f'tf=__version__: {tf.__version__}')
logging.basicConfig(level=logging.INFO, format=" %(message)s")
logger = logging.getLogger("tf_recursive")
logger.setLevel(logging.INFO)

"""
This is a sample script to demonstrate how to export a tf.Module.
It contains a module with an arbitrary compute method comprised of
two variables. The variables are created outside the module and are
passed to the module as arguments. The module is exported and loaded
back to demonstrate that the variables are saved and loaded correctly.
"""

def create_variables() -> Tuple[tf.Variable, tf.Variable]:
    """This function creates the variables x and y and initializes them with random values.
    Returns:
        Tuple(tf.Variable, tf.Variable): Duplet of tf.Variable
    """
    return tf.Variable(np.random.rand()), tf.Variable(np.random.rand())


class MyModule(tf.Module):
    def __init__(self, a, b):
        """
        Any trackable types that are assigned to the attributes of the  module and are tracked by the Module. Even if
        the created outside the __init__ method, they are still tracked when assigned to the module as attributes, as
        is the case here. These tracked objects are saved in the SavedModel and are available when the SavedModel is
        loaded. The python types however are not saved in the SavedModel.

            Args:
                a (tf.Variable): The variable a, that is part of the module and tracked by the it.
                b (tf.Variable): The variable b, that is part of the module and tracked by the it.
        """
        super().__init__()
        self.a, self.b = (a, b)

    @tf.function
    def compute(self):
        y = self.a**2 + self.b**2 - 6 * self.a + 2 * self.b + 9
        return y**2


if __name__ == "__main__":
    """This is the main function that creates the variables within a trackable Module.
    The module also contains the function that computes the loss. The module is exported
    top a SavedModel and the SavedModel is loaded and the loss is computed.

    Learnings:

    """
    with tf.device("/cpu:0"):
        a, b = create_variables()
        module = MyModule(a, b)
       
    graph = module.compute.get_concrete_function().graph

    # print the resource handles
    logger.debug(f"\nResource Handles:")
    logger.debug(f"resource/a = {module.a.handle}")
    logger.debug(f"resource/b = {module.b.handle}")

    # captured variables/placeholders
    print("\nVariable name to placeholder mapping:")
    for var, placeholder in graph.captures:
        logger.debug(f"{var} -->  {placeholder}")

    # export the module to a SavedModel
    with tempfile.TemporaryDirectory() as export_dir:
        logger.info(f"\nExporting the module to {export_dir}")
        tf.saved_model.save(module, export_dir)
        [logger.info(item) for item in list(Path(export_dir).rglob("*"))]

        # list all variables in the SavedModel
        with tf.device("/cpu:0"):
            restored_module = tf.saved_model.load(export_dir)
            logger.info(f'\nRestored model of type: {type(restored_module)}')

        logger.info("\nVariables in the original module:")
        #logger.info(f'variables: {module.variables}')
        logger.info(f"(a,  b): ({module.a.numpy()}, {module.b.numpy()})\n")
        

        logger.info("\nVariables in the restored module:")
        #logger.info(f'variables: {restored_module.variables}')
        logger.info(f"(a,  b): ({restored_module.a.numpy()}, {restored_module.b.numpy()})\n")
        
        restored_graph = restored_module.compute.get_concrete_function().graph
        logger.info("\nVariable name to placeholder mapping in the restored function:")
        for var, placeholder in restored_graph.captures:
            logger.info(f"{var} -->  {placeholder}")