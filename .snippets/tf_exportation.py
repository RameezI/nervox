import tempfile
import numpy as np
import tensorflow as tf
from typing import Tuple
from pathlib import Path


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
    def compute_loss(self):
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
        print(f"(a,  b): ({module.a.numpy()}, {module.b.numpy()})")
        print("\n")

    graph = module.compute_loss.get_concrete_function().graph
    print("\n")

    # print the resource handles
    print(f"resource/x = {module.a.handle}")
    print(f"resource/y = {module.b.handle}")
    print("\n")

    # captured variables/placeholders
    print("Variable name to placeholder mapping:")
    for var, placeholder in graph.captures:
        print(f"{var} -->  {placeholder}")
    print("\n")

    # export the module to a SavedModel
    with tempfile.TemporaryDirectory() as export_dir:
        tf.saved_model.save(module, export_dir)
        [print(item) for item in list(Path(export_dir).rglob("*"))]
        print("\n")

        # list all variables in the SavedModel
        with tf.device("/cpu:0"):
            restored_module = tf.saved_model.load(export_dir)
            print(type(restored_module))

        print(module.trainable_variables)
        print("Variables in the Restored Model:")
        print(f"a: {restored_module.a}")
        print(f"b: {restored_module.b}")
        print("\n")

        restored_graph = restored_module.compute_loss.get_concrete_function().graph
        print("Variable name to placeholder mapping in the restored function:")
        for var, placeholder in restored_graph.captures:
            print(f"{var} -->  {placeholder}")
        print("\n")
