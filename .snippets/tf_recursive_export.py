import os
import tempfile
import tensorflow as tf
from typing import Tuple
from pathlib import Path
import logging


print(f"tf=__version__: {tf.__version__}")
logging.basicConfig(level=logging.INFO, format=" %(message)s")
logger = logging.getLogger("tf_recursive")
logger.setLevel(logging.INFO)


_MAX_RECURSION_DEPTH = 50


def create_variables() -> Tuple[tf.Variable, tf.Variable]:
    """This function creates the variables.
    Returns:
        Tuple(tf.Variable, tf.Variable): Duplet of tf.Variable
    """
    return tf.Variable(0.8), tf.Variable(0.6)


class TerminalModule(tf.Module):
    def __init__(self, a, b):
        """
        A module that takes two variables as input and computes some arbitrary
        function.
            Args:
                a (tf.Variable): The variable a tracked by the module.
                b (tf.Variable): The variable b tracked by the module.
        """
        super().__init__()
        self.a, self.b = (a, b)

    def __call__(self):
        return self.compute()

    @tf.function
    def compute(self):
        y = self.a**2 + self.b**2 - 6 * self.a + 2 * self.b + 9
        return y**2


class MyModule(tf.Module):
    max_depth: int = _MAX_RECURSION_DEPTH

    def __init__(self):
        """This recursive module creates a tree of modules.
        The structure (at depth==2) of the tree is as follows:

            ²{ ¹{ ⁰{p, q}⁰, q }¹, q}²

        MyModule (parent)
          self.p MyModule (child)
              --> self.p (grandchild - terminal module)
              --> self.q (grandchild - terminal module)
          self.q (child - terminal module)
        """
        super().__init__()

        with self.name_scope:
            if MyModule.max_depth:
                MyModule.max_depth -= 1
                p = MyModule()
            else:
                p = TerminalModule(*create_variables())

            self.p = p
            self.q = TerminalModule(*create_variables())

    def __call__(self):
        return self.compute()

    @tf.function
    def compute(self):
        y = self.p.compute() + self.q.compute()
        return y


class MyModuleMono(tf.Module):
    max_depth: int = _MAX_RECURSION_DEPTH

    def __init__(self):
        """Here we create a monolithic module, that does the exact same computations
        as the recursive module above, but in a single call. This is to understand the
        implications, memory and storage wise, of using a recursive module vs a
        monolithic module afterwards by looking at the SavedModel.

        The recursive module simply adds the compute results of all the terminal modules in the
        recursive tree. Each recursion level creates an additional module (asymmetric development).

        --> at recursion depth == 0
             2 + 0 = 2 modules and 4 variables are created.
        --> at recursion depth == 1
             2 + 1 = 3 modules and 6 variables are created.
        --> at recursion depth == 2
             2 + 2 = 4 modules and 8 variables are created.
        ...
        ...
        --> at recursion depth == n
             2 + n = n + 2 modules and 2(n+2) variables are created.
        """
        super().__init__()

        with self.name_scope:
            self.terminals = [
                TerminalModule(*create_variables()) for _ in range(self.max_depth + 2)
            ]

    def __call__(self):
        return self.compute()

    @tf.function
    def compute(self):
        y = 0
        for terminal in self.terminals:
            y += terminal.compute()
        return y


# get directory size  in KB
def get_dir_size(dir_path):
    total_size = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / 1024


if __name__ == "__main__":
    """This is the main function that creates a recursive module and a monolithic module with same number
    of variables and compute. The final modules are exported to a SavedModel.

    Observations:
          - Using the recursive module, the SavedModel contains the entire tree of modules, wastes  memory
            and storage space. For example, if we set the recursion depth to 50, the SavedModel will
            contain 2 + 50 = 52 modules, 50 of which are child modules to whom we want to restrict access.
            The naively saved SavedModel ~3x increase in model definition size and ~3x increase in variable
            storage compared to a monolithic module.

          - The monolithic module here is  for comparison purposes, it has same number of variables and compute.
            The ideal solution would be to able to use resource restriction while exporting a recursively built
            module:  restrict access to variables and compute to only the immediate module or to children at
            certain depth only.

    """
    with tf.device("/cpu:0"):
        module = MyModule()
        module_mono = MyModuleMono()

    # get concrete functions and print graphs
    graph1 = module.compute.get_concrete_function().graph
    graph2 = module_mono.compute.get_concrete_function().graph
    print("\n")

    logger.info("Variables in the Module:")
    logger.info(
        f"trainable_variables [module_recursive]: {len(module.trainable_variables)}"
    )
    logger.info(
        f"trainable_variables [module_monolithic]: {len(module.trainable_variables)}\n"
    )

    # export the recursive module to a SavedModel
    with tempfile.TemporaryDirectory() as export_dir:
        tf.saved_model.save(module, export_dir)
        [logger.info(item) for item in list(Path(export_dir).rglob("*"))]

        # print stats about the SavedModel
        logger.info(f"\nRecursive Model:")
        logger.info(
            f"  saved_model.pb: {os.path.getsize(str(Path(export_dir, 'saved_model.pb')))/1024} KB"
        )
        logger.info(
            f"  variables: {get_dir_size(str(Path(export_dir, 'variables')))} KB"
        )
        logger.info(f"  assets: {get_dir_size(str(Path(export_dir, 'assets')))} KB\n")

        # list all variables in the SavedModel
        with tf.device("/cpu:0"):
            restored_recursive_module = tf.saved_model.load(export_dir)

    restored_graph = restored_recursive_module.compute.get_concrete_function().graph
    logger.debug(f"\nRestored_object_type: {type(restored_recursive_module).__name__}")
    logger.debug("variable names to placeholder mapping in the restored function:")
    for var, placeholder in restored_graph.captures:
        logger.debug(f"{var} -->  {placeholder}")

    # export the monolithic module to a SavedModel
    with tempfile.TemporaryDirectory() as export_dir:
        tf.saved_model.save(module_mono, export_dir)
        [logger.info(item) for item in list(Path(export_dir).rglob("*"))]

        # print stats about the SavedModel
        logger.info(f"\nMonolithic Model:")
        logger.info(
            f"  saved_model.pb: {os.path.getsize(str(Path(export_dir, 'saved_model.pb')))/1024} KB"
        )
        logger.info(
            f"  variables: {get_dir_size(str(Path(export_dir, 'variables')))} KB"
        )
        logger.info(f"  assets: {get_dir_size(str(Path(export_dir, 'assets')))} KB")

        # list all variables in the SavedModel
        with tf.device("/cpu:0"):
            restored_monolithic_module = tf.saved_model.load(export_dir)

    restored_graph = restored_monolithic_module.compute.get_concrete_function().graph
    logger.debug(f"Restored_object_type: {type(restored_monolithic_module).__name__}")
    logger.debug("variables name to placeholder mapping in the restored function:")
    for var, placeholder in restored_graph.captures:
        logger.debug(f"{var} -->  {placeholder}")

    logger.info("\nEquivalence of the restored model and the original model:")
    logger.info(f"recursive_module.compute: {module()}")
    logger.info(f"monolithic_module.compute: {module_mono()}")
    logger.info(
        f"restored_recursive_module.compute: {restored_recursive_module.compute()}"
    )
    logger.info(
        f"restored_monolithic_module.compute: {restored_monolithic_module.compute()}\n"
    )
