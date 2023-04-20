import io
import numpy as np
import tensorflow as tf
from typing import Tuple

"""
This is a sample script to demonstrate how to use TensorFlow to compute using tf variables.
lets solve the following problem:

    ```
    consider the following polynomial relation:
    a^2 + b^2 - 6a + 2b + 9 = 0
    we want to find a solution for the above equation!

    we can write it as minimization problem:
    minimize the f(a, b)^2 wrt a and b:

        where y = f(a, b) = a^2 + b^2 - 6a + 2b + 9
    ```
"""

def create_variables()->Tuple[tf.Variable, tf.Variable]:
    """This function creates the variables x and y and initializes them with random values.
    Returns:
        Tuple(tf.Variable, tf.Variable): Duplet of tf.Variable
    """
    return tf.Variable(np.random.rand()), tf.Variable(np.random.rand())

@tf.function
def compute_loss():
    y = a**2+ b**2 - 6*a + 2*b + 9  
    return y**2

if  __name__ == "__main__":
    """This is the main function that creates the variables and computes the loss.
       The loss is computed using the function compute_loss and is wrapped in a tf.function
       Learnings:

        - Variables are created using tf.Variable, normally, outside the graph building context,
          they are not part of the graph but are represented by placeholders nodes in the graph.
          For example:

            ```
            The graph building context created using tf.function, includes the following node:
            
            Node 1 (name: "ReadVariableOp/resource", op: "Placeholder"): 
                This node creates a placeholder tensor that will be fed with a resource.
        
            Node 2 (name: "ReadVariableOp", op: "ReadVariableOp"):
                This node reads a variable from the resource specified in Node 1 and
                outputs the value of the variable.
            ```

        -  Tensorflow resource management
        
            Variables and Input Tensors are mapped to the placeholders in a graph definition by the
            graph.captures. When a resource, variables, queues, mutexes, and more, is created, the
            tf resource manager generates a unique identifier for it, called a handle, and associates
            this handle with the resource. The handle is what is actually passed around and used by
            TensorFlow to reference the resource.

            The variables are assigned the resource data type used to represent a reference to a
            mutable tensor value that is stored outside of the TensorFlow runtime. This type is 
            used to manage resources such as variables, queues, and mutexes, which have state that
            persists across multiple graph executions.The resource type is similar to a pointer in
            other programming languages, in that it provides a way to refer to a value stored in 
            memory. However, unlike a pointer, a resource value in TensorFlow is an opaque  handle
            that cannot be directly accessed or manipulated by the user. Instead, the TensorFlow 
            runtime provides a set of operations that can be used to interact with resources, such
            as tf.assign() to assign a new value to a variable, or tf.QueueBase.enqueue() to add a
            new element to a queue.

      
            So, when you print graph.captures, you will see both captured tensors and captured
            variables, and their corresponding placeholders. The captured tensors will be passed
            to the function as regular tensors, while the captured variables will be handled by
            TensorFlow's resource management system. This means that the variables will be read
            from the resource specified in the placeholder, and the result will be passed to the
            function as a regular tensor. 


        - A graph building context is created using tf.function,

    """
    with tf.device("/cpu:0"):
        a, b = create_variables()
    print(f"(a,  b): ({a.numpy()}, {b.numpy()})")
    loss = compute_loss()
    print(loss) 
    graph = compute_loss.get_concrete_function().graph
    print(graph.as_graph_def())
    print("\n")

    # print the resource handles 
    print(f'resource/x = {a.handle}')
    print(f'resource/y = {b.handle}')
    print('\n')

    # captured variables/placeholders
    print("Variable name to placeholder mapping:")
    for var, placeholder in graph.captures:
        print(f'{var} -->  {placeholder}')
    print("\n")

    del a, b    # The underlying resource is not deleted!    
    # execute the function without tracing, to confirm that the variables are not deleted
    print(f"loss before deleting the variables: {loss}"
           f" == loss after deleting the variables: {compute_loss()}")



