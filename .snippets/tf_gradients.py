"""
This script shows how to compute the gradient of a function with respect to a variable
is computed using the tf.GradientTape context manager.
"""

import tensorflow as tf
import logging
logging.basicConfig(level=logging.INFO, format=' %(message)s')
logger = logging.getLogger("tf_gradients")

# Define a function
def f(x):
    return tf.square(a * x) + b * x + c


if __name__ == "__main__":
    # Define some input Tensors
    x = tf.Variable(1.0)
    a = tf.Variable(2.0)
    b = tf.Variable(3.0)
    c = tf.Variable(4.0)
    
    # Compute the gradient of a  with respect to x, treating a, b, and c as constants

    # equation = (a*x)^2 + b*x + c --> d/dx = 2ax^2 *(2x) + b = 4*2 + 3 = 11
    #                             --> d/da =  2ax^2 = 4
    #                             --> d/db =  b = 1
    #                             --> d/dc =  c = 1

    with tf.GradientTape(persistent=True) as tape:
        output = f(x)

    grad_x = tape.gradient(output, x)
    grad_a = tape.gradient(output, a)
    grad_b = tape.gradient(output, b)
    grad_c = tape.gradient(output, c)
    
       # Print the result
    logger.info('\nGradients Computed by tf.GradientTape:')
    logger.info(f'd/dx: --> {grad_x:>5}')
    logger.info(f'd/da: --> {grad_a:>5}')
    logger.info(f'd/db: --> {grad_b:>5}')
    logger.info(f'd/dc: --> {grad_c:>5}')

    del tape  # Drop the reference to the tape