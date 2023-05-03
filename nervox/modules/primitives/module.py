import numpy as np
import tensorflow as tf
from typing import Union, Collection, Mapping
from nervox.utils import capture_params, to_tensor_shape
from abc import abstractmethod


class Module(tf.Module):
    """This is an abstract class from which all modules are inherited from.

    A Module is a callable object that takes as input one or more tensors and
     outputs one or more tensors. It involves *computation*, defined in
    the `compute()` method, and a *state* variables and constants.

    State is intended to  be created in one of two places, at the convenience
    of the subclass implementer:
        * in `__init__()` during construction of the module;
        * in `build()` method, which may be called automatically on the first
          `__call__()` invocation or can be explicitly invoked by supplying the
          shape(s) of the input(s);

    Modules are recursively composable: If you assign a modules instance as an
    attribute of another module, the outer module will start tracking the state
    created by the inner layer.

    Args:

      trainable:    Boolean, whether the layer's variables should be trainable,
                    this applies recursively to all constituents of the module.

      name:         String name of the layer.

      dtype:        The dtype of the modules variables.

    Attributes:

      name:         The name of the layer (string).

      dtype:        The dtype of the module variables.

      variables:    List of all state variables with respect to which gradient descent
                    backpropogation is performed. Depending on the `trainable` attribute,
                    the variable might not be updated during backpropogation.

      datastore:    List of non-trainable persistent states, these can be constants
                    or variables that are updated manually. These are not listed in
                    `variables` properties and thus are not updated by optimizers.

      state:        List of all variables and the tensors in the datastore.
                    The state of a composite module is the union of the state
                    of all of its constituents.

      trainable:    Whether the module variables can be trained (boolean).
                    When a module is not trainable, the variables property
                    returns an empty list thus skipping the update of the
                    variables.

    The descendants of `Module` implements the following methods:

    * `__init__()`: Defines module's attributes, and creates state variables. When
    the state is requires input_shape, it is recommended to defer the creation to
    `build()` instead.

    * `build(self, input_shape)`: This method can be used to create variables that
      depend on the shape(s) of the input(s), state. `__call__()` will automatically
      build the layer (if it has not been built yet by calling `build()`.

    * `compute(self, inputs, *args, **kwargs)`: Called in `__call__` after making
      sure `build()` has been called. `compute()` performs the forward computation.
      The first invocation may additionally create state.
      A typical signature for this method is `call(self, inputs, training=True)`.
      A reserved keyword arguments you can optionally use in `compute` is `training`:
        - `training` (boolean, whether the call is in inference mode or training mode).

    Examples:

    Here's a basic example: a module with two variables, `w` and `b`,
    that returns `y = w . x + b`.
    It shows how to implement `build()` and `compute()`.
    variables set as attributes of Module are automatically collected.

    ```python
    import numpy as np
    import nervox as nvx

    class SimpleDense(nvx.Module):

      def __init__(self, units=32):
          super().__init__()
          self.units = units

      def build(self, input_shape):
        # Create the state of the module

        kernel = tf.random_normal_initializer()(
                    shape=(input_shape[-1], self.units),
                    dtype=self.dtype)

        bias = tf.zeros_initializer()(shape=(self.units,),
                    dtype=self.dtype)

        self.kernel = tf.Variable(initial_value=kernel,
                        trainable=True)

        self.bias = tf.Variable(initial_value=bias
                        trainable=True)

      def compute(self, inputs):
          return tf.matmul(inputs, self.kernel) + self.bias

    # Instantiates the layer.
    linear_module = SimpleDense(4)

    # This will also call `build` and create the weights.
    y = linear_module(tf.ones((2, 2)))
    assert len(linear_module.weights) == 2

    # These weights are trainable, so they're listed in `variables`:
    assert len(linear_module.trainable_weights) == 2
    ```
    """

    def __init_subclass__(cls, *args, **kwargs):
        """The __init_subclass__ method is called when a subclass of Module is defined.
        This method wraps the user `__init__` method with the capture_params, which enables
        automatic capturing of the objects parameterization. The parameterization is stored
        in the `params` attribute of the object.
        """
        super().__init_subclass__(*args, **kwargs)
        __user_init = getattr(cls, "__init__")
        __user_build = getattr(cls, "build", Module.build)

        # if the user has not already decorated the __init__ method, decorate it...
        if not hasattr(__user_init, "_wrapper_capture_params_"):
            __user_init = capture_params(__user_init, **kwargs)

        def __wrapped_init(self, *args, **kwargs):
            # call the user's __init__ method
            # we call the __init__ method of the base class already,
            # so if the user forgets to call super().__init__ in their
            # __init__ method, a default super().__init__ is in place.
            if cls.__base__ is Module:
                super(cls, self).__init__()
            __user_init(self, *args, **kwargs)

        def __wrapped_build(self, *args, **kwargs):
            # call the user's build method
            __user_build(self, *args, **kwargs)
            if __user_build is not Module.build:
                Module.build(self, *args, **kwargs)  # 1
                # ¹if the user build is other than\
                # the Module.build, also call the
                # default Module.build

        cls.__init__ = __wrapped_init
        cls.build = __wrapped_build

    def __init__(
        self,
        trainable: bool = True,
        name: str = None,
        dtype: tf.DType = tf.float32,
    ):
        """The __init__ method of the Module class. This is the base class for all modules.
        The __init__ method controls the `name`, `trainable` and `dtype` attributes of the
        module. The `name` attribute is useful for debugging and visualization purposes.
        The `trainable` attribute controls whether the variables of the module are trainable
        or not. The `dtype` attribute controls the data type of the variables of the module.

        Args:

            trainable (bool, optional): Weather the module variables are trainable.
                                        Defaults to True.

            name (str, optional):       The name of the module this name scope is prepended
                                        to all variables created within the module. Default
                                        is None, which derives name from the top level class.

            dtype (tf.DType, optional): The dtype of the module computations.This is to enable
                                        mixed precision training. Defaults to tf.float32.

        Raises:
            TypeError: When the `trainable` argument is not of type boolean.
            TypeError: When the `dtype` argument is not of type tf.DType.

        """
        super(Module, self).__init__(name=name)

        if not (
            isinstance(trainable, bool)
            or (
                isinstance(trainable, (tf.Tensor, tf.Variable))
                and trainable.dtype is tf.bool
            )
        ):
            raise TypeError(
                "Expected `trainable` argument to be of type boolean, "
                f"but got: {trainable}: {type(trainable).__name__}"
            )

        if not isinstance(dtype, tf.DType):
            raise TypeError(
                "Expected `dtype` argument to be of type tf.DType, "
                f"but got: {dtype}: {type(dtype).__name__}"
            )

        # private attributes
        self._trainable = trainable
        self._dtype = dtype

        # attributes set lazily by `build`
        self._input_spec = None
        self._built = False

        # Save outer name scope at module declaration so that it is preserved at
        # the actual layer construction.
        self._name_scope_on_declaration = tf.get_current_name_scope()

        self._datastore = []
        self._variables = []

        # private attributes that indicate if the caches are invalid
        self._variables_cache_invalid = True
        self._datastore_cache_invalid = True

    def build(self, input_shape: Union[tf.TensorShape, Collection[tf.TensorShape]]):
        """Create variables of the module.

        This is a method that implementers of subclasses of `nervox.Module` must override
        if they need a state-creation. It is invoked, when not yet built, automatically at
        the very first invocation of `compute()`.

        Args:
          input_shape: Instance of `TensorShape`, or a collection of such instances
            depending upon the input signature of the module.
        """
        if isinstance(input_shape, tf.TensorShape):
            self._input_spec = tf.TensorSpec(shape=input_shape, dtype=self.dtype)

        elif isinstance(input_shape, Mapping):
            self._input_spec = {
                name: tf.TensorSpec(shape=shape, dtype=self.dtype)
                for name, shape in input_shape.items()
            }
        elif isinstance(input_shape, Collection):
            self._input_spec = [
                tf.TensorSpec(shape=shape, dtype=self.dtype) for shape in input_shape
            ]
        else:
            raise TypeError(
                "Expected input shape to be an instance of TensorShape, or a Collection of "
                f"such instance but got:\n {input_shape}: {type(input_shape).__name__}"
            )
        self._built = True
        self._datastore_cache_invalid = True
        self._variables_cache_invalid = True

    @abstractmethod
    def compute(
        self,
        inputs: Union[tf.Tensor, Collection[tf.Tensor]],
        /,
        **kwargs,
    ):
        """This is where the modules computations lives.

        The `compute()` method may not create any state (except in its first
        invocation, wrapping the creation of variables or other resources in
        `tf.init_scope()`).  It is recommended, however,  to create all states,
        including `tf.Variable` and nested `Module` instances, in `__init__()`,
        or in the `build()` method. When a state mutation is required, adding
        or deleting variables from the `Module`'s `variables` collection, for
        example, it is recommended to do...

        Args:
          inputs:   Input tensor, or a collection of input tensors, this is a
                    position only  argument and the first argument passed to
                    any `compute` method.

          **kwargs: Additional keyword arguments, to be implemented by the subclass.

        Returns:
          A tensor or list/tuple of tensors.
        """
        ...

    def __call__(
        self,
        inputs: Union[tf.Tensor, Collection[tf.Tensor]],
        /,
        **kwargs,
    ):
        """Wraps the forward `compute` method. This method is called when the module is invoked.

          Args:
          *inputs: Positional arguments to be passed to `self.compute`.
          **kwargs: Keyword arguments to be passed to `self.compute`.

        Returns:
          Output tensor(s).

        Raises:
          ValueError: if the layer's `compute` method returns None (an invalid value).
          RuntimeError: if `super().__init__()` was not called in the constructor.
        """

        input_list = tf.nest.flatten(inputs)

        if any(isinstance(x, (tf.Tensor, np.ndarray, float, int)) for x in input_list):
            inputs = tf.nest.map_structure(tf.convert_to_tensor, inputs)
            input_list = tf.nest.flatten(inputs)

        if not self._built:
            self._build(inputs)

        outputs = self.compute(inputs, **kwargs)
        return outputs

    def _build(self, inputs):
        if self.input_spec is not None:
            self.assert_input_compatibility(self.input_spec, inputs, self.name)

        input_list = tf.nest.flatten(inputs)
        input_shapes = None

        # Converts Tensors / CompositeTensors to TensorShapes.
        if any(hasattr(x, "shape") for x in input_list):
            input_shapes = tf.nest.map_structure(lambda x: x.shape, inputs)
        else:
            # Converts nested structure of input shapes to TensorShapes.
            try:
                input_shapes = to_tensor_shape(inputs)
            except ValueError:
                pass

        with tf.init_scope():
            self.build(input_shapes)

    @property
    def dtype(self):
        """The dtype of the variables and datastores of the module."""
        return self._dtype

    @property
    def name(self):
        """Name of the module (string), set in the constructor."""
        return self._name

    @property
    def trainable(self):
        return self._trainable

    @property
    def built(self):
        """Whether the module has been built."""
        return self._built

    @trainable.setter
    def trainable(self, value):
        """Sets trainable attribute for this module and all composite modules.

        When this value is changed during training (e.g. with a
        `tf.keras.callbacks.Callback`) you need to call the parent
        `tf.keras.Model.make_train_function` with `force=True` in order to
        recompile the training graph.

        Args:
          value: Boolean with the desired state for the layer's trainable
            attribute.
        """
        for module in self._flattened_modules():
            module._trainable = value

    @property
    def input_spec(self):
        """`InputSpec` instance describing the input format for the module.
        When creating a module subclass, one can set `self.input_spec` to
        enable the layer to run input compatibility checks during the build.
        For example, a `Conv2D` layer can only be built on a single input
        tensor of rank 4. As such, you can set it in `__init__()` to issue
        a consistent error message when the user tries to build the module
        on an input of the wrong shape:

        ```python
        ndim = 4
        self.input_spec  = tf.TensorSpec(shape=(None,) * ndim,
                                        dtype=dtype, name=name
                                        )
        ```
        Now, if you try to build the module using an input shape that isn't
        rank 4 (for instance, an input of shape `(2,)`, it will raise a
        nicely-formatted error:   TODO (rameez): Add exact error message
        ```
        ValueError: Input 0 of layer conv2d is incompatible with the layer:
        expected ndim=4, found ndim=1. Full shape received: [2]
        ```
        Returns:
          A `tf.TensorSpec` instance, or nested structure of tf.TensorSpec.
        """
        return self._input_spec

    @property
    def variables(self):
        """List of all variables tracked by this module. When a module is
        freshly built, the `variables` are  collected  through reflection.
        All subsequent calls to this property will return  the cached value
        of variables. However, if a state mutation is detected, for example
        a call to build, the cache is invalidated and the list is recomputed.

        Returns:
          A list of trainable variables.
        """
        variables = []
        if not self.trainable:
            variables = []
        elif self.built and self._variables_cache_invalid:
            variables = self.trainable_variables
            self._variables = variables
            self._variables_cache_invalid = False
        else:
            variables = self._variables
        return variables

    @property
    def datastore(self):
        """List of all non trainable tensors tracked by this module. When
        freshly built, the persistent tensors are collected through reflection.
        All subsequent calls to this property will return the cached value of
        datastore. However, if a state mutation is detected, for example a call
        to build, the cache is invalidated and the list is recomputed.

        Returns:
          A list of non-trainable variables.
        """
        datastore = []
        if self.built and self._datastore_cache_invalid:
            datastore = self.non_trainable_variables
            self._datastore = datastore
            self._datastore_cache_invalid = False
        else:
            datastore = self._datastore
        return datastore

    @property
    def state(self):
        """List of all trainable and non trainable tensors of this module.
        When a key collision occurs between a variable and a datastore tensor,
        the variable is given precedence."""
        return self.variables + self.datastore
