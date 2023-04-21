import numpy as np
import tensorflow as tf
from typing import Union, Collection, Mapping, Optional
from nervox.utils import capture_params


class Module(tf.Module):

    """This is an abstract  class which all modules are inherited from.

    A Module is a callable object that takes as input one or more tensors and
    that outputs one or more tensors. It involves *computation*, defined in
    the `compute()` method, and a *state* (variables). State is intended to  be
    created in one of two places, at the convenience of the subclass implementer:

    * in `__init__()` during construction of the module;
    * in `build()` method, which is creates the variable lazily
      on the the first `__call__()` to the module by supplying the shape(s) of the input(s);

    Layers are recursively composable: If you assign a Layer instance as an
    attribute of another Layer, the outer layer will start tracking the weights
    created by the inner layer. Nested layers should be instantiated in the
    `__init__()` method.

    Args:
      trainable:    Boolean, whether the layer's variables should be trainable,
                    this applies recursively to all layers it is composed of.

      name:         String name of the layer.
      dtype:        The dtype of the layer's variables.

    Attributes:

      name:         The name of the layer (string).

      dtype:        A tuple of (dtype_variable,  dtype_compute). Module may automatically
                    cast inputs to compute_dtype which causes the computations and output to
                    also be in compute_dtype. This is useful for mixed precision training.

    #   variables:    List of all state variables with respect to which gradient descent
    #                 backpropogation is performed. Depending on the `trainable` attribute,
    #                 the variable might not be updated during backpropogation.

    #   constants:    List of all non_trainable state variables, variables that are treated as
    #                 constants during backpropogation.These are not constants in the sense
    #                 of `tf.constant`, but rather constant variables that are not updated
    #                 during the backpropogation.

      trainable:    Whether the layer should be trained (boolean), i.e. whether its
                    potentially-trainable variables should be returned as part of
                    `layer.variables`.

    The descendants of `Module` must implement the following methods:

    * `__init__()`: Defines custom layer attributes, and creates layer weights
      that do not depend on input shapes, using `add_weight()`, or other state.

    * `build(self, input_shape)`: This method can be used to create variables that
      depend on the shape(s) of the input(s), state. `__call__()` will automatically
      build the layer (if it has not been built yet by calling `build()`.

    * `compute(self, inputs, *args, **kwargs)`: Called in `__call__` after making
      sure `build()` has been called. `compute()` performs the forward computation.
      The first invocation may additionally create state.

      A reserved keyword arguments you can optionally use in `compute` is `training`:
        - `training` (boolean, whether the call is in inference mode or training mode).
      A typical signature for this method is `call(self, inputs, training=True)`.

    Examples:

    Here's a basic example: a layer with two variables, `w` and `b`,
    that returns `y = w . x + b`.
    It shows how to implement `build()` and `compute()`.
    variables set as attributes of Module are automatically collected.

    ```python
    import numpy as np

    class SimpleDense(Layer):

      def __init__(self, units=32):
          super(SimpleDense, self).__init__()
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
    linear_layer = SimpleDense(4)

    # This will also call `build(input_shape)` and create the weights.
    y = linear_layer(tf.ones((2, 2)))
    assert len(linear_layer.weights) == 2

    # These weights are trainable, so they're listed in `variables`:
    assert len(linear_layer.trainable_weights) == 2
    ```
    """

    def __init_subclass__(cls, *args, **kwargs):
        """The __init_subclass__ method is called when a subclass of Module is defined.
        This method wraps the user `__init__` method with the capture_params, which enables
        automatic capturing of the objects parameterization. The parameterization is stored
        in the `params` attribute of the object.
        """
        super().__init_subclass__(*args, **kwargs)
        cls.__user_init__ = cls.__init__
        # if the user has not already decorated the __init__ method, decorate it...
        if not hasattr(cls.__user_init__, "_wrapper_capture_params_"):
            cls.__user_init__ = capture_params(cls.__init__, **kwargs)

        def _wrapped_init(self, *args, **kwargs):
            # call the user's __init__ method
            super().__init__()
            self.__user_init__(*args, **kwargs)

        cls.__init__ = _wrapped_init

    def __init__(self, trainable=True, name=None, dtype=None):
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

        # private attributes
        self._trainable = trainable
        self._built = False
        self._input_spec = None
        self._dtype = dtype

        # A list of variables created by this module, all state variables with respect to which
        #  gradient descent backpropogation is performed. Depending on the `trainable` attribute,
        # the variable might not be updated during backpropogation.
        self._variables = []

        # A list of constant variables created by this module, these variables are not trainable
        #  and must not be included for backpropogation / credit assignment.
        self._constants = []

        # A list of tensors containing activity regularizers and losses manually added
        #  through `add_loss` method of the module.
        self._losses = []

        # A list of metric instances corresponding to the metric tensors added using the
        #  `add_metric` method of the module.
        self._metrics = []

        # Save outer name scope at module declaration so that it is preserved at
        # the actual layer construction.
        self._name_scope_on_declaration = tf.get_current_name_scope()

    def build(self, input_shape: Union[tf.TensorShape, Collection[tf.TensorShape]]):
        """Creates the variables of the module.

        This is a method that implementers of subclasses of `nervox.Module`
        must override if they need a state-creation .It is invoked automatically
        before the first execution of `compute()` if not yet built.

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
        self.built = True


    def compute(
        self,
        inputs: Union[tf.Tensor, Collection[tf.Tensor]],
        /,
        *,
        training: Optional[Union[bool, tf.bool]] = None,
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

          training: An optional boolean indicating whether the requested computations
                    are to be carried out in training mode or otherwise.By default,
                    the `training` argument is set to `None`.

          **kwargs: Additional keyword arguments, to be implemented by the subclass.

        Returns:
          A tensor or list/tuple of tensors.
        """
        ...

    def __call__(
        self,
        inputs: Union[tf.Tensor, Collection[tf.Tensor]],
        /,
        *,
        training: Optional[Union[bool, tf.bool]] = None,
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

        if not self.built:
            self._maybe_build(inputs)
            outputs =self.compute(inputs, training=training, **kwargs)

        return outputs

    @property
    def dtype(self):
        """The dtype of the layer weights.

        This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless
        mixed precision is used, this is the same as `Layer.compute_dtype`, the
        dtype of the layer's computations.
        """
        return self._dtype

    @property
    def name(self):
        """Name of the layer (string), set in the constructor."""
        return self._name

    @property
    def trainable(self):
        return self._trainable

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
        """`InputSpec` instance(s) describing the input format for this module.

        When creating a layer subclass, one can set `self.input_spec` to
        enable the layer to run input compatibility checks during the build.
        Consider a `Conv2D` layer: it can only be built on a single input
        tensor of rank 4. As such, you can set it in `__init__()`:

        ```python
        ndim = 4
        self.input_spec  = tf.TensorSpec(shape=(None,) * ndim,
                                        dtype=dtype, name=name
                                        )
        ```

        Now, if you try to build the module using an input shape that isn't rank 4
        (for instance, an input of shape `(2,)`, it will raise a nicely-formatted error:
        TODO@Rameez: Add exact error message
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
        """List of all variables tracked by this module. 
        These variables are updated via gradient descent during training.
        TODO: Cache variables unless a state mutation is in sight.
              When can state mutation happen? 
              - When a variable is added or removed, during build or first call?
              - In which cases to recompute the flattened list of variables.

        Returns:
          A list of trainable variables.
        """
        variables = []
        if self.trainable and self.built:
            variables = self.trainable_variables 

        return variables 

    @property
    def non_trainable_weights(self):
        """List of all non-trainable weights tracked by this layer.

        Non-trainable weights are *not* updated during training. They are
        expected to be updated manually in `call()`.

        Returns:
          A list of non-trainable variables.
        """
        self._update_trackables()
        if self.trainable:
            children_weights = self._gather_children_attribute(
                "non_trainable_variables"
            )
            non_trainable_weights = self._non_trainable_weights + children_weights
        else:
            children_weights = self._gather_children_attribute("variables")
            non_trainable_weights = (
                self._trainable_weights + self._non_trainable_weights + children_weights
            )
        return self._dedup_weights(non_trainable_weights)

    @property
    def weights(self):
        """Returns the list of all layer variables/weights.

        Returns:
          A list of variables.
        """
        return self.trainable_weights + self.non_trainable_weights

    @property
    @doc_controls.do_not_generate_docs
    def updates(self):
        warnings.warn(
            "`layer.updates` will be removed in a future version. "
            "This property should not be used in TensorFlow 2.0, "
            "as `updates` are applied automatically.",
            stacklevel=2,
        )
        return []

    @property
    def losses(self):
        """List of losses added using the `add_loss()` API.

        Variable regularization tensors are created when this property is
        accessed, so it is eager safe: accessing `losses` under a
        `tf.GradientTape` will propagate gradients back to the corresponding
        variables.

        Examples:

        >>> class MyLayer(tf.keras.layers.Layer):
        ...   def call(self, inputs):
        ...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
        ...     return inputs
        >>> l = MyLayer()
        >>> l(np.ones((10, 1)))
        >>> l.losses
        [1.0]

        >>> inputs = tf.keras.Input(shape=(10,))
        >>> x = tf.keras.layers.Dense(10)(inputs)
        >>> outputs = tf.keras.layers.Dense(1)(x)
        >>> model = tf.keras.Model(inputs, outputs)
        >>> # Activity regularization.
        >>> len(model.losses)
        0
        >>> model.add_loss(tf.abs(tf.reduce_mean(x)))
        >>> len(model.losses)
        1

        >>> inputs = tf.keras.Input(shape=(10,))
        >>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
        >>> x = d(inputs)
        >>> outputs = tf.keras.layers.Dense(1)(x)
        >>> model = tf.keras.Model(inputs, outputs)
        >>> # Weight regularization.
        >>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
        >>> model.losses
        [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]

        Returns:
          A list of tensors.
        """
        collected_losses = []
        for layer in self._flattened_modules():
            # If any eager losses are present, we assume the model to be part of
            # an eager training loop (either a custom one or the one used when
            # `run_eagerly=True`) and so we always return just the eager losses.
            if layer._eager_losses:
                # Filter placeholder losses that may have been added by revived
                # layers.  (see base_layer_utils for details).
                if (
                    layer._eager_losses[0]
                    is not base_layer_utils.REVIVED_LOSS_PLACEHOLDER
                ):
                    collected_losses.extend(layer._eager_losses)
            else:
                collected_losses.extend(layer._losses)
            for regularizer in layer._callable_losses:
                loss_tensor = regularizer()
                if loss_tensor is not None:
                    collected_losses.append(loss_tensor)
        return collected_losses

    def add_loss(self, losses, **kwargs):
        """Add loss tensor(s), potentially dependent on layer inputs.

        Some losses (for instance, activity regularization losses) may be
        dependent on the inputs passed when calling a layer. Hence, when reusing
        the same layer on different inputs `a` and `b`, some entries in
        `layer.losses` may be dependent on `a` and some on `b`. This method
        automatically keeps track of dependencies.

        This method can be used inside a subclassed layer or model's `call`
        function, in which case `losses` should be a Tensor or list of Tensors.

        Example:

        ```python
        class MyLayer(tf.keras.layers.Layer):
          def call(self, inputs):
            self.add_loss(tf.abs(tf.reduce_mean(inputs)))
            return inputs
        ```

        The same code works in distributed training: the input to `add_loss()`
        is treated like a regularization loss and averaged across replicas
        by the training loop (both built-in `Model.fit()` and compliant custom
        training loops).

        The `add_loss` method can also be called directly on a Functional Model
        during construction. In this case, any loss Tensors passed to this Model
        must be symbolic and be able to be traced back to the model's `Input`s.
        These losses become part of the model's topology and are tracked in
        `get_config`.

        Example:

        ```python
        inputs = tf.keras.Input(shape=(10,))
        x = tf.keras.layers.Dense(10)(inputs)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs, outputs)
        # Activity regularization.
        model.add_loss(tf.abs(tf.reduce_mean(x)))
        ```

        If this is not the case for your loss (if, for example, your loss
        references a `Variable` of one of the model's layers), you can wrap your
        loss in a zero-argument lambda. These losses are not tracked as part of
        the model's topology since they can't be serialized.

        Example:

        ```python
        inputs = tf.keras.Input(shape=(10,))
        d = tf.keras.layers.Dense(10)
        x = d(inputs)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs, outputs)
        # Weight regularization.
        model.add_loss(lambda: tf.reduce_mean(d.kernel))
        ```

        Args:
          losses: Loss tensor, or list/tuple of tensors. Rather than tensors,
            losses may also be zero-argument callables which create a loss
            tensor.
          **kwargs: Used for backwards compatibility only.
        """
        kwargs.pop("inputs", None)
        if kwargs:
            raise TypeError(f"Unknown keyword arguments: {kwargs.keys()}")

        def _tag_callable(loss):
            """Tags callable loss tensor as `_unconditional_loss`."""
            if callable(loss):
                # We run the loss without autocasting, as regularizers are often
                # numerically unstable in float16.
                with autocast_variable.enable_auto_cast_variables(None):
                    loss = loss()
            if loss is None:
                # Will be filtered out when computing the .losses property
                return None
            if not tf.is_tensor(loss):
                loss = tf.convert_to_tensor(loss, dtype=backend.floatx())
            loss._unconditional_loss = True
            return loss

        losses = tf.nest.flatten(losses)

        callable_losses = []
        eager_losses = []
        symbolic_losses = []
        for loss in losses:
            if callable(loss):
                callable_losses.append(functools.partial(_tag_callable, loss))
                continue
            if loss is None:
                continue
            if not tf.is_tensor(loss) and not isinstance(
                loss, keras_tensor.KerasTensor
            ):
                loss = tf.convert_to_tensor(loss, dtype=backend.floatx())
            # TF Functions should take the eager path.
            if (
                tf_utils.is_symbolic_tensor(loss)
                or isinstance(loss, keras_tensor.KerasTensor)
            ) and not base_layer_utils.is_in_tf_function():
                symbolic_losses.append(loss)
            elif tf.is_tensor(loss):
                eager_losses.append(loss)

        self._callable_losses.extend(callable_losses)

        in_call_context = base_layer_utils.call_context().in_call
        if eager_losses and not in_call_context:
            raise ValueError(
                "Expected a symbolic Tensors or a callable for the loss value. "
                "Please wrap your loss computation in a zero argument `lambda`."
            )

        self._eager_losses.extend(eager_losses)

        for symbolic_loss in symbolic_losses:
            if getattr(self, "_is_graph_network", False):
                self._graph_network_add_loss(symbolic_loss)
            else:
                # Possible a loss was added in a Layer's `build`.
                self._losses.append(symbolic_loss)

    @property
    def metrics(self):
        """List of metrics added using the `add_metric()` API.

        Example:

        >>> input = tf.keras.layers.Input(shape=(3,))
        >>> d = tf.keras.layers.Dense(2)
        >>> output = d(input)
        >>> d.add_metric(tf.reduce_max(output), name='max')
        >>> d.add_metric(tf.reduce_min(output), name='min')
        >>> [m.name for m in d.metrics]
        ['max', 'min']

        Returns:
          A list of `Metric` objects.
        """
        collected_metrics = []
        for layer in self._flattened_modules():
            if not hasattr(layer, "_metrics_lock"):
                continue
            with layer._metrics_lock:
                collected_metrics.extend(layer._metrics)
        return collected_metrics

    def add_metric(self, value, name=None, **kwargs):
        """Adds metric tensor to the layer.

        This method can be used inside the `call()` method of a subclassed layer
        or model.

        ```python
        class MyMetricLayer(tf.keras.layers.Layer):
          def __init__(self):
            super(MyMetricLayer, self).__init__(name='my_metric_layer')
            self.mean = tf.keras.metrics.Mean(name='metric_1')

          def call(self, inputs):
            self.add_metric(self.mean(inputs))
            self.add_metric(tf.reduce_sum(inputs), name='metric_2')
            return inputs
        ```

        This method can also be called directly on a Functional Model during
        construction. In this case, any tensor passed to this Model must
        be symbolic and be able to be traced back to the model's `Input`s. These
        metrics become part of the model's topology and are tracked when you
        save the model via `save()`.

        ```python
        inputs = tf.keras.Input(shape=(10,))
        x = tf.keras.layers.Dense(10)(inputs)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs, outputs)
        model.add_metric(math_ops.reduce_sum(x), name='metric_1')
        ```

        Note: Calling `add_metric()` with the result of a metric object on a
        Functional Model, as shown in the example below, is not supported. This
        is because we cannot trace the metric result tensor back to the model's
        inputs.

        ```python
        inputs = tf.keras.Input(shape=(10,))
        x = tf.keras.layers.Dense(10)(inputs)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs, outputs)
        model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
        ```

        Args:
          value: Metric tensor.
          name: String metric name.
          **kwargs: Additional keyword arguments for backward compatibility.
            Accepted values:
            `aggregation` - When the `value` tensor provided is not the result
            of calling a `keras.Metric` instance, it will be aggregated by
            default using a `keras.Metric.Mean`.
        """
        kwargs_keys = list(kwargs.keys())
        if len(kwargs_keys) > 1 or (
            len(kwargs_keys) == 1 and kwargs_keys[0] != "aggregation"
        ):
            raise TypeError(
                f"Unknown keyword arguments: {kwargs.keys()}. "
                "Expected `aggregation`."
            )

        from_metric_obj = hasattr(value, "_metric_obj")
        is_symbolic = isinstance(value, keras_tensor.KerasTensor)
        in_call_context = base_layer_utils.call_context().in_call

        if name is None and not from_metric_obj:
            # Eg. `self.add_metric(math_ops.reduce_sum(x))` In eager mode, we
            # use metric name to lookup a metric. Without a name, a new Mean
            # metric wrapper will be created on every model/layer call. So, we
            # raise an error when no name is provided. We will do the same for
            # symbolic mode for consistency although a name will be generated if
            # no name is provided.

            # We will not raise this error in the foll use case for the sake of
            # consistency as name in provided in the metric constructor.
            # mean = metrics.Mean(name='my_metric')
            # model.add_metric(mean(outputs))
            raise ValueError(
                "Please provide a name for your metric like "
                "`self.add_metric(tf.reduce_sum(inputs), "
                "name='mean_activation')`"
            )
        elif from_metric_obj:
            name = value._metric_obj.name

        if not in_call_context and not is_symbolic:
            raise ValueError(
                "Expected a symbolic Tensor for the metric value, received: "
                + str(value)
            )

        # If a metric was added in a Layer's `call` or `build`.
        if in_call_context or not getattr(self, "_is_graph_network", False):
            # TF Function path should take the eager path.

            # If the given metric is available in `metrics` list we just update
            # state on it, otherwise we create a new metric instance and
            # add it to the `metrics` list.
            metric_obj = getattr(value, "_metric_obj", None)
            # Tensors that come from a Metric object already updated the Metric
            # state.
            should_update_state = not metric_obj
            name = metric_obj.name if metric_obj else name

            with self._metrics_lock:
                match = self._get_existing_metric(name)
                if match:
                    metric_obj = match
                elif metric_obj:
                    self._metrics.append(metric_obj)
                else:
                    # Build the metric object with the value's dtype if it
                    # defines one
                    metric_obj = metrics_mod.Mean(
                        name=name, dtype=getattr(value, "dtype", None)
                    )
                    self._metrics.append(metric_obj)

            if should_update_state:
                metric_obj(value)
        else:
            if from_metric_obj:
                raise ValueError(
                    "Using the result of calling a `Metric` object "
                    "when calling `add_metric` on a Functional "
                    "Model is not supported. Please pass the "
                    "Tensor to monitor directly."
                )

            # Insert layers into the Keras Graph Network.
            aggregation = None if from_metric_obj else "mean"
            self._graph_network_add_metric(value, aggregation, name)

    @doc_controls.do_not_doc_inheritable
    def add_update(self, updates):
        """Add update op(s), potentially dependent on layer inputs.

        Weight updates (for instance, the updates of the moving mean and
        variance in a BatchNormalization layer) may be dependent on the inputs
        passed when calling a layer. Hence, when reusing the same layer on
        different inputs `a` and `b`, some entries in `layer.updates` may be
        dependent on `a` and some on `b`. This method automatically keeps track
        of dependencies.

        This call is ignored when eager execution is enabled (in that case,
        variable updates are run on the fly and thus do not need to be tracked
        for later execution).

        Args:
          updates: Update op, or list/tuple of update ops, or zero-arg callable
            that returns an update op. A zero-arg callable should be passed in
            order to disable running the updates by setting `trainable=False`
            on this Layer, when executing in Eager mode.
        """
        call_context = base_layer_utils.call_context()
        # No need to run updates during Functional API construction.
        if call_context.in_keras_graph:
            return

        # Callable updates are disabled by setting `trainable=False`.
        if not call_context.frozen:
            for update in tf.nest.flatten(updates):
                if callable(update):
                    update()

    def set_weights(self, weights):
        """Sets the weights of the layer, from NumPy arrays.

        The weights of a layer represent the state of the layer. This function
        sets the weight values from numpy arrays. The weight values should be
        passed in the order they are created by the layer. Note that the layer's
        weights must be instantiated before calling this function, by calling
        the layer.

        For example, a `Dense` layer returns a list of two values: the kernel
        matrix and the bias vector. These can be used to set the weights of
        another `Dense` layer:

        >>> layer_a = tf.keras.layers.Dense(1,
        ...   kernel_initializer=tf.constant_initializer(1.))
        >>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
        >>> layer_a.get_weights()
        [array([[1.],
               [1.],
               [1.]], dtype=float32), array([0.], dtype=float32)]
        >>> layer_b = tf.keras.layers.Dense(1,
        ...   kernel_initializer=tf.constant_initializer(2.))
        >>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
        >>> layer_b.get_weights()
        [array([[2.],
               [2.],
               [2.]], dtype=float32), array([0.], dtype=float32)]
        >>> layer_b.set_weights(layer_a.get_weights())
        >>> layer_b.get_weights()
        [array([[1.],
               [1.],
               [1.]], dtype=float32), array([0.], dtype=float32)]

        Args:
          weights: a list of NumPy arrays. The number
            of arrays and their shape must match
            number of the dimensions of the weights
            of the layer (i.e. it should match the
            output of `get_weights`).

        Raises:
          ValueError: If the provided weights list does not match the
            layer's specifications.
        """
        params = self.weights

        expected_num_weights = 0
        for param in params:
            if isinstance(param, base_layer_utils.TrackableWeightHandler):
                expected_num_weights += param.num_tensors
            else:
                expected_num_weights += 1

        if expected_num_weights != len(weights):
            raise ValueError(
                'You called `set_weights(weights)` on layer "%s" '
                "with a weight list of length %s, but the layer was "
                "expecting %s weights. Provided weights: %s..."
                % (
                    self.name,
                    len(weights),
                    expected_num_weights,
                    str(weights)[:50],
                )
            )

        weight_index = 0
        weight_value_tuples = []
        for param in params:
            if isinstance(param, base_layer_utils.TrackableWeightHandler):
                num_tensors = param.num_tensors
                tensors = weights[weight_index : weight_index + num_tensors]
                param.set_weights(tensors)
                weight_index += num_tensors
            else:
                weight = weights[weight_index]
                weight_shape = weight.shape if hasattr(weight, "shape") else ()
                ref_shape = param.shape
                if not ref_shape.is_compatible_with(weight_shape):
                    raise ValueError(
                        f"Layer {self.name} weight shape {ref_shape} "
                        "is not compatible with provided weight "
                        f"shape {weight_shape}."
                    )
                weight_value_tuples.append((param, weight))
                weight_index += 1

        backend.batch_set_value(weight_value_tuples)

        # Perform any layer defined finalization of the layer state.
        for layer in self._flattened_modules():
            layer.finalize_state()

    def get_weights(self):
        """Returns the current weights of the layer, as NumPy arrays.

        The weights of a layer represent the state of the layer. This function
        returns both trainable and non-trainable weight values associated with
        this layer as a list of NumPy arrays, which can in turn be used to load
        state into similarly parameterized layers.

        For example, a `Dense` layer returns a list of two values: the kernel
        matrix and the bias vector. These can be used to set the weights of
        another `Dense` layer:

        >>> layer_a = tf.keras.layers.Dense(1,
        ...   kernel_initializer=tf.constant_initializer(1.))
        >>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
        >>> layer_a.get_weights()
        [array([[1.],
               [1.],
               [1.]], dtype=float32), array([0.], dtype=float32)]
        >>> layer_b = tf.keras.layers.Dense(1,
        ...   kernel_initializer=tf.constant_initializer(2.))
        >>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
        >>> layer_b.get_weights()
        [array([[2.],
               [2.],
               [2.]], dtype=float32), array([0.], dtype=float32)]
        >>> layer_b.set_weights(layer_a.get_weights())
        >>> layer_b.get_weights()
        [array([[1.],
               [1.],
               [1.]], dtype=float32), array([0.], dtype=float32)]

        Returns:
            Weights values as a list of NumPy arrays.
        """
        weights = self.weights
        output_weights = []
        for weight in weights:
            if isinstance(weight, base_layer_utils.TrackableWeightHandler):
                output_weights.extend(weight.get_tensors())
            else:
                output_weights.append(weight)
        return backend.batch_get_value(output_weights)

    @doc_controls.do_not_generate_docs
    def finalize_state(self):
        """Finalizes the layers state after updating layer weights.

        This function can be subclassed in a layer and will be called after
        updating a layer weights. It can be overridden to finalize any
        additional layer state after a weight update.

        This function will be called after weights of a layer have been restored
        from a loaded model.
        """
        pass

    @doc_controls.do_not_doc_inheritable
    def get_input_mask_at(self, node_index):
        """Retrieves the input mask tensor(s) of a layer at a given node.

        Args:
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        Returns:
            A mask tensor
            (or list of tensors if the layer has multiple inputs).
        """
        inputs = self.get_input_at(node_index)
        if isinstance(inputs, list):
            return [getattr(x, "_keras_mask", None) for x in inputs]
        else:
            return getattr(inputs, "_keras_mask", None)

    @doc_controls.do_not_doc_inheritable
    def get_output_mask_at(self, node_index):
        """Retrieves the output mask tensor(s) of a layer at a given node.

        Args:
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        Returns:
            A mask tensor
            (or list of tensors if the layer has multiple outputs).
        """
        output = self.get_output_at(node_index)
        if isinstance(output, list):
            return [getattr(x, "_keras_mask", None) for x in output]
        else:
            return getattr(output, "_keras_mask", None)

    @property
    @doc_controls.do_not_doc_inheritable
    def input_mask(self):
        """Retrieves the input mask tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        Returns:
            Input mask tensor (potentially None) or list of input
            mask tensors.

        Raises:
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        inputs = self.input
        if isinstance(inputs, list):
            return [getattr(x, "_keras_mask", None) for x in inputs]
        else:
            return getattr(inputs, "_keras_mask", None)

    @property
    @doc_controls.do_not_doc_inheritable
    def output_mask(self):
        """Retrieves the output mask tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        Returns:
            Output mask tensor (potentially None) or list of output
            mask tensors.

        Raises:
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        output = self.output
        if isinstance(output, list):
            return [getattr(x, "_keras_mask", None) for x in output]
        else:
            return getattr(output, "_keras_mask", None)

    @doc_controls.do_not_doc_inheritable
    def get_input_shape_at(self, node_index):
        """Retrieves the input shape(s) of a layer at a given node.

        Args:
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        Returns:
            A shape tuple
            (or list of shape tuples if the layer has multiple inputs).

        Raises:
          RuntimeError: If called in Eager mode.
        """
        return self._get_node_attribute_at_index(
            node_index, "input_shapes", "input shape"
        )

    @doc_controls.do_not_doc_inheritable
    def get_output_shape_at(self, node_index):
        """Retrieves the output shape(s) of a layer at a given node.

        Args:
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        Returns:
            A shape tuple
            (or list of shape tuples if the layer has multiple outputs).

        Raises:
          RuntimeError: If called in Eager mode.
        """
        return self._get_node_attribute_at_index(
            node_index, "output_shapes", "output shape"
        )

    @doc_controls.do_not_doc_inheritable
    def get_input_at(self, node_index):
        """Retrieves the input tensor(s) of a layer at a given node.

        Args:
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first input node of the layer.

        Returns:
            A tensor (or list of tensors if the layer has multiple inputs).

        Raises:
          RuntimeError: If called in Eager mode.
        """
        return self._get_node_attribute_at_index(node_index, "input_tensors", "input")

    @doc_controls.do_not_doc_inheritable
    def get_output_at(self, node_index):
        """Retrieves the output tensor(s) of a layer at a given node.

        Args:
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first output node of the layer.

        Returns:
            A tensor (or list of tensors if the layer has multiple outputs).

        Raises:
          RuntimeError: If called in Eager mode.
        """
        return self._get_node_attribute_at_index(node_index, "output_tensors", "output")

    @property
    def input(self):
        """Retrieves the input tensor(s) of a layer.

        Only applicable if the layer has exactly one input,
        i.e. if it is connected to one incoming layer.

        Returns:
            Input tensor or list of input tensors.

        Raises:
          RuntimeError: If called in Eager mode.
          AttributeError: If no inbound nodes are found.
        """
        if not self._inbound_nodes:
            raise AttributeError(
                "Layer " + self.name + " is not connected, no input to return."
            )
        return self._get_node_attribute_at_index(0, "input_tensors", "input")

    @property
    def output(self):
        """Retrieves the output tensor(s) of a layer.

        Only applicable if the layer has exactly one output,
        i.e. if it is connected to one incoming layer.

        Returns:
          Output tensor or list of output tensors.

        Raises:
          AttributeError: if the layer is connected to more than one incoming
            layers.
          RuntimeError: if called in Eager mode.
        """
        if not self._inbound_nodes:
            raise AttributeError("Layer " + self.name + " has no inbound nodes.")
        return self._get_node_attribute_at_index(0, "output_tensors", "output")

    @property
    @doc_controls.do_not_doc_inheritable
    def input_shape(self):
        """Retrieves the input shape(s) of a layer.

        Only applicable if the layer has exactly one input,
        i.e. if it is connected to one incoming layer, or if all inputs
        have the same shape.

        Returns:
            Input shape, as an integer shape tuple
            (or list of shape tuples, one tuple per input tensor).

        Raises:
            AttributeError: if the layer has no defined input_shape.
            RuntimeError: if called in Eager mode.
        """
        if not self._inbound_nodes:
            raise AttributeError(
                f'The layer "{self.name}" has never been called '
                "and thus has no defined input shape. Note that the "
                "`input_shape` property is only available for "
                "Functional and Sequential models."
            )
        all_input_shapes = set([str(node.input_shapes) for node in self._inbound_nodes])
        if len(all_input_shapes) == 1:
            return self._inbound_nodes[0].input_shapes
        else:
            raise AttributeError(
                'The layer "' + str(self.name) + '" has multiple inbound nodes, '
                "with different input shapes. Hence "
                'the notion of "input shape" is '
                "ill-defined for the layer. "
                "Use `get_input_shape_at(node_index)` "
                "instead."
            )

    def count_params(self):
        """Count the total number of scalars composing the weights.

        Returns:
            An integer count.

        Raises:
            ValueError: if the layer isn't yet built
              (in which case its weights aren't yet defined).
        """
        if not self.built:
            if getattr(self, "_is_graph_network", False):
                with tf_utils.maybe_init_scope(self):
                    self._maybe_build(self.inputs)
            else:
                raise ValueError(
                    "You tried to call `count_params` "
                    f"on layer {self.name}"
                    ", but the layer isn't built. "
                    "You can build it manually via: "
                    f"`{self.name}.build(batch_input_shape)`."
                )
        return layer_utils.count_params(self.weights)

    @property
    @doc_controls.do_not_doc_inheritable
    def output_shape(self):
        """Retrieves the output shape(s) of a layer.

        Only applicable if the layer has one output,
        or if all outputs have the same shape.

        Returns:
            Output shape, as an integer shape tuple
            (or list of shape tuples, one tuple per output tensor).

        Raises:
            AttributeError: if the layer has no defined output shape.
            RuntimeError: if called in Eager mode.
        """
        if not self._inbound_nodes:
            raise AttributeError(
                f'The layer "{self.name}" has never been called '
                "and thus has no defined output shape."
            )
        all_output_shapes = set(
            [str(node.output_shapes) for node in self._inbound_nodes]
        )
        if len(all_output_shapes) == 1:
            return self._inbound_nodes[0].output_shapes
        else:
            raise AttributeError(
                'The layer "%s"'
                " has multiple inbound nodes, "
                "with different output shapes. Hence "
                'the notion of "output shape" is '
                "ill-defined for the layer. "
                "Use `get_output_shape_at(node_index)` "
                "instead." % self.name
            )

    @property
    def dtype_policy(self):
        """The dtype policy associated with this layer.

        This is an instance of a `tf.keras.mixed_precision.Policy`.
        """
        return self._dtype_policy

    @property
    def compute_dtype(self):
        """The dtype of the layer's computations.

        This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless
        mixed precision is used, this is the same as `Layer.dtype`, the dtype of
        the weights.

        Layers automatically cast their inputs to the compute dtype, which
        causes computations and the output to be in the compute dtype as well.
        This is done by the base Layer class in `Layer.__call__`, so you do not
        have to insert these casts if implementing your own layer.

        Layers often perform certain internal computations in higher precision
        when `compute_dtype` is float16 or bfloat16 for numeric stability. The
        output will still typically be float16 or bfloat16 in such cases.

        Returns:
          The layer's compute dtype.
        """
        return self._dtype_policy.compute_dtype

    @property
    def variable_dtype(self):
        """Alias of `Layer.dtype`, the dtype of the weights."""
        return self.dtype

    @property
    @doc_controls.do_not_doc_inheritable
    def inbound_nodes(self):
        """Return Functional API nodes upstream of this layer."""
        return self._inbound_nodes

    @property
    @doc_controls.do_not_doc_inheritable
    def outbound_nodes(self):
        """Return Functional API nodes downstream of this layer."""
        return self._outbound_nodes

    ############################################################################
    # Methods & attributes below are public aliases of other methods.          #
    ############################################################################

    @property
    @doc_controls.do_not_generate_docs
    def variables(self):
        """Returns the list of all layer variables/weights.

        Alias of `self.weights`.

        Note: This will not track the weights of nested `tf.Modules` that are
        not themselves Keras layers.

        Returns:
          A list of variables.
        """
        return self.weights

    @property
    @doc_controls.do_not_generate_docs
    def trainable_variables(self):
        return self.trainable_weights

    @property
    @doc_controls.do_not_generate_docs
    def non_trainable_variables(self):
        return self.non_trainable_weights

    @doc_controls.do_not_doc_inheritable
    def add_variable(self, *args, **kwargs):
        """Deprecated, do NOT use! Alias for `add_weight`."""
        warnings.warn(
            "`layer.add_variable` is deprecated and "
            "will be removed in a future version. "
            "Please use the `layer.add_weight()` method instead.",
            stacklevel=2,
        )
        return self.add_weight(*args, **kwargs)

    def get_build_config(self):
        if self._build_input_shape is not None:

            def convert_tensorshapes(x):
                if isinstance(x, tf.TensorShape):
                    return tuple(x.as_list())
                return x

            return {
                "input_shape": tf.nest.map_structure(
                    convert_tensorshapes, self._build_input_shape
                )
            }

    def build_from_config(self, config):
        input_shape = config["input_shape"]
        if input_shape is not None:
            self.build(input_shape)

    ############################################################################
    # Methods & attributes below are all private and only used by the framework.
    ############################################################################

    # See tf.Module for the usage of this property.
    # The key for _obj_reference_counts_dict is a Trackable, which could be a
    # variable or layer etc. tf.Module._flatten will fail to flatten the key
    # since it is trying to convert Trackable to a string. This attribute can be
    # ignored even after the fix of nest lib, since the trackable object should
    # already been available as individual attributes.
    # _obj_reference_counts_dict just contains a copy of them.
    _TF_MODULE_IGNORED_PROPERTIES = frozenset(
        itertools.chain(
            ("_obj_reference_counts_dict",),
            tf.Module._TF_MODULE_IGNORED_PROPERTIES,
        )
    )

    # When loading from a SavedModel, Layers typically can be revived into a
    # generic Layer wrapper. Sometimes, however, layers may implement methods
    # that go beyond this wrapper, as in the case of PreprocessingLayers'
    # `adapt` method. When this is the case, layer implementers can override
    # must_restore_from_config to return True; layers with this property must
    # be restored into their actual objects (and will fail if the object is
    # not available to the restoration code).
    _must_restore_from_config = False

    def _get_cell_name(self):
        canonical_name = get_canonical_name_for_symbol(
            self.__class__, api_name="keras", add_prefix_to_v1_names=True
        )
        if canonical_name is not None:
            return f"tf.{canonical_name}"
        return self.__class__.__module__ + "." + self.__class__.__name__

    def _instrument_layer_creation(self):
        self._instrumented_keras_api = False
        self._instrumented_keras_layer_class = False
        self._instrumented_keras_model_class = False
        if not getattr(self, "_disable_keras_instrumentation", False):
            keras_api_gauge.get_cell("layer").set(True)
            self._instrumented_keras_api = True
            if getattr(self, "_is_model_for_instrumentation", False):
                keras_models_gauge.get_cell(self._get_cell_name()).set(True)
                self._instrumented_keras_model_class = True
            else:
                keras_layers_gauge.get_cell(self._get_cell_name()).set(True)
                self._instrumented_keras_layer_class = True
        else:
            # This is a legacy layer that has disabled instrumentation
            # as a native keras object. We still instrument this as
            # legacy usage.
            keras_api_gauge.get_cell("legacy_layer").set(True)

    @doc_controls.for_subclass_implementers
    def _add_trackable(self, trackable_object, trainable):
        """Adds a Trackable object to this layer's state.

        Args:
          trackable_object: The tf.tracking.Trackable object to add.
          trainable: Boolean, whether the variable should be part of the layer's
            "trainable_variables" (e.g. variables, biases) or
            "non_trainable_variables" (e.g. BatchNorm mean and variance).

        Returns:
          The TrackableWeightHandler used to track this object.
        """
        if isinstance(trackable_object, base_layer_utils.TrackableWeightHandler):
            handler = trackable_object
        else:
            handler = base_layer_utils.TrackableWeightHandler(trackable_object)
        if trainable:
            self._trainable_weights.append(handler)
        else:
            self._non_trainable_weights.append(handler)
        return handler

    def _clear_losses(self):
        """Used every step in eager to reset losses."""
        # Set to thread local directly to avoid Layer.__setattr__ overhead.
        if not getattr(
            self, "_self_tracked_trackables", None
        ):  # Fast path for single Layer.
            self._thread_local._eager_losses = []
        else:
            for layer in self._flattened_modules():
                layer._thread_local._eager_losses = []

    def _keras_tensor_symbolic_call(self, inputs, input_masks, args, kwargs):
        if self.dynamic:
            # We will use static shape inference to return symbolic tensors
            # matching the specifications of the layer outputs.
            # Since `self.dynamic` is True, we will never attempt to
            # run the underlying TF graph (which is disconnected).
            # TODO(fchollet): consider py_func as an alternative, which
            # would enable us to run the underlying graph if needed.
            input_signature = tf.nest.map_structure(
                lambda x: tf.TensorSpec(shape=x.shape, dtype=x.dtype), inputs
            )
            output_signature = self.compute_output_signature(input_signature)
            return tf.nest.map_structure(keras_tensor.KerasTensor, output_signature)
        else:
            return self._infer_output_signature(inputs, args, kwargs, input_masks)

    def _infer_output_signature(self, inputs, args, kwargs, input_masks):
        """Call the layer on input KerasTensors, returns output KerasTensors."""

        keras_tensor_inputs = inputs
        call_fn = self.call
        # Wrapping `call` function in autograph to allow for dynamic control
        # flow and control dependencies in call. We are limiting this to
        # subclassed layers as autograph is strictly needed only for
        # subclassed layers and models.
        # tf_convert will respect the value of autograph setting in the
        # enclosing tf.function, if any.
        if base_layer_utils.is_subclassed(
            self
        ) and not base_layer_utils.from_saved_model(self):
            call_fn = tf.__internal__.autograph.tf_convert(
                self.call, tf.__internal__.autograph.control_status_ctx()
            )

        call_fn = traceback_utils.inject_argument_info_in_traceback(
            call_fn,
            object_name=f'layer "{self.name}" (type {self.__class__.__name__})',
        )

        # We enter a scratch graph and build placeholder inputs inside of it
        # that match the input args.
        # We then call the layer inside of the scratch graph to identify the
        # output signatures, then we build KerasTensors corresponding to those
        # outputs.
        scratch_graph = tf.__internal__.FuncGraph(str(self.name) + "_scratch_graph")
        with scratch_graph.as_default():
            inputs = tf.nest.map_structure(
                keras_tensor.keras_tensor_to_placeholder, inputs
            )
            args = tf.nest.map_structure(keras_tensor.keras_tensor_to_placeholder, args)
            kwargs = tf.nest.map_structure(
                keras_tensor.keras_tensor_to_placeholder, kwargs
            )
            input_masks = tf.nest.map_structure(
                keras_tensor.keras_tensor_to_placeholder, input_masks
            )

            with backend.name_scope(self._name_scope()):
                with autocast_variable.enable_auto_cast_variables(
                    self._compute_dtype_object
                ):
                    # Build layer if applicable (if the `build` method has been
                    # overridden).
                    # TODO(kaftan): do we maybe_build here, or have we already
                    # done it?
                    self._maybe_build(inputs)
                    inputs = self._maybe_cast_inputs(inputs)
                    outputs = call_fn(inputs, *args, **kwargs)

                self._handle_activity_regularization(inputs, outputs)
            self._set_mask_metadata(inputs, outputs, input_masks, build_graph=False)
            outputs = tf.nest.map_structure(
                keras_tensor.keras_tensor_from_tensor, outputs
            )

        self._set_save_spec(keras_tensor_inputs, args, kwargs)
        if hasattr(self, "_set_inputs") and not self.inputs:
            # TODO(kaftan): figure out if we need to do this at all
            # Subclassed network: explicitly set metadata normally set by
            # a call to self._set_inputs().
            self._set_inputs(inputs, outputs)
        del scratch_graph
        return outputs

    def _functional_construction_call(self, inputs, args, kwargs, input_list):
        call_context = base_layer_utils.call_context()

        # Accept NumPy and scalar inputs by converting to Tensors.
        if any(isinstance(x, (tf.Tensor, np.ndarray, float, int)) for x in input_list):

            def _convert_non_tensor(x):
                # Don't call `ops.convert_to_tensor` on all `inputs` because
                # `SparseTensors` can't be converted to `Tensor`.
                if isinstance(x, (tf.Tensor, np.ndarray, float, int)):
                    return tf.convert_to_tensor(x)
                return x

            inputs = tf.nest.map_structure(_convert_non_tensor, inputs)
            input_list = tf.nest.flatten(inputs)

        # Handle `mask` propagation from previous layer to current layer. Masks
        # can be propagated explicitly via the `mask` argument, or implicitly
        # via setting the `_keras_mask` attribute on the inputs to a Layer.
        # Masks passed explicitly take priority.
        mask_arg_passed_by_framework = False
        input_masks, mask_is_implicit = self._get_input_masks(
            inputs, input_list, args, kwargs
        )
        if self._expects_mask_arg and mask_is_implicit:
            kwargs["mask"] = input_masks
            mask_arg_passed_by_framework = True

        # If `training` argument is None or not explicitly passed,
        # propagate `training` value from this layer's calling layer.
        training_value = None
        training_arg_passed_by_framework = False
        # Priority 1: `training` was explicitly passed a non-None value.
        if self._call_spec.arg_was_passed("training", args, kwargs):
            training_value = self._call_spec.get_arg_value("training", args, kwargs)
            if not self._expects_training_arg:
                kwargs.pop("training")

        if training_value is None:
            # Priority 2: `training` was passed to a parent layer.
            if call_context.training is not None:
                training_value = call_context.training
            # Priority 3: `learning_phase()` has been set.
            elif backend.global_learning_phase_is_set():
                training_value = backend.learning_phase()
                # Force the training_value to be bool type which matches to the
                # contract for layer/model call args.
                if tf.is_tensor(training_value):
                    training_value = tf.cast(training_value, tf.bool)
                else:
                    training_value = bool(training_value)
            # Priority 4: trace layer with the default training argument
            # specified in the `call` signature (or in inference mode if the
            # `call` signature specifies no non-None default).
            else:
                training_value = self._call_spec.default_training_arg
            # In cases (2), (3), (4) the training argument is passed
            # automatically by the framework, and will not be hard-coded into
            # the model.
            if self._expects_training_arg:
                args, kwargs = self._call_spec.set_arg_value(
                    "training", training_value, args, kwargs
                )
                training_arg_passed_by_framework = True

        with call_context.enter(
            layer=self, inputs=inputs, build_graph=True, training=training_value
        ):
            # Check input assumptions set after layer building, e.g. input
            # shape.
            outputs = self._keras_tensor_symbolic_call(
                inputs, input_masks, args, kwargs
            )

            if outputs is None:
                raise ValueError(
                    "A layer's `call` method should return a "
                    "Tensor or a list of Tensors, not None "
                    "(layer: " + self.name + ")."
                )
            if training_arg_passed_by_framework:
                args, kwargs = self._call_spec.set_arg_value(
                    "training", None, args, kwargs, pop_kwarg_if_none=True
                )
            if mask_arg_passed_by_framework:
                kwargs.pop("mask")
            # Node connectivity does not special-case the first argument.
            outputs = self._set_connectivity_metadata((inputs,) + args, kwargs, outputs)
            return outputs

    def _set_training_mode(self, args, kwargs, call_context):
        training_mode = None
        if self._expects_training_arg:
            # (1) `training` was passed to this `Layer.call`.
            if self._call_spec.arg_was_passed("training", args, kwargs):
                training_mode = self._call_spec.get_arg_value("training", args, kwargs)
            # If no `training` arg was passed, or `None` was explicitly passed,
            # the framework will make a decision about the training mode is.
            if training_mode is None:
                call_ctx_training = call_context.training
                # (2) `training` mode is inferred from an outer `Layer.call`.
                if call_ctx_training is not None:
                    training_mode = call_ctx_training
                # (3) User set `tf.keras.backend.set_learning_phase`.
                elif backend.global_learning_phase_is_set():
                    training_mode = backend.learning_phase()
                    # Ensure value is a `bool` or `tf.bool`.
                    if isinstance(training_mode, bool):
                        pass
                    elif tf.is_tensor(training_mode):
                        training_mode = tf.cast(training_mode, tf.bool)
                    else:
                        training_mode = bool(training_mode)
                # (4) We default to using `call`'s default value for `training`,
                # or treating the layer as if it is in inference if no non-None
                # default is specified in the `call` signature.
                else:
                    training_mode = self._call_spec.default_training_arg

                # For case (2), (3), (4) `training` arg is passed by framework.
                args, kwargs = self._call_spec.set_arg_value(
                    "training", training_mode, args, kwargs
                )
        else:
            if "training" in kwargs:
                # `training` was passed to this `Layer` but is not needed for
                # `Layer.call`. It will set the default mode for inner
                # `Layer.call`s.
                training_mode = kwargs.pop("training")
            else:
                # Grab the current `training` mode from any outer `Layer.call`.
                training_mode = call_context.training

        return args, kwargs, training_mode

    def _autographed_call(self):
        # Wrapping `call` function in autograph to allow for dynamic control
        # flow and control dependencies in call. We are limiting this to
        # subclassed layers as autograph is strictly needed only for
        # subclassed layers and models.
        # tf_convert will respect the value of autograph setting in the
        # enclosing tf.function, if any.
        if base_layer_utils.is_subclassed(
            self
        ) and not base_layer_utils.from_saved_model(self):
            return tf.__internal__.autograph.tf_convert(
                self.call, tf.__internal__.autograph.control_status_ctx()
            )
        else:
            return self.call

    @property
    def _inbound_nodes(self):
        return self._inbound_nodes_value

    @_inbound_nodes.setter
    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def _inbound_nodes(self, value):
        self._inbound_nodes_value = value

    @property
    def _outbound_nodes(self):
        return self._outbound_nodes_value

    @_outbound_nodes.setter
    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def _outbound_nodes(self, value):
        self._outbound_nodes_value = value

    def _set_dtype_policy(self, dtype):
        """Sets self._dtype_policy."""
        if isinstance(dtype, policy.Policy):
            self._dtype_policy = dtype
        elif isinstance(dtype, dict):
            self._dtype_policy = policy.deserialize(dtype)
        elif isinstance(dtype, str) and dtype in (
            "mixed_float16",
            "mixed_bfloat16",
        ):
            # The isinstance check is required since np.dtype raises an error if
            # compared to a non-dtype string.
            self._dtype_policy = policy.Policy(dtype)
        elif dtype:
            self._dtype_policy = policy.Policy(tf.as_dtype(dtype).name)
        else:
            self._dtype_policy = policy.global_policy()
        if (
            self._dtype_policy.name == "mixed_float16"
            and not loss_scale_optimizer.strategy_supports_loss_scaling()
        ):
            # Although only loss scaling doesn't support certain strategies, to
            # avoid confusion, we disallow the 'mixed_float16' policy with
            # unsupported strategies. This is because 'mixed_float16' requires
            # loss scaling for numeric stability.
            strategy = tf.distribute.get_strategy()
            raise ValueError(
                "Mixed precision is not supported with the "
                "tf.distribute.Strategy: %s. Either stop using mixed "
                'precision by removing the use of the "%s" policy or '
                "use a different Strategy, e.g. a MirroredStrategy."
                % (strategy.__class__.__name__, self._dtype_policy.name)
            )

        # Performance optimization: cache the compute dtype as a Dtype object or
        # None, so that str to Dtype conversion doesn't happen in
        # Layer.__call__.
        # TODO(b/157486353): Investigate returning DTypes in Policy.
        if self._dtype_policy.compute_dtype:
            self._compute_dtype_object = tf.as_dtype(self._dtype_policy.compute_dtype)
        else:
            self._compute_dtype_object = None

    @property
    def _compute_dtype(self):
        """Deprecated alias of `compute_dtype`."""
        return self._dtype_policy.compute_dtype

    def _maybe_cast_inputs(self, inputs, input_list=None):
        """Maybe casts the inputs to the compute dtype.

        If self._compute_dtype is floating-point, and self_autocast is True,
        floating-point inputs are casted to self._compute_dtype.

        Args:
          inputs: Input tensor, or structure of input tensors.
          input_list: Flat list of input tensors.

        Returns:
          `inputs`, but tensors may have been casted to self._compute_dtype
        """
        if not input_list:
            input_list = tf.nest.flatten(inputs)

        compute_dtype_object = self._compute_dtype_object
        should_autocast = (
            self._autocast and compute_dtype_object and compute_dtype_object.is_floating
        )

        if should_autocast and any(map(self._should_cast_single_input, input_list)):
            # Only perform expensive `nest` operation when needed.
            return tf.nest.map_structure(self._cast_single_input, inputs)
        else:
            return inputs

    def _should_cast_single_input(self, x):
        if isinstance(x, _AUTOCAST_TYPES):
            return (
                self._compute_dtype_object
                and x.dtype != self._compute_dtype_object
                and x.dtype.is_floating
            )
        return False

    def _cast_single_input(self, x):
        """Cast a single Tensor or TensorSpec to the compute dtype."""
        if self._should_cast_single_input(x):
            return tf.cast(x, self._compute_dtype_object)
        else:
            return x

    # _dtype used to be an attribute set in the constructor. We still expose it
    # because some clients still use it.
    # TODO(reedwm): Deprecate, then remove the _dtype property.
    @property
    def _dtype(self):
        # This is equivalent to returning self.dtype . We do not return
        # self.dtype as it would cause infinite recursion in a few subclasses,
        # which override "dtype" to return self._dtype.
        return self._dtype_policy.variable_dtype

    @_dtype.setter
    def _dtype(self, value):
        value = tf.as_dtype(value).name
        self._set_dtype_policy(policy.Policy(value))

    def _name_scope(self):
        if not tf.__internal__.tf2.enabled():
            return self.name
        name_scope = self.name
        current_name_scope = tf.__internal__.get_name_scope()
        if current_name_scope:
            name_scope = current_name_scope + "/" + name_scope
        if name_scope:
            # Note that the trailing `/` prevents autogenerated
            # numerical suffixes to get appended. It will also fully reset
            # nested name scope (i.e. the outer name scope has no effect).
            name_scope += "/"
        return name_scope

    def _init_set_name(self, name, zero_based=True):
        if name is None:
            self._name = backend.unique_object_name(
                generic_utils.to_snake_case(self.__class__.__name__),
                zero_based=zero_based,
            )
        elif isinstance(name, str):
            backend.observe_object_name(name)
            self._name = name
        else:
            raise TypeError(f"Expected `name` argument to be a string, but got: {name}")

    def _get_existing_metric(self, name=None):
        match = [m for m in self._metrics if m.name == name]
        if not match:
            return
        if len(match) > 1:
            raise ValueError(
                "Please provide different names for the metrics you have "
                'added. We found {} metrics with the name: "{}"'.format(
                    len(match), name
                )
            )
        return match[0]

    def _handle_weight_regularization(self, name, variable, regularizer):
        """Create lambdas which compute regularization losses."""

        def _loss_for_variable(v):
            """Creates a regularization loss `Tensor` for variable `v`."""
            with backend.name_scope(name + "/Regularizer"):
                regularization = regularizer(v)
            return regularization

        if base_layer_utils.is_split_variable(variable):
            for v in variable:
                self.add_loss(functools.partial(_loss_for_variable, v))
        elif isinstance(variable, lazy_variable.LazyInitVariable):
            self._captured_weight_regularizer.append((name, variable, regularizer))
        else:
            self.add_loss(functools.partial(_loss_for_variable, variable))

    def _handle_activity_regularization(self, inputs, outputs):
        # Apply activity regularization.
        # Note that it should be applied every time the layer creates a new
        # output, since it is output-specific.
        if self._activity_regularizer:
            output_list = tf.nest.flatten(outputs)
            with backend.name_scope("ActivityRegularizer"):
                for output in output_list:
                    activity_loss = tf.convert_to_tensor(
                        self._activity_regularizer(output)
                    )
                    batch_size = tf.cast(tf.shape(output)[0], activity_loss.dtype)
                    # Make activity regularization strength batch-agnostic.
                    mean_activity_loss = activity_loss / batch_size
                    self.add_loss(mean_activity_loss)

    def _set_mask_metadata(self, inputs, outputs, previous_mask, build_graph):
        # Many `Layer`s don't need to call `compute_mask`.
        # This method is optimized to do as little work as needed for the common
        # case.
        if not self._supports_masking:
            return

        flat_outputs = tf.nest.flatten(outputs)

        mask_already_computed = getattr(
            self, "_compute_output_and_mask_jointly", False
        ) or all(getattr(x, "_keras_mask", None) is not None for x in flat_outputs)
        if mask_already_computed:
            if build_graph:
                self._set_mask_keras_history_checked(flat_outputs)
            return

        output_masks = self.compute_mask(inputs, previous_mask)
        if output_masks is None:
            return

        flat_masks = tf.nest.flatten(output_masks)
        for tensor, mask in zip(flat_outputs, flat_masks):
            try:
                tensor._keras_mask = mask
            except AttributeError:
                # C Type such as np.ndarray.
                pass

        if build_graph:
            self._set_mask_keras_history_checked(flat_outputs)

    def _set_mask_keras_history_checked(self, flat_outputs):
        for output in flat_outputs:
            if getattr(output, "_keras_mask", None) is not None:
                # Do not track masks for `TensorFlowOpLayer` construction.
                output._keras_mask._keras_history_checked = True

    def _get_input_masks(self, inputs, input_list, args, kwargs):
        if not self._supports_masking and not self._expects_mask_arg:
            # Input masks only need to be retrieved if they are needed for
            # `call` or `compute_mask`.
            input_masks = None
            implicit_mask = False
        elif self._call_spec.arg_was_passed("mask", args, kwargs):
            input_masks = self._call_spec.get_arg_value("mask", args, kwargs)
            implicit_mask = False
        else:
            input_masks = [getattr(t, "_keras_mask", None) for t in input_list]
            if all(mask is None for mask in input_masks):
                input_masks = None
                implicit_mask = False
            else:
                # Only do expensive `nest` op when masking is actually being
                # used.
                input_masks = tf.nest.pack_sequence_as(inputs, input_masks)
                implicit_mask = True
        return input_masks, implicit_mask

    def _set_connectivity_metadata(self, args, kwargs, outputs):
        # If the layer returns tensors from its inputs unmodified,
        # we copy them to avoid loss of KerasHistory metadata.
        flat_outputs = tf.nest.flatten(outputs)
        flat_inputs = tf.nest.flatten((args, kwargs))
        input_ids_set = {id(i) for i in flat_inputs}
        outputs_copy = []
        for x in flat_outputs:
            if id(x) in input_ids_set:
                with backend.name_scope(self.name):
                    x = tf.identity(x)
            outputs_copy.append(x)
        outputs = tf.nest.pack_sequence_as(outputs, outputs_copy)

        # Create node, Node wires itself to inbound and outbound layers.  The
        # Node constructor actually updates this layer's self._inbound_nodes,
        # sets _keras_history on the outputs, and adds itself to the
        # `_outbound_nodes` of the layers that produced the inputs to this layer
        # call.
        node_module.Node(self, call_args=args, call_kwargs=kwargs, outputs=outputs)
        return outputs

    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        """Private utility to retrieves an attribute (e.g. inputs) from a node.

        This is used to implement the methods:
            - get_input_shape_at
            - get_output_shape_at
            - get_input_at
            etc...

        Args:
            node_index: Integer index of the node from which
                to retrieve the attribute.
            attr: Exact node attribute name.
            attr_name: Human-readable attribute name, for error messages.

        Returns:
            The layer's attribute `attr` at the node of index `node_index`.

        Raises:
            RuntimeError: If the layer has no inbound nodes, or if called in
                Eager mode.
            ValueError: If the index provided does not match any node.
        """
        if not self._inbound_nodes:
            raise RuntimeError(
                f"The layer {self.name} has never been called "
                f"and thus has no defined {attr_name}."
            )
        if not len(self._inbound_nodes) > node_index:
            raise ValueError(
                f"Asked to get {attr_name} at node "
                f"{node_index}, but the layer has only "
                f"{len(self._inbound_nodes)} inbound nodes."
            )
        values = getattr(self._inbound_nodes[node_index], attr)
        if isinstance(values, list) and len(values) == 1:
            return values[0]
        else:
            return values

    def _maybe_build(self, inputs):
        # Check input assumptions set before layer building, e.g. input rank.
        if not self.built:
            input_spec.assert_input_compatibility(self.input_spec, inputs, self.name)
            input_list = tf.nest.flatten(inputs)
            if input_list and self._dtype_policy.compute_dtype is None:
                try:
                    dtype = input_list[0].dtype.base_dtype.name
                except AttributeError:
                    pass
                else:
                    self._set_dtype_policy(policy.Policy(dtype))
            input_shapes = None
            # Converts Tensors / CompositeTensors to TensorShapes.
            if any(hasattr(x, "shape") for x in input_list):
                input_shapes = tf_utils.get_shapes(inputs)
            else:
                # Converts input shape to TensorShapes.
                try:
                    input_shapes = tf_utils.convert_shapes(inputs, to_tuples=False)
                except ValueError:
                    pass
            # Only call `build` if the user has manually overridden the build
            # method.
            if not hasattr(self.build, "_is_default"):
                # Any setup work performed only once should happen in an
                # `init_scope` to avoid creating symbolic Tensors that will
                # later pollute any eager operations.
                with tf_utils.maybe_init_scope(self):
                    self.build(input_shapes)
            # We must set also ensure that the layer is marked as built, and the
            # build shape is stored since user defined build functions may not
            # be calling `super.build()`
            Layer.build(self, input_shapes)

        # Optionally load weight values specified at layer instantiation.
        if self._initial_weights is not None:
            with tf.init_scope():
                # Using `init_scope` since we want variable assignment in
                # `set_weights` to be treated like variable initialization.
                self.set_weights(self._initial_weights)
            self._initial_weights = None

    def _get_trainable_state(self):
        """Get the `trainable` state of each sublayer.

        Returns:
          A dict mapping all sublayers to their `trainable` value.
        """
        trainable_state = weakref.WeakKeyDictionary()
        for layer in self._flattened_modules():
            trainable_state[layer] = layer.trainable
        return trainable_state

    def _set_trainable_state(self, trainable_state):
        """Set `trainable` state for each sublayer."""
        for layer in self._flattened_modules():
            if layer in trainable_state:
                layer.trainable = trainable_state[layer]

    @property
    def _obj_reference_counts(self):
        """A dict counting the number of attributes referencing an object."""
        self._maybe_create_attribute(
            "_obj_reference_counts_dict",
            object_identity.ObjectIdentityDictionary(),
        )
        return self._obj_reference_counts_dict

    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def _maybe_create_attribute(self, name, default_value):
        """Create attribute (with the default value) if it hasn't been created.

        This is useful for fields that is used for tracking purpose,
        _trainable_weights, or _layers. Note that user could create a layer
        subclass and assign an internal field before invoking the
        Layer.__init__(), the __setattr__() need to create the tracking fields
        and __init__() need to not override them.

        Args:
          name: String, the name of the attribute.
          default_value: Object, the default value of the attribute.
        """
        if not hasattr(self, name):
            self.__setattr__(name, default_value)

    def __delattr__(self, name):
        # For any super.__delattr__() call, we will directly use the
        # implementation in Trackable and skip the behavior in AutoTrackable.
        # The Layer was originally use Trackable as base class, the change of
        # using Module as base class forced us to have AutoTrackable in the
        # class hierarchy.
        #
        # TODO(b/180760306) Keeping the status quo of skipping _delattr__ and
        # __setattr__ in AutoTrackable may be unsustainable.
        existing_value = getattr(self, name, None)

        # If this value is replacing an existing object assigned to an
        # attribute, we should clean it out to avoid leaking memory. First we
        # check if there are other attributes referencing it.
        reference_counts = self._obj_reference_counts
        if existing_value not in reference_counts:
            super(tf.__internal__.tracking.AutoTrackable, self).__delattr__(name)
            return

        reference_count = reference_counts[existing_value]
        if reference_count > 1:
            # There are other remaining references. We can't remove this object
            # from _layers etc.
            reference_counts[existing_value] = reference_count - 1
            super(tf.__internal__.tracking.AutoTrackable, self).__delattr__(name)
            return
        else:
            # This is the last remaining reference.
            del reference_counts[existing_value]

        super(tf.__internal__.tracking.AutoTrackable, self).__delattr__(name)

        if isinstance(existing_value, Layer) or base_layer_utils.has_weights(
            existing_value
        ):
            super(tf.__internal__.tracking.AutoTrackable, self).__setattr__(
                "_self_tracked_trackables",
                [l for l in self._self_tracked_trackables if l is not existing_value],
            )
        if isinstance(existing_value, tf.Variable):
            super(tf.__internal__.tracking.AutoTrackable, self).__setattr__(
                "_trainable_weights",
                [w for w in self._trainable_weights if w is not existing_value],
            )
            super(tf.__internal__.tracking.AutoTrackable, self).__setattr__(
                "_non_trainable_weights",
                [w for w in self._non_trainable_weights if w is not existing_value],
            )

    def __setattr__(self, name, value):
        if (
            name == "_self_setattr_tracking"
            or not getattr(self, "_self_setattr_tracking", True)
            # Exclude @property.setters from tracking
            or hasattr(self.__class__, name)
        ):
            try:
                super(tf.__internal__.tracking.AutoTrackable, self).__setattr__(
                    name, value
                )
            except AttributeError:
                raise AttributeError(
                    (
                        'Can\'t set the attribute "{}", likely because it '
                        "conflicts with an existing read-only @property of the "
                        "object. Please choose a different name."
                    ).format(name)
                )
            return

        # Wraps data structures in `Trackable`, unwraps `NoDependency` objects.
        value = tf.__internal__.tracking.sticky_attribute_assignment(
            trackable=self, value=value, name=name
        )

        reference_counts = self._obj_reference_counts
        reference_counts[value] = reference_counts.get(value, 0) + 1

        # When replacing an existing tf.Variable with a new one, we want to
        # check its existing position in the
        # self._trainable/non_trainable_variable, so that we can put it back to
        # the original position.
        if isinstance(value, tf.Variable) and isinstance(
            getattr(self, name, None), tf.Variable
        ):
            existing_variable = getattr(self, name)

            def _get_variable_from_list(var_list, var):
                # helper function to get the tf.variable from the list
                # the default list.index() use == for comparison, which will
                # cause issue for eager tensor.
                for i in range(len(var_list)):
                    if var_list[i] is var:
                        return i
                return None

            if existing_variable.trainable:
                self._maybe_create_attribute("_trainable_weights", [])
                position = _get_variable_from_list(
                    self._trainable_weights, existing_variable
                )
            else:
                self._maybe_create_attribute("_non_trainable_variable", [])
                position = _get_variable_from_list(
                    self._non_trainable_variable, existing_variable
                )
        else:
            position = None

        # Clean out the old attribute, which clears _layers and
        # _trainable_weights if necessary.
        try:
            self.__delattr__(name)
        except AttributeError:
            pass

        # Keep track of metric instance created in subclassed layer.
        for val in tf.nest.flatten(value):
            if isinstance(val, metrics_mod.Metric) and hasattr(self, "_metrics"):
                self._metrics.append(val)

        # Append value to self._self_tracked_trackables if relevant
        if getattr(self, "_auto_track_sub_layers", True) and (
            isinstance(value, tf.Module) or base_layer_utils.has_weights(value)
        ):
            self._maybe_create_attribute("_self_tracked_trackables", [])
            # We need to check object identity to avoid de-duplicating empty
            # container types which compare equal.
            if not any((layer is value for layer in self._self_tracked_trackables)):
                self._self_tracked_trackables.append(value)
                if hasattr(value, "_use_resource_variables"):
                    # Legacy layers (V1 tf.layers) must always use
                    # resource variables.
                    value._use_resource_variables = True

        # Append value to list of trainable / non-trainable weights if relevant
        # TODO(b/125122625): This won't pick up on any variables added to a
        # list/dict after creation.
        self._track_variables(value, position=position)

        # TODO(b/180760306) Skip the auto trackable from tf.Module to keep
        # status quo. See the comment at __delattr__.
        super(tf.__internal__.tracking.AutoTrackable, self).__setattr__(name, value)

    def _update_trackables(self):
        """Track variables added to lists/dicts after creation"""
        for trackable_obj in self._self_tracked_trackables:
            if isinstance(
                trackable_obj, tf.__internal__.tracking.TrackableDataStructure
            ):
                self._track_variables(trackable_obj)

    def _track_variables(self, value, position=None):
        """Tracks `Variable`s including `Variable`s in `CompositeTensor`s."""
        for val in tf.nest.flatten(value):
            if isinstance(val, tf.Variable):
                self._track_variable(val, position=position)
            elif tf_utils.is_extension_type(val):
                # Manually expand extension types to track resource variables.
                nested_vals = tf_utils.type_spec_from_value(val)._to_components(val)
                self._track_variables(nested_vals, position=position)

    def _track_variable(self, val, position=None):
        """Tracks the given `tf.Variable`."""
        # Users may add extra weights/variables simply by assigning them to
        # attributes (invalid for graph networks)
        self._maybe_create_attribute("_trainable_weights", [])
        self._maybe_create_attribute("_non_trainable_weights", [])
        if val.trainable:
            if any(val is w for w in self._trainable_weights):
                return
            if position is None:
                self._trainable_weights.append(val)
            else:
                self._trainable_weights.insert(position, val)
        else:
            if any(val is w for w in self._non_trainable_weights):
                return
            if position is None:
                self._non_trainable_weights.append(val)
            else:
                self._non_trainable_weights.insert(position, val)
        backend.track_variable(val)

    def _gather_children_attribute(self, attribute):
        assert attribute in {
            "variables",
            "trainable_variables",
            "non_trainable_variables",
        }
        if hasattr(self, "_self_tracked_trackables"):
            nested_layers = self._flatten_modules(include_self=False, recursive=False)
            return list(
                itertools.chain.from_iterable(
                    getattr(layer, attribute) for layer in nested_layers
                )
            )
        return []

    def _flattened_modules(self, recursive=True, include_self=True):
        for m in self._flatten_modules(recursive=recursive, include_self=include_self):
            if isinstance(m, Layer):
                yield m

    def _flatten_modules(self, recursive=True, include_self=True):
        """Flattens `tf.Module` instances (excluding `Metrics`).

        Args:
          recursive: Whether to recursively flatten through submodules.
          include_self: Whether to include this `Layer` instance.

        Yields:
          `tf.Module` instance tracked by this `Layer`.
        """
        if include_self:
            yield self

        # Only instantiate set and deque if needed.
        trackables = getattr(self, "_self_tracked_trackables", None)
        if trackables:
            seen_object_ids = set()
            deque = collections.deque(trackables)
            while deque:
                trackable_obj = deque.popleft()
                trackable_id = id(trackable_obj)
                if trackable_id in seen_object_ids:
                    continue
                seen_object_ids.add(trackable_id)

                # Metrics are not considered part of the Layer's topology.
                if isinstance(trackable_obj, tf.Module) and not isinstance(
                    trackable_obj, metrics_mod.Metric
                ):
                    yield trackable_obj
                    # Introspect recursively through sublayers.
                    if recursive:
                        subtrackables = getattr(
                            trackable_obj, "_self_tracked_trackables", None
                        )
                        if subtrackables:
                            deque.extendleft(reversed(subtrackables))
                elif isinstance(
                    trackable_obj,
                    tf.__internal__.tracking.TrackableDataStructure,
                ):
                    # Data structures are introspected even with
                    # `recursive=False`.
                    tracked_values = trackable_obj._values
                    if tracked_values:
                        deque.extendleft(reversed(tracked_values))

    # This is a hack so that the is_layer (within
    # training/trackable/layer_utils.py) check doesn't get the weights attr.
    # TODO(b/110718070): Remove when fixed.
    def _is_layer(self):
        return True

    def _init_call_fn_args(self, expects_training_arg=None):
        self._call_spec = layer_utils.CallFunctionSpec(
            tf_inspect.getfullargspec(self.call)
        )
        if expects_training_arg is not None:
            self._call_spec.expects_training_arg = expects_training_arg

    @property
    def _expects_training_arg(self):
        """Whether the call function uses 'training' as a parameter."""
        return self._call_spec.expects_training_arg

    @property
    def _expects_mask_arg(self):
        return self._call_spec.expects_mask_arg

    @property
    def _eager_losses(self):
        # A list of loss values containing activity regularizers and losses
        # manually added through `add_loss` during eager execution. It is
        # cleared after every batch. Because we plan on eventually allowing a
        # same model instance to be trained in eager mode or graph mode
        # alternatively, we need to keep track of eager losses and symbolic
        # losses via separate attributes.
        if not hasattr(self._thread_local, "_eager_losses"):
            self._thread_local._eager_losses = []
        return self._thread_local._eager_losses

    @_eager_losses.setter
    def _eager_losses(self, losses):
        self._thread_local._eager_losses = losses

    def _dedup_weights(self, weights):
        """Dedupe weights while maintaining order as much as possible."""
        output, seen_ids = [], set()
        for w in weights:
            if id(w) not in seen_ids:
                output.append(w)
                # Track the Variable's identity to avoid __eq__ issues.
                seen_ids.add(id(w))
        return output

    # SavedModel properties. Please see keras/saving/saved_model for details.

    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def _set_save_spec(self, inputs, args=None, kwargs=None):
        """Defines the save spec so that serialization can trace layer calls.

        The TensorSpecs of the call function `inputs`, `args`, and `kwargs` are
        saved into a tuple of `([inputs] + args, kwargs)`.

        Args:
          inputs: possibly nested inputs passed into the call function.
          args: a list of positional arguments passed into call.
          kwargs: a dictionary of keyword arguments passed into call.
        """
        if self._saved_model_inputs_spec is not None:
            return  # Already set.

        inputs_spec = tf.nest.map_structure(tf_utils.get_tensor_spec, inputs)
        args_spec = tf.nest.map_structure(tf_utils.get_tensor_spec, args or [])
        kwargs_spec = {}
        # Filter out non-tensor arguments from kwargs.
        for key, kwarg in kwargs.items():
            flat_kwarg = tf.nest.flatten(kwarg)
            flat_specs = [tf_utils.get_tensor_spec(x) for x in flat_kwarg]
            if any(s is None for s in flat_specs):
                continue
            kwargs_spec[key] = tf.nest.pack_sequence_as(kwarg, flat_specs)

        self._saved_model_inputs_spec = inputs_spec
        self._saved_model_arg_spec = (
            [inputs_spec] + list(args_spec),
            kwargs_spec,
        )

    def _get_save_spec(self, dynamic_batch=True, inputs_only=True):
        if self._saved_model_inputs_spec is None:
            return None

        spec = tf.nest.map_structure(
            lambda t: tf_utils.get_tensor_spec(t, dynamic_batch=dynamic_batch),
            self._saved_model_arg_spec,
        )
        return spec[0][0] if inputs_only else spec

    @property
    def _trackable_saved_model_saver(self):
        return layer_serialization.LayerSavedModelSaver(self)

    @property
    def _object_identifier(self):
        return self._trackable_saved_model_saver.object_identifier

    @property
    def _tracking_metadata(self):
        """Info about this layer to be saved into the SavedModel."""
        return self._trackable_saved_model_saver.tracking_metadata

    def _trackable_children(self, save_type="checkpoint", **kwargs):
        if save_type == "savedmodel":
            cache = kwargs["cache"]
            # TODO(b/213628533): This must be called before super() to ensure
            # that any input shape changes are applied before getting the config
            # of the model.
            children = self._trackable_saved_model_saver.trackable_children(cache)
        else:
            children = {}
        children.update(super()._trackable_children(save_type, **kwargs))
        return children

    @property
    def _use_input_spec_as_call_signature(self):
        # Whether input spec can be used as the call signature when tracing the
        # Layer for SavedModel. By default, this is set to `True` for layers
        # exported from the Keras library, because the layers more rigidly
        # define the `input_specs` property (many custom layers only set the
        # `ndims`)
        return get_canonical_name_for_symbol(type(self), api_name="keras") is not None

    def __getstate__(self):
        # Override to support `copy.deepcopy` and pickling.
        # Thread-local objects cannot be copied in Python 3, so pop these.
        # Thread-local objects are used to cache losses in MirroredStrategy, and
        # so shouldn't be copied.
        state = self.__dict__.copy()
        state.pop("_thread_local", None)
        state.pop("_metrics_lock", None)
        return state

    def __setstate__(self, state):
        state["_thread_local"] = threading.local()
        state["_metrics_lock"] = threading.Lock()
        # Bypass Trackable logic as `__dict__` already contains this info.
        object.__setattr__(self, "__dict__", state)

    def _save_own_variables(self, store):
        """Experimental method for saving the state of this layer object."""
        all_vars = self._trainable_weights + self._non_trainable_weights
        for i, v in enumerate(all_vars):
            store[f"{i}"] = v.numpy()

    def _load_own_variables(self, store):
        """Experimental method for loading the state of this layer object."""
        self._update_trackables()
        all_vars = self._trainable_weights + self._non_trainable_weights
        if len(store.keys()) != len(all_vars):
            raise ValueError(
                f"Layer '{self.name}' expected {len(all_vars)} variables, "
                "but received "
                f"{len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )
        for i, v in enumerate(all_vars):
            # TODO(rchao): check shapes and raise errors.
            v.assign(store[f"{i}"])


class TensorFlowOpLayer(Layer):
    """Wraps a TensorFlow Operation in a Layer.

    This class is used internally by the Functional API. When a user
    uses a raw TensorFlow Operation on symbolic tensors originating
    from an `Input` Layer, the resultant operation will be wrapped
    with this Layer object in order to make the operation compatible
    with the Keras API.

    This Layer will create a new, identical operation (except for inputs
    and outputs) every time it is called. If `run_eagerly` is `True`,
    the op creation and calculation will happen inside an Eager function.

    Instances of this Layer are created when `autolambda` is called, which
    is whenever a Layer's `__call__` encounters symbolic inputs that do
    not have Keras metadata, or when a Network's `__init__` encounters
    outputs that do not have Keras metadata.

    Attributes:
      node_def: String, the serialized NodeDef of the Op this layer will wrap.
      name: String, the name of the Layer.
      constants: Dict of NumPy arrays, the values of any Tensors needed for this
        Operation that do not originate from a Keras `Input` Layer. Since all
        placeholders must come from Keras `Input` Layers, these Tensors must be
        treated as constant in the Functional API.
      trainable: Bool, whether this Layer is trainable. Currently Variables are
        not supported, and so this parameter has no effect.
      dtype: The default dtype of this Layer. Inherited from `Layer` and has no
        effect on this class, however is used in `get_config`.
    """

    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def __init__(self, node_def, name, constants=None, trainable=True, dtype=None):
        # Pass autocast=False, as if inputs are cast, input types might not
        # match Operation type.
        super(TensorFlowOpLayer, self).__init__(
            name=_TF_OP_LAYER_NAME_PREFIX + name,
            trainable=trainable,
            dtype=dtype,
            autocast=False,
        )
        if isinstance(node_def, dict):
            self.node_def = json_format.ParseDict(node_def, tf.compat.v1.NodeDef())
        else:
            if not isinstance(node_def, bytes):
                node_def = node_def.encode("utf-8")
            self.node_def = tf.compat.v1.NodeDef.FromString(node_def)
        # JSON serialization stringifies keys which are integer input indices.
        self.constants = (
            {int(index): constant for index, constant in constants.items()}
            if constants is not None
            else {}
        )
        # Layer uses original op unless it is called on new inputs.
        # This means `built` is not set in `__call__`.
        self.built = True

        # Do not individually trace TensorflowOpLayers in the SavedModel.
        self._must_restore_from_config = True

    def call(self, inputs):
        if tf.executing_eagerly():
            return self._defun_call(inputs)
        return self._make_op(inputs)

    def _make_node_def(self, graph):
        node_def = tf.compat.v1.NodeDef()
        node_def.CopyFrom(self.node_def)
        # Used in TPUReplicateContext to indicate whether this node has been
        # cloned and to not add TPU attributes.
        node_def.attr["_cloned"].b = True
        node_def.name = graph.unique_name(node_def.name)
        return node_def

    def _make_op(self, inputs):
        inputs = tf.nest.flatten(inputs)
        graph = inputs[0].graph
        node_def = self._make_node_def(graph)
        with graph.as_default():
            for index, constant in self.constants.items():
                # Recreate constant in graph to add distribution context.
                value = tf.get_static_value(constant)
                if value is not None:
                    constant = tf.constant(value, name=node_def.input[index])
                inputs.insert(index, constant)
            # TODO(b/183990973): We should drop or consolidate these private api
            # calls for adding an op to the graph and recording its gradient.
            c_op = tf.__internal__.create_c_op(
                graph, node_def, inputs, control_inputs=[]
            )
            op = graph._create_op_from_tf_operation(c_op)
            op._control_flow_post_processing()

            # Record the gradient because custom-made ops don't go through the
            # code-gen'd eager call path
            op_type = tf.compat.as_str(op.op_def.name)
            attr_names = [tf.compat.as_str(attr.name) for attr in op.op_def.attr]
            attrs = []
            for attr_name in attr_names:
                attrs.append(attr_name)
                attrs.append(op.get_attr(attr_name))
            attrs = tuple(attrs)
            tf.__internal__.record_gradient(op_type, op.inputs, attrs, op.outputs)

            if len(op.outputs) == 1:
                return op.outputs[0]
            return op.outputs

    @tf.function
    def _defun_call(self, inputs):
        """Wraps op creation method in an Eager function for `run_eagerly`."""
        return self._make_op(inputs)

    def get_config(self):
        config = super(TensorFlowOpLayer, self).get_config()
        config.update(
            {
                # `__init__` prefixes the name. Revert to the constructor
                # argument.
                "name": config["name"][len(_TF_OP_LAYER_NAME_PREFIX) :],
                "node_def": json_format.MessageToDict(self.node_def),
                "constants": {
                    i: backend.get_value(c) for i, c in self.constants.items()
                },
            }
        )
        return config


class AddLoss(Layer):
    """Adds its inputs as a loss.

    Attributes:
      unconditional: Whether or not the loss should be conditioned on the
        inputs.
    """

    def __init__(self, unconditional, **kwargs):
        # Pass autocast=False, as there is no reason to cast loss to a different
        # dtype.
        kwargs["autocast"] = False
        super(AddLoss, self).__init__(**kwargs)
        self.unconditional = unconditional

    def call(self, inputs):
        self.add_loss(inputs, inputs=(not self.unconditional))
        return inputs

    def get_config(self):
        config = super(AddLoss, self).get_config()
        config.update({"unconditional": self.unconditional})
        return config


class AddMetric(Layer):
    """Adds its inputs as a metric.

    Attributes:
      aggregation: 'mean' or None. How the inputs should be aggregated.
      metric_name: The name to use for this metric.
    """

    def __init__(self, aggregation=None, metric_name=None, **kwargs):
        super(AddMetric, self).__init__(**kwargs)
        self.aggregation = aggregation
        self.metric_name = metric_name

    def call(self, inputs):
        self.add_metric(inputs, aggregation=self.aggregation, name=self.metric_name)
        return inputs

    def get_config(self):
        config = super(AddMetric, self).get_config()
        config.update(
            {"aggregation": self.aggregation, "metric_name": self.metric_name}
        )
        return config


def _in_functional_construction_mode(layer, inputs, args, kwargs, input_list):
    """Check the arguments to see if we are constructing a functional model."""
    # We are constructing a functional model if any of the inputs
    # are KerasTensors
    return any(
        isinstance(tensor, keras_tensor.KerasTensor)
        for tensor in tf.nest.flatten([inputs, args, kwargs])
    )


def _convert_numpy_or_python_types(x):
    if isinstance(x, (tf.Tensor, np.ndarray, float, int)):
        return tf.convert_to_tensor(x)
    return x


@keras_export("keras.__internal__.apply_name_scope_on_model_declaration", v1=[])
def _apply_name_scope_on_model_declaration(enable):
    """Apply `with tf.name_scope(...)` on model declaration.

    ```python
    tf.keras.__internal__.apply_name_scope_on_model_declaration(True)

    inputs = input_layer.Input((3,))
    with tf.name_scope('MyScope'):
      outputs = layers.Dense(10, name='MyDense')(inputs)
    model = tf.keras.Model(inputs, outputs)

    # with `tf.keras.__internal__.apply_name_scope_on_model_declaration(True)`,
    # The name of the dense layer is "model/MyScope/MyDense/*", and without,
    # "model/MyDense/*"
    ```

    Args:
      enable: Enables if `True`, disables if `False`.
    """
    if not isinstance(enable, bool):
        raise TypeError(f"`enable` argument must be `True` or `False`, got {enable}")

    global _is_name_scope_on_model_declaration_enabled
    _is_name_scope_on_model_declaration_enabled = enable


@keras_export("keras.__internal__.layers.BaseRandomLayer")
class BaseRandomLayer(Layer):
    """A layer handle the random number creation and savemodel behavior."""

    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def __init__(self, seed=None, force_generator=False, rng_type=None, **kwargs):
        """Initialize the BaseRandomLayer.

        Note that the constructor is annotated with
        @no_automatic_dependency_tracking. This is to skip the auto
        tracking of self._random_generator instance, which is an AutoTrackable.
        The backend.RandomGenerator could contain a tf.random.Generator instance
        which will have tf.Variable as the internal state. We want to avoid
        saving that state into model.weights and checkpoints for backward
        compatibility reason. In the meantime, we still need to make them
        visible to SavedModel when it is tracing the tf.function for the
        `call()`.
        See _list_extra_dependencies_for_serialization below for more details.

        Args:
          seed: optional integer, used to create RandomGenerator.
          force_generator: boolean, default to False, whether to force the
            RandomGenerator to use the code branch of tf.random.Generator.
          rng_type: string, the rng type that will be passed to backend
            RandomGenerator. Default to `None`, which will allow RandomGenerator
            to choose types by itself. Valid values are "stateful", "stateless",
            "legacy_stateful".
          **kwargs: other keyword arguments that will be passed to the parent
            *class
        """
        super().__init__(**kwargs)
        self._random_generator = backend.RandomGenerator(
            seed, force_generator=force_generator, rng_type=rng_type
        )

    def build(self, input_shape):
        super().build(input_shape)
        self._random_generator._maybe_init()

    def _trackable_children(self, save_type="checkpoint", **kwargs):
        if save_type == "savedmodel":
            cache = kwargs["cache"]
            # TODO(b/213628533): This must be called before super() to ensure
            # that any input shape changes are applied before getting the config
            # of the model.
            children = self._trackable_saved_model_saver.trackable_children(cache)
            # This method exposes the self._random_generator to SavedModel only
            # (not layer.weights and checkpoint).
            children["_random_generator"] = self._random_generator
        else:
            children = {}
        children.update(super()._trackable_children(save_type, **kwargs))
        return children

    def _lookup_dependency(self, name):
        # When loading from a Keras SavedModel load, make sure that the loader
        # can find the random generator, otherwise the loader will assume that
        # it does not exist, and will try to create a new generator.
        if name == "_random_generator":
            return self._random_generator
        else:
            return super()._lookup_dependency(name)


class Module(tf.Module):
    def __init_subclass__(cls, *args, **kwargs):
        """This wraps the user's __init__ method with a capture_params decorator.
        This allows the user to define their __init__ method as they normally would,
        while this method takes care of the automatic parameter capturing.
        """
        super().__init_subclass__(*args, **kwargs)
        cls.__user_init__ = cls.__init__
        # if the user has not already decorated the __init__ method with a capture decorator, decorate it.
        if not hasattr(cls.__user_init__, "_wrapper_capture_params_"):
            cls.__user_init__ = capture_params(cls.__init__, **kwargs)

        def _wrapped_init(self, *args, **kwargs):
            # call the user's __init__ method
            super().__init__(name=kwargs.pop("name", None))
            self.__user_init__(*args, **kwargs)

        cls.__init__ = _wrapped_init

    def __init__(self, name: str = None):
        super().__init__(name=name)
        self._built = False
        raise NotImplementedError("Subclasses must implement an __init__ method")

    def __call__(self, inputs):
        if not self._built:
            self.build(inputs)
            self._built = True
            self.forward(inputs)
        else:
            return self.forward(inputs)

    @property
    def built(self):
        return self._built

    @property
    def variables(self):
        return self.variables

    def build(
        self,
        input_shape: Union[tf.TensorSpec, Collection[tf.TensorSpec]],
        dtype: tf.DType = tf.float32,
    ):
        raise NotImplementedError("Subclasses must implement _build_layer method")

    def forward(self, inputs):
        raise NotImplementedError("Subclasses must implement forward method")
