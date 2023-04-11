import tensorflow as tf
from nervox.utils import capture_params


class TestCaptureParams(tf.test.TestCase):
    def test_capture_params_no_outer_parms(self):
        class MyClass:
            @capture_params
            def my_method(self, arg1, arg2, arg3="default"):
                pass

        my_obj = MyClass()
        my_obj.my_method(arg1="value1", arg2="value2")

        my_config = my_obj.params["my_method"]
        self.assertEqual(my_config, {"arg1": "value1", "arg2": "value2", "arg3": "default"})

    def test_capture_params_with_kwargs(self):
        class MyClass:
            @capture_params(ignore=["arg1"])
            def my_method(self, arg1, arg2, arg3="default"):
                pass

        my_obj = MyClass()
        my_obj.my_method(arg1="ignore", arg2="value2")

        my_config = my_obj.params["my_method"]
        self.assertEqual(my_config, {"arg2": "value2", "arg3": "default"})

    def test_capture_params_with_args(self):
        class MyClass:
            @capture_params(ignore=["arg1"])
            def my_method(self, arg1, arg2, arg3="default"):
                pass

        my_obj = MyClass()
        my_obj.my_method("ignore", "value2")
        my_config = my_obj.params["my_method"]
        self.assertEqual(my_config, {"arg2": "value2", "arg3": "default"})

    def test_capture_params_apply_local_updates(self):
        class MyClass:
            @capture_params(apply_local_updates=True)
            def my_method(self, arg1, arg2, arg3="default"):
                arg1 = "new value"

        my_obj = MyClass()
        my_obj.my_method("value1", "value2")
        my_config = my_obj.params["my_method"]
        self.assertEqual(
            my_config, {"arg1": "new value", "arg2": "value2", "arg3": "default"}
        )

    def test_capture_params_wrapped_call(self):
        class MyClass:
            @capture_params(apply_local_updates=True)
            def my_method(self, arg1, arg2, arg3="default"):
                arg1 = "new value"

        my_obj = MyClass()

        def wrapped_call(*args, **kwargs):
            my_obj.my_method(*args, **kwargs)

        wrapped_call("value1", "value2", arg3="updated_arg3")
        my_config = my_obj.params["my_method"]
        self.assertEqual(
            my_config,
            {"arg1": "new value", "arg2": "value2", "arg3": "updated_arg3"},
        )

    def test_capture_params_keywords_only(self):
        class MyClass:
            @capture_params(apply_local_updates=True)
            def my_method(self, arg1, arg2, *, arg3="default"):
                arg3 = "updated_arg3"

        my_obj = MyClass()
        my_obj.my_method("value1", "value2", arg3="updated_arg3")
        my_config = my_obj.params["my_method"]
        self.assertEqual(
            my_config,
            {"arg1": "value1", "arg2": "value2", "arg3": "updated_arg3"},
        )

    def test_capture_params_varkwargs(self):
        class MyClass:
            @capture_params(apply_local_updates=True)
            def my_method(self, arg1, arg2, arg3="default", **kwargs):
                arg3 = "updated_arg3"

        my_obj = MyClass()
        my_obj.my_method("value1", "value2", arg3="updated_arg3", arg4="new_kwarg4")
        my_config = my_obj.params["my_method"]
        self.assertEqual(
            my_config,
            {
                "arg1": "value1",
                "arg2": "value2",
                "arg3": "updated_arg3",
                "arg4": "new_kwarg4",
            },
        )

    def test_capture_params_positional_only(self):
        with self.assertRaises(TypeError):

            class MyClass:
                @capture_params
                def my_method(self, arg1, /, arg2, *, arg3="default"):
                    pass

            my_obj = MyClass()
            my_obj.my_method("value1", "value2", arg3="updated_arg3")

    def test_capture_params_var_positional(self):
        with self.assertRaises(TypeError):

            class MyClass:
                @capture_params
                def my_method(self, *args, arg1, arg2, arg3="default"):
                    pass

            my_obj = MyClass()
            my_obj.my_method("value1", "value2", arg3="updated_arg3")


if __name__ == "__main__":
    tf.test.main()
