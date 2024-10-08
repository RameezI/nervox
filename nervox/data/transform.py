from nervox.utils import capture_params


class Transform:
    def __init_subclass__(cls, *args, **kwargs):
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
