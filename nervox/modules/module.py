import tensorflow as tf
from nervox.utils import capture_params

class Module:
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__user_init__ = cls.__init__ 
        # if the user has not already decorated the __init__ method with capture method, decorate it...         
        if not hasattr(cls.__user_init__, '_wrapper_capture_params_'):
            cls.__user_init__ = capture_params(cls.__init__, **kwargs)

        def _wrapped_init(self, *args, **kwargs):
                # call the user's __init__ method
                super().__init__()
                self.__user_init__(*args, **kwargs)

        cls.__init__ = _wrapped_init

    def build(self, inputs):
        raise NotImplementedError("Subclasses must implement _build_layer method")
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement __init__ method")

    def __call__(self, inputs):
        return self.forward(inputs)
 


