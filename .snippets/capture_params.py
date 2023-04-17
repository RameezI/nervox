import functools


def capture_params(*args_outer):
    def capture_params(func):
        """
        Decorator that captures parameters of a method.
        """
        @functools.wraps(func)

        def wrapper(self, *args, **kwargs):
            self.params = self.params if hasattr(self, "params") else {}
            self.params.update(kwargs)
            return func(self, *args, **kwargs)
        wrapper._wrapper_capture_params_ = True
        return wrapper
    
    if args_outer and callable(args_outer[0]):
        return capture_params(args_outer[0])
    else:
        return capture_params



class Transform:
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__user_init__ = cls.__init__          
        if not hasattr(cls.__user_init__, '_wrapper_capture_params_'):
            cls.__user_init__ = capture_params(cls.__init__, **kwargs)

        def _wrapped_init(self, *args, **kwargs):
                super().__init__()
                self.__user_init__(*args, **kwargs)

        cls.__init__ = _wrapped_init

class MyTransform(Transform):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

if __name__ == '__main__':
    mt = MyTransform(1, 2)
    print(mt.params)
