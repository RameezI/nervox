import contextlib
import io
import sys

# define a function to block stdout from an underlying library
def block_underlying_stdout(func):
    def wrapper(*args, **kwargs):
        # redirect stdout to a buffer
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            # call the function with redirected stdout
            result = func(*args, **kwargs)
        # return the buffered output and the function result
        return buffer.getvalue(), result
    return wrapper


@block_underlying_stdout
def my_function():
    # call the underlying library function that writes to stdout
    # ...

    # call my_function and get the buffered output and the function result
    output, result = (23, 45)

    # display the buffered output
    print(output)

    # use the function result
    # ...
    return result, output

if __name__ == '__main__':
   print("Hello World", file=sys.stdout)