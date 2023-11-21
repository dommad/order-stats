"""Various utility objects"""
import time


def fetch_instance(class_name, attribute_name, *args, **kwargs):
    """general fetches for class attributes by name and possibly initializing them"""
    try:
        return getattr(class_name, attribute_name)(*args, **kwargs)
    except AttributeError as exc:
        raise ValueError(f"Unsupported or invalid instance type: {class_name}") from exc


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds.")
        return result
    return wrapper


def log_function_call(func):

    def wrapper(*args, **kwargs):
        #print(f"Calling {func.__name__} with args {args} and kwargs {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result

    return wrapper



def _is_numeric(value):
    if not isinstance(value, str):
        return False
    try:
        float(value)
        return True

    except ValueError:
        return False


def largest_factors(n):
    for i in range(n // 2, 0, -1):
        if n % i == 0:
            return n // i, i


class StrClassNameMeta(type):

    def __str__(cls):
        return cls.__name__


class ParserError(Exception):
    """A custom exception class."""
    def __init__(self, message="An error occurred."):
        self.message = message
        super().__init__(self.message)

