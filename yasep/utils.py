import functools
from typing import Callable


def reusable(gen_func: Callable) -> Callable:
    """
    Function decorator that turns your generator function into an
    iterator, thereby making it reusable.

    Parameters
    ----------
    gen_func: Callable
        Generator function, that you want to be reusable

    Returns
    ----------
    _multigen: Callable
        Sneakily created iterator class wrapping the generator function
    """

    @functools.wraps(gen_func, updated=())
    class _multigen:
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs

        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)

    return _multigen
