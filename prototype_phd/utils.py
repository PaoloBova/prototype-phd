import functools
import numpy as np

def thread_macro(current_value, *funcs, identifier="self"):
    """Pipes current_value through each function in funcs.

    Each element in funcs is either a function or a list/tuple containing
    a function followed by its other arguments.
    This function imitates the Clojure as-if threading macro.

    Notes: By default current_value is threaded as the first argument of the
    function call. Yet, one can use the syntax [func, arg1, "self", arg2] (or
    (func, arg1, "self", arg2)) so that current_value will instead be threaded
    in whatever place "self" would be. If you need to, you can set this "self"
    identifier to a different value.
    """

    for func in funcs:
        if isinstance(func, (list, tuple)):
            place = 0
            for i, el in enumerate(func[1:]):
                try:
                    if el == identifier:
                        place = i
                        func = [el for el in func if el != identifier]
                except:
                    pass
            func, args1, args2 = func[0], func[1:place + 1], func[place + 1:]
            current_value = func(*args1, current_value, *args2)
        else:
            current_value = func(current_value)
    return current_value

def assoc(m:dict, *kargs):
    "Add every two elements in kargs as a new key-value item in dictionary `m`."
    return {**m, **dict(zip(kargs[::2],
                            kargs[1::2]))}

def get(m:dict, k:str):
    "Get attribute k from dictionary m."
    return m.get(k)

def broadcast_concatenate_axes(ax1, ax2):
    """Broadcast both numpy axes and concatenate along last dimension"""
    ax1new = ax1
    for _ in range(np.ndim(ax2) - 1):
        ax1new = ax1new[..., None, :]
    ax2new = ax2
    for _ in range(np.ndim(ax1) - 1):
        ax2new = ax2new[None, ..., :]
    ax1new = np.broadcast_to(ax1new,
                             (*ax1.shape[:-1], *ax2.shape[:-1], ax1.shape[-1]))
    ax2new = np.broadcast_to(ax2new,
                             (*ax1.shape[:-1], *ax2.shape[:-1], ax2.shape[-1]))
    ax = np.concatenate((ax1new, ax2new), axis=-1)
    return ax

def build_grid_from_axes(axes:list, # Each axis in axes gives an array of values that should be repeated for each value in the other axes. Primitive types and lists of primitive types are first promoted to numpy arrays.
                         override:bool=False, # whether to build the grid if it is very large
                        ) -> np.ndarray: # A 2D numpy array with all combinations of elements specified in axes
    """Build a numpy array with all combinations of elements specified in axes."""

    dtypes = (float, int, bool, str)
    for i, axis in enumerate(axes):
        condition = (isinstance(axis, dtypes)
                     or all(isinstance(el, dtypes) for el in list(axis))
                     or (isinstance(axis, np.ndarray) and np.ndim(axis)==1))
        axes[i] = np.array([axis]).T if condition else axis
    final_size = np.prod([axis.shape[0] for axis in axes])
    if (final_size > 5*10**6) & (not override):
        raise ValueError(f"""Your axes imply you want to create a grid with {final_size} > 5 million rows!
        If you're confident you can do this without crashing your computer, pass override=True to this function.""")
    tensor = functools.reduce(broadcast_concatenate_axes, axes)
    return tensor.reshape((-1, tensor.shape[-1]))

def multi(dispatch_fn):
    def _inner(*args, **kwargs):
        return _inner.__multi__.get(
            dispatch_fn(*args, **kwargs),
            _inner.__multi_default__
        )(*args, **kwargs)
    
    _inner.__dispatch_fn__ = dispatch_fn
    _inner.__multi__ = {}
    _inner.__multi_default__ = lambda *args, **kwargs: None  # Default default
    return _inner

def method(dispatch_fn, dispatch_key=None):
    def apply_decorator(fn):
        if dispatch_key is None:
            # Default case
            dispatch_fn.__multi_default__ = fn
        else:
            dispatch_fn.__multi__[dispatch_key] = fn
        return dispatch_fn
    return apply_decorator

@multi
def area(shape):
    return shape.get('type')

@method(area, 'square')
def area(square):
    return square['width'] * square['height']

@method(area, 'circle')
def area(circle):
    return circle['radius'] ** 2 * 3.14159

@method(area)
def area(unknown_shape):
    raise Exception("Can't calculate the area of this shape")

def string_to_tuple(string):
    """Convert a string containing only integers and dashes to a tuple of
    integers in reverse order."""
    return thread_macro(string,
                        (str.split, "-"),
                        (map, int, "self"),
                        list,
                        reversed,
                        list,
                        np.array,
                        )

