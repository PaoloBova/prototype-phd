import functools
import itertools
import logging
import numpy
import time
from typing import Any, Dict, Generator, Sequence, List, Optional, Tuple, Union, Iterable
import warnings

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

def dict_values_are_scalar(dictionary):
    return all(not isinstance(v, Iterable)
               or isinstance(v, str) for v in dictionary.values())

def dict_list(params):
    """Convert a dictionary of lists into a list of dictionaries. See DrWatson.jl
    for a similar function in Julia.
    
    See: https://juliadynamics.github.io/DrWatson.jl/dev/run&list/#DrWatson.dict_list
    """
    keys = params.keys()
    values = [v if isinstance(v, list) else [v] for v in params.values()]
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def set_nested_value(d: Dict[str, Any],
                     keypath: List[str],
                     value: Any) -> None:
    """
    Set a value in a nested dictionary using a list of keys as the path.

    This function traverses or creates nested dictionaries based on the provided keypath and sets the specified value at the final key in the path. If any key in the path does not exist, it creates a new dictionary at that key.

    Parameters:
    d (dict): The dictionary in which to set the value.
    keypath (list): A list of keys representing the path to the nested value. Each element in the list is a key in the dictionary.
    value: The value to set at the specified keypath.

    Returns:
    None

    Example:
    >>> d = {'a': {'b': {'c': {}, 'd': 2}}}
    >>> set_nested_value(d, ['a', 'b', 'c'], 2)
    >>> print(d)
    {'a': {'b': {'c': 2, 'd': 2}}}

    Notes:
    - If the keypath is empty, the function does nothing.
    - If the keypath contains only one key, the value is set at the top level of the dictionary.
    - The function modifies the dictionary in place and does not return a new dictionary.
    """
    for key in keypath[:-1]:
        d = d.setdefault(key, {})
    d[keypath[-1]] = value

def assoc_in(d: Dict[str, Any],
             keypath: List[str],
             value: Any) -> Dict[str, Any]:
    """
    Set a value in a nested dictionary using a list of keys as the path.

    This function traverses or creates nested dictionaries based on the provided keypath and sets the specified value at the final key in the path. If any key in the path does not exist, it creates a new dictionary at that key.

    Parameters:
    d (dict): The dictionary in which to set the value.
    keypath (list): A list of keys representing the path to the nested value. Each element in the list is a key in the dictionary.
    value: The value to set at the specified keypath.

    Returns:
    dict: A new dictionary with the updated value.

    Example:
    >>> d = {'a': {'b': {'c': {}, 'd': 2}}}
    >>> new_d = assoc_in(d, ['a', 'b', 'c'], 2)
    >>> print(new_d)
    {'a': {'b': {'c': 2, 'd': 2}}}

    Notes:
    - If the keypath is empty, the function returns the original dictionary.
    - If the keypath contains only one key, the value is set at the top level of the dictionary.
    - The function creates and returns a new dictionary and does not modify the original dictionary.

    Inspired by the Clojure function `assoc-in`.
    See: https://clojuredocs.org/clojure.core/assoc-in
    """
    if not keypath:
        return d

    new_d = d.copy()
    current = new_d
    for key in keypath[:-1]:
        current[key] = current.get(key, {}).copy()
        current = current[key]
    current[keypath[-1]] = value
    return new_d

def get_in(d: Dict[str, Any],
           keypath: List[str],
           default: Optional[Any] = None) -> Any:
    """
    Retrieve a value from a nested dictionary using a list of keys as the path.

    This function traverses the nested dictionary based on the provided keypath and returns the value at the final key in the path. If any key in the path does not exist, it returns the specified default value.

    Parameters:
    d (dict): The dictionary from which to retrieve the value.
    keypath (list): A list of keys representing the path to the nested value. Each element in the list is a key in the dictionary.
    default: The value to return if any key in the path does not exist. Default is None.

    Returns:
    The value at the specified keypath, or the default value if any key in the path does not exist.

    Example:
    >>> d = {'a': {'b': {'c': 42}}}
    >>> value = get_in(d, ['a', 'b', 'c'])
    >>> print(value)
    42
    >>> value = get_in(d, ['a', 'b', 'd'], default='not found')
    >>> print(value)
    'not found'

    Inspired by the Clojure function `get-in`.
    See: https://clojuredocs.org/clojure.core/get-in
    """
    for key in keypath:
        if key in d:
            d = d[key]
        else:
            return default
    return d

def is_expandable(value: Any) -> bool:
    """
    Check if a value is an expandable iterable.

    This function checks if the given value is an iterable (excluding strings and dictionaries),
    and if it can be converted to a (non-empty) numpy array.

    Parameters:
    value: The value to check.

    Returns:
    bool: True if the value is an expandable iterable, False otherwise.
    """
    if isinstance(value, Iterable) and not isinstance(value, (str, dict)):
        try:
            arr = numpy.array(value)
            return arr.size > 0
        except (TypeError, ValueError):
            # Handle cases where value contains non-hashable elements or other issues
            return False
    return False

def is_homogeneous(arr: Iterable[Any]) -> bool:
    """
    Check if all elements in the iterable are of the same type.

    Parameters:
    arr (Iterable): The iterable to check.

    Returns:
    bool: True if all elements are of the same type, False otherwise.
    """
    arr = list(arr)  # Convert to list to handle multiple iterations
    if not arr:
        return True  # An empty iterable is considered homogeneous

    first_type = type(arr[0])
    return all(isinstance(x, first_type) for x in arr)

def find_expandable_items(d: Dict[str, Any],
                          current_path: Optional[List[str]] = None) -> Generator[Tuple[List[str], Tuple[int, ...]], None, None]:
    """
    Recursively find expandable items in a nested dictionary.

    This function traverses a nested dictionary and yields the paths and shapes of expandable items.
    An item is considered expandable if it is an iterable (excluding strings and dictionaries) and
    can be converted to a numpy array.

    Parameters:
    d (dict): The dictionary to search for expandable items.
    current_path (list, optional): The current path in the dictionary traversal. Default is None.

    Yields:
    tuple: A tuple containing the path to the expandable item and its shape.

    Example:
    >>> d = {'a': {'b': [1, 2, 2, 3], 'c': {'d': [1, 1, 1]}}}
    >>> list(find_expandable_items(d))
    [(['a', 'b'], (4,))]
    """
    if current_path is None:
        current_path = []
 
    for key, value in d.items():
        path = current_path + [key]
        if isinstance(value, dict):
            yield from find_expandable_items(value, path)
        elif is_expandable(value):
            yield path, numpy.array(value).shape

def broadcast_concatenate_axes(ax1, ax2):
    """Broadcast both numpy axes and concatenate along last dimension"""
    ax1new = ax1
    for _ in range(numpy.ndim(ax2) - 1):
        ax1new = ax1new[..., None, :]
    ax2new = ax2
    for _ in range(numpy.ndim(ax1) - 1):
        ax2new = ax2new[None, ..., :]
    ax1new = numpy.broadcast_to(ax1new,
                             (*ax1.shape[:-1], *ax2.shape[:-1], ax1.shape[-1]))
    ax2new = numpy.broadcast_to(ax2new,
                             (*ax1.shape[:-1], *ax2.shape[:-1], ax2.shape[-1]))
    ax = numpy.concatenate((ax1new, ax2new), axis=-1)
    return ax

def build_grid_from_axes(axes:list, # Each axis in axes gives an array of values that should be repeated for each value in the other axes. Primitive types and lists of primitive types are first promoted to numpy arrays.
                         override:bool=False, # whether to build the grid if it is very large
                        ) -> numpy.ndarray: # A 2D numpy array with all combinations of elements specified in axes
    """Build a numpy array with all combinations of elements specified in axes."""

    dtypes = (float, int, bool, str)
    for i, axis in enumerate(axes):
        condition = (isinstance(axis, dtypes)
                     or all(isinstance(el, dtypes) for el in list(axis))
                     or (isinstance(axis, numpy.ndarray) and numpy.ndim(axis)==1))
        axes[i] = numpy.array([axis]).T if condition else axis
    final_size = numpy.prod([axis.shape[0] for axis in axes])
    if (final_size > 5*10**6) & (not override):
        raise ValueError(f"""Your axes imply you want to create a grid with {final_size} > 5 million rows!
        If you're confident you can do this without crashing your computer, pass override=True to this function.""")
    tensor = functools.reduce(broadcast_concatenate_axes, axes)
    return tensor.reshape((-1, tensor.shape[-1]))

def expand_parameters(params: Dict[str, Any], 
                      expand_keys: Sequence[Union[str, Sequence[str]]], 
                      verbose: bool = False,
                      drop_repeats: bool = False,
                      size_limit: int = 1_000_000) -> Dict[str, Any]:
    """
    Expand specified parameters into arrays for simulation use, generating all combinations.

    This function takes a dictionary of parameters and expands specified parameters
    into numpy arrays, creating all combinations of their values. It supports
    nested structures, provides verbose logging, and handles repeat values.
    
    This is useful when a model runs an array of simulations in a single run.

    Args:
        params (Dict[str, Any]): A dictionary of parameter names and their values.
            Values can be scalars, lists, numpy arrays, or nested dictionaries.
        expand_keys (List[Union[str, List[str]]]): A list of keys or keypaths
            from params that should be expanded. Keypaths for nested 
            should be provided as lists of keys.
        verbose (bool, optional): If True, log information about expansion
            process. Defaults to False.
        drop_repeats (bool, optional): If True, drop repeat values in expandable
            parameters. Defaults to False.
        size_limit (int, optional): The maximum number of rows allowed
            in the expanded arrays before throwing an error. Defaults to 1,000,000.

    Returns:
        Dict[str, Any]: A new dictionary where specified parameters are expanded into 
        numpy arrays representing all combinations, and others remain unchanged.

    Raises:
        KeyError: If a key or keypath specified in expand_keys is not found in params.
        ValueError: If a value cannot be converted to a numpy array.
        MemoryError: If the size of the expanded arrays exceeds the size_limit.

    Notes:
        - Parameters specified in expand_keys will be expanded to numpy arrays if possible.
        - Non-expandable items will be skipped and logged if verbose is True.
        - Scalar values in expand_keys will be treated as single-element arrays.
        - The function generates all combinations of the expanded parameters.
        - Nested structures are supported through the use of keypaths.
        - When verbose is True, it logs information about all potential expansion
        candidates and the expansion process.
        - Warns about repeat values when drop_repeats is False.

    Examples:
        >>> params = {'a': [1, 2, 2], 'b': 3, 'c': {'d': [4, 5], 'e': {'f': [6, 7, 7]}}}
        >>> expand_parameters(params, ['a', ['c', 'd'], ['c', 'e', 'f'], 'b'], verbose=True, drop_repeats=True)
        {'a': array([1, 2]), 'b': 3, 'c': {'d': array([4, 5]), 'e': {'f': array([6, 7])}}}
    """

    # Prepare values for expansion
    values_to_expand = []
    expand_keypaths = []
    skipped_keypaths = []

    for key in expand_keys:
        keypath = [key] if isinstance(key, str) else key
        value = get_in(params, keypath)
        if numpy.isscalar(value):
            # We treat scalars as single-element arrays.
            value = [value]
        if value is None:
            # Keypath not found in params or has invalid value None
            skipped_keypaths.append(keypath)
            continue

        if not is_expandable(value):
            # Skip non-expandable items.
            skipped_keypaths.append(keypath)
            continue
        
        # Convert value to numpy array
        try:
            arr = numpy.array(value)
        except ValueError as e:
            raise ValueError(f"Unable to convert value at {keypath} to numpy array: {e}")
        
        # Drop repeat values if specified
        if drop_repeats:
            unique_arr = numpy.unique(arr)
            if unique_arr.size != arr.size and verbose:
                logging.info(f"Repeat values dropped at {' -> '.join(map(str, keypath))}. Original: {arr}, Unique: {unique_arr}")
            arr = unique_arr
        elif numpy.unique(arr).size != arr.size:
            warnings.warn(f"Repeat values found at {' -> '.join(map(str, keypath))}. Consider using drop_repeats=True if this is unintended.")
        
        values_to_expand.append(arr)
        expand_keypaths.append(keypath)

    # Calculate the size of the final arrays before expanding
    num_combinations = numpy.prod([len(arr) for arr in values_to_expand])

    # Log potential expansion candidates, skipped keypaths, and expansion keypaths
    if verbose:
        candidates = list(find_expandable_items(params))
        candidates = [(keypath, _) for keypath, _ in candidates
                      if keypath not in expand_keys]

        if len(candidates) > 0:
            logging.info("Potential missing expansion candidates:")
            for keypath, arr_shape in candidates:
                if keypath not in expand_keys:
                    logging.info(f"  {' -> '.join(map(str, keypath))}, shape: {arr_shape}")

        if skipped_keypaths:
            logging.info(f"The following keypaths were skipped as they are not expandable: {skipped_keypaths}")
        logging.info(f"Expanding the following keypaths: {expand_keypaths}")

        logging.info(f"Number of rows in the expanded arrays: {num_combinations}")

    # Check if the size exceeds the threshold
    if num_combinations > size_limit:
        raise MemoryError(f"The size of the expanded arrays ({num_combinations} rows) exceeds the size_limit ({size_limit} rows).")

    # Check if values_to_expand and expand_keypaths are not empty
    if not values_to_expand or not expand_keypaths:
        return params

    # Start timing
    start_time = time.time()
    
    # Otherwise, generate all combinations as a numpy array
    combinations = build_grid_from_axes(values_to_expand, override=True)

    # End timing
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    if verbose:
         logging.info((f"Time taken: {elapsed_time:.6f} seconds"))

    # Create expanded arrays
    expanded_params = params.copy()  # Create a copy to avoid modifying the original
    for i, keypath in enumerate(expand_keypaths):
        expanded_value = combinations[:, i]
        set_nested_value(expanded_params, keypath, expanded_value)

    return expanded_params

def get_keypaths(d: Dict[str, Any], current_path: Union[List[str], None] = None) -> List[List[Union[str, int]]]:
    """
    Recursively get all keypaths from a nested dictionary.

    Args:
        d (Dict[str, Any]): The dictionary to traverse.
        current_path (Union[List[str], None], optional): The current path in the traversal. Defaults to None.

    Returns:
        List[List[Union[str, int]]]: A list of keypaths, where each keypath is a list of keys.
    """
    if current_path is None:
        current_path = []

    keypaths = []
    for key, value in d.items():
        new_path = current_path + [key]
        if isinstance(value, dict):
            keypaths.extend(get_keypaths(value, new_path))
        else:
            keypaths.append(new_path)
    
    return keypaths

def string_to_tuple(string):
    """Convert a string containing only integers and dashes to a tuple of
    integers in reverse order."""
    return thread_macro(string,
                        (str.split, "-"),
                        (map, int, "self"),
                        list,
                        reversed,
                        list,
                        numpy.array,
                        )

# **Disclaimer:** Unlike the code above, this code is not my invention. All credit
# goes to Adam Bard for coming up with this (and Guido for writing an earlier
# implementation). Adam Bard made this code freely available at 
# https://adambard.com/blog/implementing-multimethods-in-python/.

# Alternatively, this package exists but only works based on type hints: https://pypi.org/project/multimethod/#description.
# In my opinion, the clojure dispatch function approach is far more versatile. Big thanks to Adam Bard for implementing this.

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


def expand_sweep_config(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand a parameter sweep configuration into individual run configurations.
    
    This function takes a sweep configuration with a base template and parameter
    sweep specifications, then generates all combinations of parameter values
    as individual run configurations.
    
    Args:
        sweep_config (Dict[str, Any]): A dictionary containing:
            - 'base_config': Template configuration that serves as the base for all runs
            - '_sweep': List of sweep specifications, each containing:
                - 'path': List representing nested path to parameter (supports integer indices)
                - 'values': List of values to sweep over for this parameter
    
    Returns:
        List[Dict[str, Any]]: List of individual run configurations, one for each
        parameter combination. Each config is a complete copy of base_config with
        parameter values applied.
    
    Raises:
        ValueError: If sweep_config is missing required keys or has invalid structure
        TypeError: If path elements or values have incorrect types
    
    Example:
        >>> sweep_config = {
        ...     "base_config": {
        ...         "model": {"lr": 0.01, "layers": [64, 32]},
        ...         "training": {"epochs": 100}
        ...     },
        ...     "_sweep": [
        ...         {"path": ["model", "lr"], "values": [0.01, 0.001]},
        ...         {"path": ["model", "layers", 0], "values": [64, 128]},
        ...         {"path": ["training", "epochs"], "values": [100, 200]}
        ...     ]
        ... }
        >>> configs = expand_sweep_config(sweep_config)
        >>> len(configs)
        8
        >>> configs[0]["model"]["lr"]
        0.01
        >>> configs[4]["model"]["lr"] 
        0.001
    
    Notes:
        - Generates cartesian product of all sweep parameter values
        - Supports nested dictionary paths and array indices in paths
        - Each generated config includes '_sweep_metadata' with combination info
        - Path elements can be strings (dict keys) or integers (array indices)
        - Base config is deep copied for each run to avoid mutation
    """
    import copy
    
    # Validate input structure
    if not isinstance(sweep_config, dict):
        raise TypeError("sweep_config must be a dictionary")
    
    if "base_config" not in sweep_config:
        raise ValueError("sweep_config must contain 'base_config' key")
    
    if "_sweep" not in sweep_config:
        raise ValueError("sweep_config must contain '_sweep' key")
    
    base_config = sweep_config["base_config"]
    sweep_specs = sweep_config["_sweep"]
    
    if not isinstance(sweep_specs, list):
        raise TypeError("_sweep must be a list of sweep specifications")
    
    # Validate each sweep specification
    for i, spec in enumerate(sweep_specs):
        if not isinstance(spec, dict):
            raise TypeError(f"Sweep specification {i} must be a dictionary")
        
        if "path" not in spec:
            raise ValueError(f"Sweep specification {i} missing 'path' key")
        if "values" not in spec:
            raise ValueError(f"Sweep specification {i} missing 'values' key")
        
        if not isinstance(spec["path"], list):
            raise TypeError(f"Sweep specification {i}: 'path' must be a list")
        if not isinstance(spec["values"], list):
            raise TypeError(f"Sweep specification {i}: 'values' must be a list")
        
        # Validate path elements
        for j, path_element in enumerate(spec["path"]):
            if not isinstance(path_element, (str, int)):
                raise TypeError(f"Sweep specification {i}: path element {j} must be string or integer")
    
    # Extract parameter combinations
    param_paths = [spec["path"] for spec in sweep_specs]
    param_values = [spec["values"] for spec in sweep_specs]
    
    # Generate all combinations using itertools.product
    run_configs = []
    
    for combination_values in itertools.product(*param_values):
        # Deep copy base config for this run
        run_config = copy.deepcopy(base_config)
        
        # Apply each parameter value to its path
        combination_dict = {}
        for path, value in zip(param_paths, combination_values):
            _set_nested_value_by_path(run_config, path, value)
            # Store combination info for metadata
            path_str = ".".join(str(p) for p in path)
            combination_dict[path_str] = value
        
        # Add metadata about this parameter combination
        run_config["_sweep_metadata"] = {
            "parameter_combination": combination_dict,
            "combination_index": len(run_configs)
        }
        
        run_configs.append(run_config)
    
    return run_configs


def _set_nested_value_by_path(config: Dict[str, Any], path: List[Union[str, int]], value: Any) -> None:
    """
    Set a value in a nested dictionary/list structure using a path.
    
    This helper function navigates through nested dictionaries and lists to set
    a value at the specified path. It handles both dictionary keys (strings) and
    list indices (integers).
    
    Args:
        config (Dict[str, Any]): The configuration dictionary to modify
        path (List[Union[str, int]]): Path to the target location
        value (Any): Value to set at the target location
    
    Raises:
        KeyError: If a dictionary key in the path doesn't exist
        IndexError: If a list index in the path is out of bounds
        TypeError: If path navigation encounters wrong types
    
    Example:
        >>> config = {"model": {"params": [1, 2, 3]}}
        >>> _set_nested_value_by_path(config, ["model", "params", 1], 99)
        >>> config["model"]["params"][1]
        99
    """
    current = config
    
    # Navigate to the parent of the target location
    for path_element in path[:-1]:
        if isinstance(path_element, str):
            # Dictionary key
            if not isinstance(current, dict):
                raise TypeError(f"Expected dict at path element '{path_element}', got {type(current)}")
            if path_element not in current:
                raise KeyError(f"Key '{path_element}' not found in configuration")
            current = current[path_element]
        elif isinstance(path_element, int):
            # List index
            if not isinstance(current, list):
                raise TypeError(f"Expected list at path element {path_element}, got {type(current)}")
            if path_element >= len(current) or path_element < -len(current):
                raise IndexError(f"List index {path_element} out of bounds")
            current = current[path_element]
        else:
            raise TypeError(f"Path element must be string or int, got {type(path_element)}")
    
    # Set the final value
    final_element = path[-1]
    if isinstance(final_element, str):
        if not isinstance(current, dict):
            raise TypeError(f"Expected dict for final key '{final_element}', got {type(current)}")
        current[final_element] = value
    elif isinstance(final_element, int):
        if not isinstance(current, list):
            raise TypeError(f"Expected list for final index {final_element}, got {type(current)}")
        if final_element >= len(current) or final_element < -len(current):
            raise IndexError(f"List index {final_element} out of bounds")
        current[final_element] = value
    else:
        raise TypeError(f"Final path element must be string or int, got {type(final_element)}")