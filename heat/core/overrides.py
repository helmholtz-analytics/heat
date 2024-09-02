"""Implementation of __array_function__ overrides from NEP-18."""

import collections
from functools import wraps
from typing import Any, Callable, Iterable, List, Set, Type

from heat.core.dndarray import DNDarray

_DNDARRAY_HEAT_FUNCTION = DNDarray.__heat_function__


def heat_function_dispatch(dispatcher, module=None):
    """Wrap a function for dispatch with the __heat_function__ protocol."""

    def decorator(implementation):
        @wraps(implementation)
        def public_api(*args, **kwargs):
            relevant_args = dispatcher(*args, **kwargs)
            return heat_function_implementation_or_override(
                implementation, public_api, relevant_args, args, kwargs
            )
            # return implement_heat_function(public_api, relevant_args, args, kwargs)

        if module is not None:
            public_api.__module__ = module
        # for dndarray.__heat_function__
        public_api._implementation = implementation
        return public_api

    return decorator


# pytorch and original numpy code style for the override included here. The main difference is that numpy loops through the argument types and removes
# all arguments that are not overriding the function and then have an early out if the list is empty, while pytorch
# does not and postpones it. Leads to a little change in the new overriding class subclass checking.
def _get_overloaded_args(
    relevant_args: Iterable[Any],
    get_type_fn: Callable[[Any], Type] = None,
) -> List[Any]:
    """Returns a list of arguments on which to call __heat_function__.

    Checks arguments in relevant_args for __heat_function__ implementations,
    storing references to the arguments and their types in overloaded_args and
    overloaded_types in order of calling precedence. Only distinct types are
    considered. If a type is a subclass of another type it will have higher
    precedence, otherwise the precedence order is the same as the order of
    arguments in relevant_args, that is, from left-to-right in the argument list.

    The precedence-determining algorithm implemented in this function is
    described in `NEP-0018`_.

    See torch::append_overloaded_arg for the equivalent function in the C++
    implementation.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of array-like arguments to check for __torch_function__
        methods.

    get_type_fn : callable, optional
        Function to call on each argument in relevant_args to get its type.

    Returns
    -------
    overloaded_args : list
        Arguments from relevant_args on which to call __torch_function__
        methods, in the order in which they should be called.

    .. _NEP-0018:
       https://numpy.org/neps/nep-0018-array-function-protocol.html
    """
    if get_type_fn is None:
        get_type_fn = type

    # Runtime is O(num_arguments * num_unique_types)
    overloaded_types: Set[Type] = set()
    overloaded_args: List[Any] = []

    # We only collect arguments if they have a unique type, which ensures
    # reasonable performance even with a long list of possibly overloaded
    # arguments.
    for arg in relevant_args:
        arg_type = get_type_fn(arg)
        if arg_type not in overloaded_types and hasattr(arg_type, "__heat_function__"):
            # Create lists explicitly for the first type (usually the only one
            # done) to avoid setting up the iterator for overloaded_args.
            if overloaded_types:
                overloaded_types.add(arg_type)
                # By default, insert argument at the end, but if it is
                # subclass of another argument, insert it before that argument.
                # This ensures "subclasses before superclasses".
                index = len(overloaded_args)
                for i, old_arg in enumerate(overloaded_args):
                    if issubclass(arg_type, get_type_fn(old_arg)):
                        index = i
                        break
                overloaded_args.insert(index, arg)
            else:
                overloaded_types = {arg_type}
                overloaded_args = [arg]
    return overloaded_args


def get_overloaded_types_and_args(relevant_args):
    """Returns a list of arguments on which to call __heat_function__.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of array-like arguments to check for __heat_function__
        methods.

    Returns
    -------
    overloaded_types : collection of types
        Types of arguments from relevant_args with __heat_function__ methods.
    overloaded_args : list
        Arguments from relevant_args on which to call __heat_function__
        methods, in the order in which they should be called.
    """
    # Runtime is O(num_arguments * num_unique_types)
    overloaded_types = []
    overloaded_args = []
    for arg in relevant_args:
        arg_type = type(arg)
        # We only collect arguments if they have a unique type, which ensures
        # reasonable performance even with a long list of possibly overloaded
        # arguments.
        if arg_type not in overloaded_types and hasattr(arg_type, "__heat_function__"):

            overloaded_types.append(arg_type)

            # By default, insert this argument at the end, but if it is
            # subclass of another argument, insert it before that argument.
            # This ensures "subclasses before superclasses".
            index = len(overloaded_args)
            for i, old_arg in enumerate(overloaded_args):
                if issubclass(arg_type, type(old_arg)):
                    index = i
                    break
            overloaded_args.insert(index, arg)

    # Special handling for ndarray.__array_function__
    overloaded_args = [
        arg for arg in overloaded_args if type(arg).__heat_function__ is not _DNDARRAY_HEAT_FUNCTION
    ]

    return overloaded_types, overloaded_args


def implement_heat_function(
    public_api: Callable, relevant_args: Iterable[Any], *args, **kwargs
) -> Any:
    """Implement a function with checks for ``__torch_function__`` overrides.

    See torch::autograd::handle_torch_function for the equivalent of this
    function in the C++ implementation.

    Arguments
    ---------
    public_api : function
        Function exposed by the public torch API originally called like
        ``public_api(*args, **kwargs)`` on which arguments are now being
        checked.
    relevant_args : iterable
        Iterable of arguments to check for __torch_function__ methods.
    args : tuple
        Arbitrary positional arguments originally passed into ``public_api``.
    kwargs : tuple
        Arbitrary keyword arguments originally passed into ``public_api``.

    Returns
    -------
    object
        Result from calling ``implementation`` or an ``__torch_function__``
        method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.

    Example
    -------
    >>> def func(a):
    ...     if has_torch_function_unary(a):
    ...         return handle_torch_function(func, (a,), a)
    ...     return a + 0
    """
    # Check for __torch_function__ methods.
    overloaded_args = _get_overloaded_args(relevant_args)
    # overloaded_args already have unique types.
    types = tuple(map(type, overloaded_args))

    # Call overrides
    for overloaded_arg in overloaded_args:
        heat_func_method = overloaded_arg.__heat_function__

        # Use `public_api` instead of `implementation` so __torch_function__
        # implementations can do equality/identity comparisons.
        result = heat_func_method(public_api, types, *args, **kwargs)

        if result is not NotImplemented:
            return result

    func_name = f"{public_api.__module__}.{public_api.__name__}"
    msg = (
        f"no implementation found for '{func_name}' on types that implement "
        f"__heat_function__: {[type(arg) for arg in overloaded_args]}"
    )

    raise TypeError(msg)


def heat_function_implementation_or_override(
    implementation, public_api, relevant_args, args, kwargs
):
    """Implement a function with checks for __array_function__ overrides.

    Arguments
    ---------
    implementation : function
        Function that implements the operation on NumPy array without
        overrides when called like ``implementation(*args, **kwargs)``.
    public_api : function
        Function exposed by NumPy's public API riginally called like
        ``public_api(*args, **kwargs`` on which arguments are now being
        checked.
    relevant_args : iterable
        Iterable of arguments to check for __array_function__ methods.
    args : tuple
        Arbitrary positional arguments originally passed into ``public_api``.
    kwargs : tuple
        Arbitrary keyword arguments originally passed into ``public_api``.

    Returns
    -------
    Result from calling `implementation()` or an `__array_function__`
    method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.
    """
    # Check for __heat_function__ methods.
    types, overloaded_args = get_overloaded_types_and_args(relevant_args)
    if not overloaded_args:
        return implementation(*args, **kwargs)

    # Call overrides
    for overloaded_arg in overloaded_args:
        # Use `public_api` instead of `implementation` so __heat_function__
        # implementations can do equality/identity comparisons.
        result = overloaded_arg.__heat_function__(public_api, types, args, kwargs)

        if result is not NotImplemented:
            return result

    raise TypeError(
        "no implementation found for {} on types that implement "
        "__heat_function__: {}".format(public_api, list(map(type, overloaded_args)))
    )
