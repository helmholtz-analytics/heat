# MIT License

# Copyright (c) 2021 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


###############################################################################
"""
This provides a wrapper around SPMD-based HeAT
(github.com/helmholtz-analytics/heat) to operate in controller-worker mode.

The goal is to provide a compliant implementation of the array API
(github.com/data-apis/arra-api).

Returned array (DNDArray) objects are handles/futures only. Their content is
available through __int__ etc., through __partitioned__ or heat(). Notice: this
allows for delayed execution and optimizations of the workflow/task-graph and
communication.

For a function/method of the array-API that is executed on the controller
process, this wrapper generates the equivalent source code to be executed on
the worker processes.  The code is then sent to each remote worker and
executed there.

It's up to the distribution layer (e.g. distributor) to make sure the code is
executed in the right order on each process/worker so that collective
communication in HeAT can operate correctly without dead-locks.

To allow workflow optimizations array dependences and to avoid
pickle-dependencies to the array inputs we separate scalar/non-array arguments
from array arguments. For this we assume that array arguments never occur
after non-array arguments.  Each function.task handles and passes array-typed
and non-array-types arguments separately.
"""
###############################################################################

import atexit
from . import distributor
from .arrayapi import (
    aa_attributes,
    aa_tlfuncs,
    aa_datatypes,
    aa_constants,
    aa_methods_s,
    aa_methods_a,
    aa_inplace_operators,
    aa_reflected_operators,
)

# just in case we find another SPMD/MPI implementation of numpy...
import heat as impl
from heat import DNDarray as dndarray

impl_str = "impl"
dndarray_str = "impl.DNDarray"


def init():
    """
    Initialize distribution engine. Automatically when when importing cw4heat.
    For now we assume all ranks (controller and workers) are started through mpirun,
    workers will never leave distributor.start() and so this function.
    """
    distributor.init()
    distributor.start()


def fini():
    """
    Finalize/shutdown distribution engine. Automatically called at exit.
    When called on controller, workers will sys.exit from init().
    """
    distributor.fini()


class _Task:
    """
    A work item, executing functions provided as code.
    """

    def __init__(self, func, args, kwargs, unwrap="*"):
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._unwrap = unwrap

    def run(self, deps):
        if deps:
            return eval(f"{self._func}({self._unwrap}deps, *self._args, **self._kwargs)")
        else:
            return eval(f"{self._func}(*self._args, **self._kwargs)")


class _PropertyTask:
    """
    A work item, executing class properties provided as code.
    """

    def __init__(self, func):
        self._func = func

    def run(self, deps):
        return eval(f"deps[0].{self._func}")


def _submit(name, args, kwargs, unwrap="*", numout=1):
    """
    Create a _Task and submit, return PManager/Future.
    """
    scalar_args = tuple(x for x in args if not isinstance(x, DDParray))
    deps = [x._handle.getId() for x in args if isinstance(x, DDParray)]
    return distributor.submitPP(_Task(name, scalar_args, kwargs, unwrap=unwrap), deps, numout)


def _submitProperty(name, self):
    """
    Create a _PropertyTask (property) and submit, return PManager/Future.
    """
    t = _PropertyTask(name)
    try:
        res = distributor.submitPP(t, [self._handle.getId()])
    except Exception:
        assert False
    return res


# setitem has scalar arg key before array arg value
# we need to provide a function accepting the inverse order
def _setitem_normalized(self, value, key):
    self.__setitem__(key, value)


#######################################################################
# Our array is just a wrapper. Actual array is stored as a handle to
# allow delayed execution.
#######################################################################
class DDParray:
    """
    Shallow wrapper class representing a distributed array.
    It will be filled dynamically from lists extracted from the array-API.
    All functionality is delegated to the underlying implementation,
    executed in tasks.
    """

    #######################################################################
    # first define methods/properties which need special care.
    #######################################################################

    def __init__(self, handle):
        """
        Do not use this array. Use creator functions instead.
        """
        self._handle = handle

    def heat(self):
        """
        Return heat native array.
        With delayed execution, triggers computation as needed and blocks until array is available.
        """
        return self._handle.get()

    def __getitem__(self, key):
        """
        Return item/slice as array.
        """
        return DDParray(_submit(f"{dndarray_str}.__getitem__", (self, key), {}))

    # bring args in the order we can process and feed into normal process
    # using global normalized version
    def __setitem__(self, key, value):
        """
        Set item/slice to given value.
        """
        _submit("_setitem_normalized", (self, value, key), {})

    @property
    def T(self):
        """
        Transpose.
        """
        return DDParray(_submitProperty("T", self))

    #######################################################################
    # Now we add methods/properties through the standard process.
    #######################################################################

    # dynamically generate class methods from list of methods in array-API
    # we simply make lambdas which submit appropriate Tasks
    # FIXME: aa_inplace_operators,others?
    fixme_afuncs = ["squeeze", "astype", "balance", "resplit"]
    for method in aa_methods_a + aa_reflected_operators + fixme_afuncs:
        if method not in ["__getitem__", "__setitem__"] and hasattr(dndarray, method):
            exec(
                f"{method} = lambda self, *args, **kwargs: DDParray(_submit('{dndarray_str}.{method}', (self, *args), kwargs))"
            )

    for method in aa_methods_s:
        if hasattr(dndarray, method):
            exec(
                f"{method} = lambda self, *args, **kwargs: _submit('{dndarray_str}.{method}', (self, *args), kwargs).get()"
            )

    for attr in aa_attributes:
        if attr != "T" and hasattr(dndarray, attr):
            exec(f"{attr} = property(lambda self: self._handle.get().{attr})")

    def __getattr__(self, attr):
        # attributes are special
        if attr not in aa_attributes:
            raise Exception(f"unknown method/attribute {attr} requested")


#######################################################################
# first define top-level functions which need special care.
#######################################################################

# np.concatenate accepts a list of arrays (not individual arrays)
# so we let the task not unwrap the list of deps
def concatenate(*args, **kwargs):
    """
    Wrapper for impl.concatenate.
    """
    return DDParray(_submit(f"{impl_str}.concatenate", *args, kwargs, unwrap=""))


#######################################################################
# first define top-level functions through the standard process.
#######################################################################
#   - creating arrays
#   - elementswise operations
#   - statistical operations
# (lists taken from list of methods in array-API)
# Again, we simply make lambdas which submit appropriate Tasks

fixme_funcs = ["load_csv", "array", "triu"]
for func in aa_tlfuncs + fixme_funcs:
    if func == "meshgrid":
        exec(
            f"{func} = lambda *args, **kwargs: list(DDParray(x) for x in _submit('{impl_str}.{func}', args, kwargs, numout=len(args)))"
        )
    else:
        exec(
            f"{func} = lambda *args, **kwargs: DDParray(_submit('{impl_str}.{func}', args, kwargs))"
        )


for func in ["concatenate", "hstack"]:
    exec(
        f"{func} = lambda *args, **kwargs: DDParray(_submit(f'{impl_str}.{func}', *args, kwargs, unwrap=''))"
    )


# Here we data types and constants
for attr in aa_datatypes + aa_constants:
    if hasattr(impl, attr):
        exec(f"{attr} = {impl_str}.{attr}")
    else:
        print(f"{impl.__name__} has no {attr}")


#######################################################################
# quick hack to provide random features
#######################################################################
class random:
    """
    Wrapper class for random.
    """

    for method, obj in impl.random.__dict__.items():
        if callable(obj):
            exec(
                f"{method} = staticmethod(lambda *args, **kwargs: DDParray(_submit('{impl_str}.random.{method}', args, kwargs)))"
            )


#######################################################################
#######################################################################
atexit.register(fini)
init()
