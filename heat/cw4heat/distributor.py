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
Distribution engine.
  - schedules same tasks on all workers
  - handles dependences seperately

Whe tasks are submitted on root rank they are pushed on a queue and a
handle/future is returned. When computation is requested by calling go()
all tasks on the queue are sent to workers and executed on all ranks
sequentially.

We store tasks in the same order as they are submitted on the root rank.
For any valid program this must be a legal ordering there is no need to check
if dependent objects are ready when a task is executed. A more sophisticated
scheduler could potentially try to parallelize. It remains to be invistigated
if this would be a profitable feature, though.

Dependent objects have a unique identifier, assigned when a handle to it is
created. We assume that all workers execute handle-creation in identical order.
Such dependences are assumed to be global entities, e.g. each worker holds
a handle/reference to it (e.g. like a heat.DNDarray). The local handles
exist on each, stored in a worker-local dictionary. Thsi allows identifying
dependences through simple integers.

Notice, mpi4py does not provide ibcast, so we cannot overlap. This makes the
above aggregation particularly promising. Another option woujld be to write
this in C/C++ and use ibcast.
"""
###############################################################################


from mpi4py import MPI
import sys
from collections import deque

_comm = MPI.COMM_WORLD

# define identifiers
END = 0
TASK = 1
GO = 2


class _TaskQueue:
    """
    A task queue, each rank holds one for queuing up local tasks.
    We currently dissallow submitting tasks by on-root ranks.
    Non-root ranks get their TaskQueue set in the recv-lop if init().
    """

    def __init__(self):
        # here we store all tasks that have not been executed yet
        self._taskQueue = deque()

    def submit(self, rtask):
        """
        Sumbit a task to queue. Will not run it.
        """
        assert _comm.rank == 0
        self._taskQueue.append(rtask)
        return rtask._handle

    def go(self):
        """
        Run all tasks in the queue.
        We assume tasks were submitted in in a valid order, e.g. in an order
        that guarntees no task is dependent on another task that is behind it in the queue.
        """
        while len(self._taskQueue):
            self._taskQueue.popleft().go()


# Our queue of tasks.
_tQueue = _TaskQueue()


def init():
    """
    Init distributor.
    """
    pass


def start():
    """
    Start distribution engine.
    Controller inits and returns.
    Workers enter recv-loop and exit program when fini si called.
    """
    if _comm.rank != 0:
        done = False
        header = None
        while not done:
            # wait in bcast for work
            header = _comm.bcast(header, 0)
            # then see what we need to do
            if header[0] == END:
                done = True
                break
            elif header[0] == TASK:
                _tQueue._taskQueue = header[1]
            elif header[0] == GO:
                # no delayed execution for now -> nothing to do
                _tQueue.go()
            else:
                raise Exception("Worker received unknown tag")
        sys.exit()


def fini():
    """
    Control sends end-tag. Workers will sys.exit.
    """
    if _comm.rank == 0:
        header = [END]
        header = _comm.bcast(header, 0)


def go():
    """
    Trigger execution of all tasks which are still in flight.
    """
    assert _comm.rank == 0
    header = [TASK, _tQueue._taskQueue]
    _, _ = _comm.bcast(header, 0)
    header = [GO]
    _ = _comm.bcast(header, 0)
    _tQueue.go()


def submitPP(task, deps, numout=1):
    """
    Submit a process-parallel task and return a handle/future.
    """
    rtask = _RemoteTask(task, deps, numout)
    return _tQueue.submit(rtask)


class Handle:
    """
    A future representing an object that will be available eventually.
    get() will return None as long as the value is not available.
    """

    # this defines the next free and globally unique identifier
    _nextId = 1

    def __init__(self):
        """
        Initialize handle.
        We assume all workers create handles to objects in identical order.
        This allows us to assign a simple integers as the unqique id.
        """
        self._obj = None
        self._id = Handle._nextId
        Handle._nextId += 1

    def set(self, obj):
        """
        Make object available.
        """
        self._obj = obj

    def getId(self):
        """
        Return future/handle id
        """
        return self._id

    def get(self):
        """
        Return object or None
        """
        go()
        return self._obj


class _RemoteTask:
    """
    A task which is executed remotely on a worker.
    It accepts a task with a run-method that it will execute at some point.
    It also accepts dependences explicitly and so allows to create
    task-graphs etc.

    We keep a static dictionary mapping globally unique identifiers to dependent
    global objects (like heat.DNDarrays). This keeps the objects alive and allows
    communicating through simple integers.
    """

    def __init__(self, task, deps, numout):
        self._depIds = deps
        self._task = task
        self._nOut = numout
        # FIXME: We currently assign a new id and store the result even when there is no result
        #        or the result is not a global object.
        if self._nOut == 1:
            self._handle = Handle()
        else:
            self._handle = tuple(Handle() for _ in range(self._nOut))

    # here we store objects that are input dependences to tasks
    s_pms = {}

    def go(self):
        """
        Actually run the task.
        """
        deps = [_RemoteTask.s_pms[i] for i in self._depIds]
        res = self._task.run(deps)
        if self._nOut == 1:
            self._handle.set(res)
            _RemoteTask.s_pms[self._handle.getId()] = res
        else:
            i = 0
            for h in self._handle:
                h.set(res[i])
                _RemoteTask.s_pms[h.getId()] = res[i]
                i += 1
        return self._handle
