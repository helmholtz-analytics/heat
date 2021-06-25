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
# Distribution engine.
#   - schedules same tasks on all workers
#   - handles dependences seperately
# This currently is a very simple eagerly executing machinery.
# We can make this better over time. A low hanging fruit seems might
# be to delay distribution until go() is called. This would allow aggregating
# multiple distribution messages into one.
#
# Dependent objects have a unique identifier, assigned when a handle to it is
# created. We assume that all workers execute handle-creation in identical order.
# Such dependences are assumed to be global entities, e.g. each worker holds
# a handle/reference to it (e.g. like a heat.DNDarray). The local handles
# exist on each, stored in a worker-local dictionary. Thsi allows identifying
# dependences through simple integers.
#
# Notice, mpi4py does not provide ibcast, so we cannot overlap. This makes the
# above aggregation particularly promising. Another option woujld be to write
# this in C/C++ and use ibcast.
###############################################################################


import sys
from mpi4py import MPI
_comm = MPI.COMM_WORLD

# define identifiers
END = 0
TASK = 1
GO = 2


def init():
    'Init distributor'
    pass


def start():
    '''
    Start distribution engine.
    Controller inits and returns.
    Workers enter recv-loop and exit program when fini si called.
    '''
    if _comm.rank != 0:
        done = False
        header = None
        rtask = None
        while(not done):
            # wait in bcast for work
            header = _comm.bcast(header, 0)
            # then see what we need to do
            if header[0] == END:
                done = True
                break
            elif header[0] == TASK:
                header[1].submit()
            elif header[0] == GO:
                # no delayed execution for now -> nothing to do
                pass
            else:
                raise Exception("Worker received unknown tag")
        sys.exit()

        
def fini():
    'Control sends end-tag. Workers will sys.exit'
    if _comm.rank == 0:
        header = [END]
        header = _comm.bcast(header, 0)


def go():
    'Trigger execution of all tasks that are still in flight'
    header = [GO]
    header = _comm.bcast(header, 0)


def submitPP(task, deps, in_order=True):
    '''
    Submit a process-parallel task and return a handle/future.
    '''
    rtask = _RemoteTask(task, deps)
    header = [TASK, rtask]
    _, rtask = _comm.bcast(header, 0)
    return rtask.submit()


class Handle:
    '''
    A future representing an object that will be available eventually.
    get() will return None as long as the value is not available.
    '''

    # this defines the next free and globally unique identifier
    _nextId = 1

    def __init__(self):
        '''
        Initialize handle.
        We assume all workers create handles to objects in identical order.
        This allows us to assign a simple integers as the unqique id.
        '''
        self._obj = None
        self._id = Handle._nextId
        Handle._nextId += 1

    def set(self, obj):
        'Make object available.'
        self._obj = obj

    def getId(self):
        'Return future/handle id'
        return self._id

    def get(self):
        'Return object or None'
        return self._obj
        

class _RemoteTask:
    '''
    A task which is executed remotely on a worker.
    It accepts a task with a run-method that it will execute at some point.
    It also accepts dependences explicitly and so allows to create
    task-graphs etc.

    We keep a static dictionary mapping globally unique identifiers to dependent
    global objects (like heat.DNDarrays). This keeps the objects alive and allows
    communicating through simple integers.
    '''

    def __init__(self, task, deps, inorder=True):
        self._depIds = deps
        self._task = task
        self._inorder = inorder

    # here we store objects that are input dependences to tasks
    s_pms = {}

    def submit(self):
        '''
        Submit task to local task scheduler.
        For now we execute eagerly, this is much simpler to implement.
        Later, we might consider lazy evaluation, task-graph-optimizations etc.
        FIXME: We currently assign a new id and store the result even when there is no result
        or the result is not a global object.
        '''
        deps = [_RemoteTask.s_pms[i] for i in self._depIds]
        res = self._task.run(deps)
        hndl = Handle()
        hndl.set(res)
        _RemoteTask.s_pms[hndl.getId()] = res
        return hndl
