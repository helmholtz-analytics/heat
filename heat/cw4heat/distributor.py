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


# define identifiers
END = 0
TASK = 1
GO = 2
GET = 3
GETPART = 4
PUBPART = 5


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

    def len(self):
        return len(self._taskQueue)

    def clear(self):
        self._taskQueue.clear()


class Distributor:
    """
    Instances of this class distribute work from controller to workers.
    Work-items are treated as dependent tasks.
    """

    def __init__(self, comm=MPI.COMM_WORLD):
        """
        Init distributor, optionally accepts MPI communicator.
        """
        self._comm = comm
        # Our queue of tasks.
        self._tQueue = _TaskQueue()

    def start(self, doExit=True, initImpl=None):
        """
        Start distribution engine.
        Controller inits and returns.
        Workers enter recv-loop and exit program when fini is called.
        """
        if initImpl:
            initImpl(self._comm)
        if self._comm.rank == 0:
            return True
        else:
            print("Entering worker loop", flush=True)
            done = False
            header = None
            while not done:
                # wait in bcast for work
                header = self._comm.bcast(header, 0)
                # then see what we need to do
                if header[0] == TASK:
                    self._tQueue._taskQueue = header[1]
                elif header[0] == GET:
                    # We do not support arrays yet, scalars do not need communication
                    assert False
                elif header[0] == GO:
                    self._tQueue.go()
                elif header[0] == GETPART:
                    if self._comm.rank == header[1]:
                        val = _RemoteTask.getVal(header[2])
                        attr = getattr(val, header[3])
                        self._comm.send(attr, dest=0, tag=GETPART)
                elif header[0] == PUBPART:
                    val = _RemoteTask.getVal(header[1])
                    attr = header[3](getattr(val, header[2]))
                    self._comm.gather(attr, root=0)
                elif header[0] == END:
                    done = True
                    self._comm.Barrier()
                    break
                else:
                    raise Exception("Worker received unknown tag")
            MPI.Finalize()
            if doExit:
                sys.exit()
            return False

    def fini(self):
        """
        Control sends end-tag. Workers will sys.exit.
        """
        if MPI.Is_initialized() and self._comm.rank == 0:
            header = [END]
            header = self._comm.bcast(header, 0)
            self._comm.Barrier()
            # MPI.Finalize()

    def go(self):
        """
        Trigger execution of all tasks which are still in flight.
        """
        assert self._comm.rank == 0
        if self._tQueue.len():
            header = [TASK, self._tQueue._taskQueue]
            _, _ = self._comm.bcast(header, 0)
            header = [GO]
            _ = self._comm.bcast(header, 0)
            self._tQueue.go()

    def get(self, handle):
        """
        Get actualy value from handle.
        Requires communication.
        We get the value from worker 0 (rank 1 in global comm).
        Does not work for arrays (yet).
        """
        assert self._comm.rank == 0
        self.go()
        return handle.get()

    def getPart(self, handle, attr):
        """
        Get local raw partition data for given handle.
        """
        if handle.rank == self._comm.rank:
            val = _RemoteTask.getVal(handle.id)
            val = getattr(val, attr)
        else:
            # FIXME what if left CW-context (SPMD mode) ?
            assert self._comm.rank == 0
            header = [GETPART, handle.rank, handle.id, attr]
            _ = self._comm.bcast(header, 0)
            val = self._comm.recv(source=handle.rank, tag=GETPART)
        return val

    def publishParts(self, id, attr, publish):
        """
        Publish array's attribute for each partition and gather handles on root.
        """
        assert self._comm.rank == 0
        header = [PUBPART, id, attr, publish]
        _ = self._comm.bcast(header, 0)
        val = publish(getattr(_RemoteTask.getVal(id), attr))
        return self._comm.gather(val, root=0)

    def submitPP(self, task, deps, numout=1):
        """
        Submit a process-parallel task and return a handle/future.
        """
        rtask = _RemoteTask(task, deps, numout)
        return self._tQueue.submit(rtask)


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
        return self._obj

    def __getstate__(self):
        # we do not pickle the actual object
        return {"_id": self._id}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._obj = None


# here we store objects that are input dependences to tasks
_s_pms = {}


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

    def go(self):
        """
        Actually run the task.
        """
        # print(self._task._func)
        deps = [_s_pms[i] for i in self._depIds]
        res = self._task.run(deps)
        if self._nOut == 1:
            self._handle.set(res)
            _s_pms[self._handle.getId()] = res
        else:
            i = 0
            for h in self._handle:
                h.set(res[i])
                _s_pms[h.getId()] = res[i]
                i += 1
        return self._handle

    @staticmethod
    def getVal(id):
        return _s_pms[id]
