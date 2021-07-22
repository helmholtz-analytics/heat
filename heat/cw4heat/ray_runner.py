# ===============================================================================
# Copyright 2014-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

"""
A Ray backend for HeAT controller-worker wrapper.

1. Init() nitializes actors
   - one for each node in the existing ray cluster
   - actors connect through MPI
2. Start actors
   - actors will sit in recv-loop and wait for work
3. fini() kills all actors.
   - Make sure you let distributor end recv-loop before calling this.
"""

from mpi4py import MPI
import ray
import ray.cloudpickle
from ray.services import get_node_ip_address as getIP
from .distributor import Distributor
import os
from os import getenv, getpid


@ray.remote
class RayActor:
    """
    A ray actor which connects to other actors and controller through MPI.
    """

    def __init__(self, node):
        self.node = node
        self._commWorld = MPI.COMM_SELF
        self._distributor = None

    def connect(self, port, nWorkers):
        """
        Let nWorkers-many processes connect to controller process.
        """
        # workers go here
        # initial connect
        intercomm = self._commWorld.Connect(port)
        # merge communicator
        self._commWorld = intercomm.Merge(1)
        intercomm.Disconnect()
        rank = self._commWorld.rank
        # collectively accept connections from all (other) clients
        for i in range(rank, nWorkers):
            # connect to next worker (collectively)
            intercomm = self._commWorld.Accept(port)
            # merge communicators
            self._commWorld = intercomm.Merge(0)
            intercomm.Disconnect()
        # setup our distributor
        assert self._distributor is None
        self._distributor = Distributor(self._commWorld)
        return None

    def start(self, initImpl=None):
        """
        Enter receive-loop as provided by distributor.
        """
        self._distributor.start(doExit=False, initImpl=initImpl)


def _pub(x):
    return ray.cloudpickle.dumps((getIP(), ray.put(x)))


def _ray_publish(id, distributor):
    """
    Return ray ObjRef for obj to be used in ray.
    """
    vals = distributor.publishParts(id, "larray", _pub)
    return [ray.cloudpickle.loads(x) for x in vals]


def _ray_get(x):
    return ray.get(x)


class RayRunner:
    """
    Using ray to launch ranks by using ray actors.
    """

    def __init__(self, initImpl=None):
        """
        Initalize our (SPMD) actors, one per node in ray cluster and make them
        connect through MPI.
        Controller (calling process) gets connection config and then
        passes it to init function on each actor.
        """
        self.publish = _ray_publish
        self.get = _ray_get
        self._actors = {}
        self._init(initImpl)

    def fini(self):
        """
        Finalize Ray Actors: killing actor processes.
        """
        if ray.is_initialized():
            print("Killing actors")
            if self._handles:
                ray.get(self._handles)
            if self._actors:
                for a in self._actors.values():
                    ray.kill(a)

    def _init(self, initImpl=None):
        if not ray.is_initialized():
            ray.init(address="auto")
        ppn = int(getenv("CW4H_PPN", default="1"))
        assert ppn >= 1
        my_ip = getIP()
        # first create one actor per node in the ray cluster
        for node in ray.cluster_resources():
            if "node" in node:
                name = node.split(":")[-1]
                _ppn = ppn - 1 if name == my_ip else ppn
                if _ppn >= 1:
                    for i in range(_ppn):
                        self._actors[f"{name}{i}"] = RayActor.options(resources={node: 1}).remote(
                            name
                        )  # runtime_env={"I_MPI_FABRICS": "ofi"}
        nw = len(self._actors)  # number of workers
        self.comm = MPI.COMM_SELF
        # Get Port for MPI connections
        port = MPI.Open_port(MPI.INFO_NULL)
        # make all actors connect
        x = [a.connect.remote(port, nw) for a in self._actors.values()]
        for i in range(nw):
            # connect to next worker (collectively)
            intercomm = self.comm.Accept(port)
            # merge communicators
            self.comm = intercomm.Merge(0)
            intercomm.Disconnect()
        # wait for connections to be established
        _ = ray.get(x)
        self._handles = [a.start.remote(initImpl) for a in self._actors.values()]
        print("All actors started", flush=True)
        # setup our distributor
        self.distributor = Distributor(self.comm)

        return self


def init(initImpl=None):
    """
    Return a Ray Runner.
    Ray runner will launch actors and connect them throuh MPI.
    """
    return RayRunner(initImpl)
