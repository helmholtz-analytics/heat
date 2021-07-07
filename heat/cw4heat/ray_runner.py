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
from ray.services import get_node_ip_address as getIP
from .distributor import Distributor
import os

_actors = {}


@ray.remote
class RayActor:
    """
    A ray actor which connects to other actors and controller through MPI.
    """

    def __init__(self, node):
        self.node = node
        self._commWorld = MPI.COMM_SELF
        self._distributor = None
        print("Actor up", flush=True)

    def connect(self, port, nWorkers):
        """
        Let nWorkers-many processes connect to controller process.
        """
        print("Actor connecting", flush=True)
        # workers go here
        # initial connect
        intercomm = self._commWorld.Connect(port)
        # merge communicator
        self._commWorld = intercomm.Merge(1)
        intercomm.Disconnect()
        rank = self._commWorld.rank
        print(f"Yey, rank {rank} connected!")
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
        print("actor.start", self._distributor, flush=True)
        self._distributor.start(doExit=False, initImpl=initImpl)
        print("Actor done!")


def _initActors(initImpl=None):
    """
    Initalize our (SPMD) actors, one per node in ray cluster and make them
    connect through MPI.
    Controller (calling process) gets connection config and then
    passes it to init function on each actor.
    """
    global _actors
    if not ray.is_initialized():
        ray.init(address="auto")
    # first create one actor per node in the ray cluster
    for node in ray.cluster_resources():
        if "node" in node:
            name = node.split(":")[-1]
            print(os.getpid(), "starting", name, flush=True)
            _actors[name] = RayActor.options(resources={node: 1}).remote(
                name
            )  # runtime_env={"I_MPI_FABRICS": "ofi"}
    nw = len(_actors)  # number of workers
    print(nw, flush=True)
    comm = MPI.COMM_SELF
    # Get Port for MPI connections
    port = MPI.Open_port(MPI.INFO_NULL)
    # make all actors connect
    x = [_actors[a].connect.remote(port, nw) for a in _actors]
    for i in range(nw):
        # connect to next worker (collectively)
        intercomm = comm.Accept(port)
        # merge communicators
        comm = intercomm.Merge(0)
        intercomm.Disconnect()
        print("Connected", i, flush=True)
    # wait for connections to be established
    r = ray.get(x)
    print("All connected", r, _actors, flush=True)
    x = [_actors[a].start.remote(initImpl) for a in _actors]
    print("All started", flush=True)
    # setup our distributor
    return (comm, Distributor(comm), x)


def _finiActors():
    """
    Finalize Ray Actors: killing actor processes.
    """
    global _actors
    if ray.is_initialized():
        print("Killing actors")
        for a in _actors.values():
            ray.kill(a)


init = _initActors
fini = _finiActors
