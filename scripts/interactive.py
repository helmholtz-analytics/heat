#!/usr/bin/env python

# start this script as a regular MPI Python script, but enable stdin forwarding, i.e.
# mpirun -np <procs> -stdin all python interactive.py

from mpi4py import MPI

import code
import sys


class HeatInterpreter(code.InteractiveConsole):
    def __init__(self):
        super().__init__()

        # remove the prompts on ranks other than zero
        self.rank = MPI.COMM_WORLD.rank
        if self.rank != 0:
            sys.ps1 = ""
            sys.ps2 = ""

    def interact(self):
        # sync up here to make sure stdin is properly forwarded
        MPI.COMM_WORLD.barrier()
        # remove the banner on all ranks other than zero
        super().interact(banner="" if self.rank != 0 else None, exitmsg="")

    def push(self, line):
        more = super().push(line)
        # run code normaly but sync up afterwards, so that the prompt appear in proper order
        MPI.COMM_WORLD.barrier()

        return more


if __name__ == "__main__":
    # bring up the "heat" interpreter
    interpreter = HeatInterpreter()
    interpreter.interact()
