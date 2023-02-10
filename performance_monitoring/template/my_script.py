"""
Template for Performance-monitoring/Benchmarking
"""
import os
import heat as ht
from mpi4py import MPI
import numpy as np

# do not modify the following...
device = os.getenv("HEAT_BENCHMARK_DEVICE")
num_nodes = int(os.getenv("HEAT_BENCHMARK_NUM_NODES"))
num_samples = int(os.getenv("HEAT_BENCHMARK_NUM_SAMPLES"))
num_procs_from_script = int(os.getenv("HEAT_BENCHMARK_NUM_MPI_PROCS"))
num_procs = MPI.COMM_WORLD.Get_size()

if not num_procs_from_script == num_procs:
    raise RuntimeError(
        "Numer of exectuted MPI-processes is not the number specified in the script. Something is wrong..."
    )

if device == "cpu":
    num_threads = int(os.getenv("HEAT_BENCHMARK_NUM_THREADS_PER_MPI_PROC"))
    num_cpus = num_threads * num_procs
    gpu_flag = False
elif device == "gpu":
    gpu_flag = True
else:
    raise RuntimeError("The value of HEAT_BENCHMARK_DEVICE needs to be either 'cpu' or 'gpu'.")

# ...until here:

"""
-------------------------------------------------------------------------------
The following needs to be modified
-------------------------------------------------------------------------------
The list measurement_keys allows to provide a list of different parameter combinations for which
the measurements will be done, e.g. different combinations of splits for the input, different combinations of
parameter choice in the algorithms etc.
"""

measurement_keys = [
    [0, 0],
    [1, 1],
    [1, 0],
    [0, 1],
    [None, 0],
    [None, 1],
    [0, None],
    [1, None],
    [None, None],
]
num_measurements = len(measurement_keys)
measurements = np.zeros(num_measurements)


def do_one_measurement(key):
    """
    Measures the time for one execution of something ... to be customized
    """
    # prepare input data (possibly randomized or depending on the measurement_key)
    A = ht.random.randn(2000, 2000, split=key[0], device=device)
    B = ht.random.randn(2000, 2000, split=key[1], device=device)

    # execute the routine under consideration and measure the time
    MPI.COMM_WORLD.Barrier()
    t0 = MPI.Wtime()
    C = ht.linalg.matmul(A, B)  # might also depend on the measurement_key
    MPI.COMM_WORLD.Barrier()
    t1 = MPI.Wtime()
    # clean up
    del A, B, C
    return t1 - t0


# do not modify the following til the end...
# run the actual measurments
for k in range(num_measurements):
    for s in range(num_samples):
        measurements[k] += do_one_measurement(measurement_keys[k])
    measurements[k] /= s

# print and save the data
if MPI.COMM_WORLD.rank == 0:
    print_measurements = "".join(["\t %2.2e [s]" % m for m in measurements])
    if gpu_flag:
        print(
            "%d \t MPI-processes (GPU, %d \t nodes):" % (num_procs, num_nodes),
            print_measurements,
            "\t (%d samples each)" % num_samples,
        )
        data_to_write = np.hstack([np.asarray([num_procs, num_nodes]), measurements]).reshape(
            [1, -1]
        )
        filename = "results_gpu.txt"
    else:
        print(
            "%d \t MPI-processes (%d \t nodes, %d \t CPUs, %d \t threads per process): "
            % (num_procs, num_nodes, num_cpus, num_threads),
            print_measurements,
            "\t (%d samples each)" % num_samples,
        )
        data_to_write = np.hstack(
            [np.asarray([num_procs, num_nodes, num_cpus, num_threads]), measurements]
        ).reshape([1, -1])
        filename = "results_cpu.txt"
    with open(filename, "a+") as file:
        np.savetxt(file, data_to_write)
