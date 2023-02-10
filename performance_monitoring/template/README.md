## Templates for Performance-Monitoring

### General

- The file `my_script.py` runs the actual Heat-application that is contained in the function `do_one_measurement`. If the runtime has to be determined for different choices of parameters/splits/etc. this can be done specifying `keys` (see the file for details).
- The files `run_my_script.sh` and `run_my_script_gpu.sh` execute `my_script.py` on CPU and GPU, respectively. They do not need to be modified or accessed.
- Benchmarking is started by
```
bash run_scaling_test_nodes_cpu.sh
```
or similar with `run_scaling_test_procs_cpu.sh`, `run_scaling_test_nodes_gpu.sh`, and `run_balancing_test_cpu.sh`

### Output
The output is printed to `scaling_test_output_cpu.txt` or `scaling_test_output_gpu.txt` (similar for the error-files) by SLURM and the measurements are saved as `results_cpu.txt`/`results_gpu.txt` by `numpy`. Note that the output is not in correct order, i.e. the rows of the table in the results-file are not sorted in ascending order of MPI-processes.

### What can be done with these scripts
- `run_scaling_test_nodes_cpu.sh`: runs the programm on an *increasing number of CPU-nodes* (each fully used). The number of threads per process is fixed.
- `run_scaling_test_nodes_gpu.sh`: runs the programm on an *increasing number of GPU-nodes* (each fully used). The number of threads per process is fixed.
- `run_scaling_test_procs_cpu.sh`: runs the programm on an *increasing number of MPI-processes*. The number of threads per process is fixed.
- `run_balancing_test_cpu.sh` runs the programm for a fixed number of nodes (fully used) with varying number of MPI-processes and threads per process (such that all CPUs are used)

### What has to be done
1. modify `my_script.py` in the way you need
2. in the files `run_scaling...`/`run_balancing...` choose the desired values. (They need to be adapted depending on your system!)
