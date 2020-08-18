#!/usr/bin/env python

import argparse
import json
import multiprocessing
import os
import sys

from typing import Dict, List

TEMPLATE = """#!/bin/bash -x
#SBATCH --account=haf
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tasks}
#SBATCH --cpus-per-task={threads}
#SBATCH --time=24:00:00
#SBATCH --mail-user={mail}
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output={output}
#SBATCH --error={error}

export OMP_NUM_THREADS={threads}
cd {workdir}

srun python -u {script} {parameters}
"""

DASK_TEMPLATE = """#!/bin/bash -x
#SBATCH --account=haf
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tasks}
#SBATCH --cpus-per-task={threads}
#SBATCH --time=24:00:00
#SBATCH --mail-user={mail}
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output={output}
#SBATCH --error={error}

export OMP_NUM_THREADS={threads}
cd {workdir}
mkdir {workdir}/dask

srun dask-mpi --scheduler-file {workdir}/dask/scheduler.json --no-nanny --interface ib0 --nthreads {threads} &
sleep 10
python -u {script} {parameters}
rm -r {workdir}/dask
rm -r worker-*

exit 0
"""

JOBSCRIPT_NAME = "{algorithm}-{benchmark}-{kind}-scale-{nodes}-nodes-{tasks}-tasks"
JOBSCRIPT_PATH = os.path.join("{jobscripts}", JOBSCRIPT_NAME) + ".sh"
OUTPUT_PATH = os.path.join("{output_path}", JOBSCRIPT_NAME) + ".out"
ERROR_PATH = os.path.join("{output_path}", JOBSCRIPT_NAME) + ".err"

SKIP = {"file", "benchmarks"}


def jobscripts_from(
    folder: str, args: argparse.Namespace, configuration: Dict[str, object]
) -> List[str]:
    jobscripts = []
    arguments = {
        "algorithm": folder,
        "path": args.path,
        "jobscripts": args.jobscripts,
        "output_path": args.output,
        "threads": args.threads,
        "workdir": os.getcwd(),
        "mail": args.mail,
    }

    parameters = []
    for key, value in configuration.items():
        if key in SKIP:
            continue

        parameters.append("--{}".format(key))
        parameters.append(str(value))

    for script, benchmark in configuration["benchmarks"].items():
        arguments["script"] = os.path.join(args.path, folder, script + ".py")
        arguments["benchmark"] = script
        template = DASK_TEMPLATE if benchmark.get("template") == "dask" else TEMPLATE

        for kind in {"strong", "weak"}:
            arguments["kind"] = kind

            for i, nodes in enumerate(benchmark["nodes"]):
                size = benchmark["size"][kind] if kind == "strong" else benchmark["size"][kind][i]
                parameters.extend(("--file", configuration["file"].format(size=size)))

                arguments["parameters"] = " ".join(parameters)
                arguments["nodes"] = nodes
                arguments["tasks"] = benchmark["tasks"][i]
                arguments["output"] = OUTPUT_PATH.format(**arguments)
                arguments["error"] = ERROR_PATH.format(**arguments)

                jobscript_path = JOBSCRIPT_PATH.format(**arguments)
                jobscripts.append(jobscript_path)

                with open(JOBSCRIPT_PATH.format(**arguments), "w") as handle:
                    handle.write(template.format(**arguments))

                # remove the file entries
                parameters.pop()
                parameters.pop()

    return jobscripts


def generate_jobscripts(args: argparse.Namespace) -> None:
    # create output directory
    os.makedirs(args.output, exist_ok=True)

    # create the jobscripts directory
    os.makedirs(args.jobscripts, exist_ok=True)

    # locate benchmarks
    try:
        benchmark_folders = os.listdir(args.path)
    except FileNotFoundError:
        print("benchmark path", args.path, "does not exists, aborting", file=sys.stderr)
        sys.exit(1)

    for benchmark in benchmark_folders:
        # skip over everything that is not a directory
        if not os.path.isdir(benchmark):
            continue

        try:
            # open the benchmark configuration
            with open(os.path.join(args.path, benchmark, "config.json"), "r") as handle:
                configuration = json.load(handle)

            # write the jobscripts to disk
            jobscripts = jobscripts_from(benchmark, args, configuration)

            # submit the jobs
            if args.submit:
                for jobscript in jobscripts:
                    os.system("sbatch {}".format(jobscript))

        # skip over directories without config.json
        except FileNotFoundError:
            continue


if __name__ == "__main__":
    file_directory = os.path.dirname(os.path.abspath(__file__))

    # set up the arguments parser
    parser = argparse.ArgumentParser(description="Benchmark jobscript generator")
    parser.add_argument(
        "--threads", type=int, help="number of threads", default=multiprocessing.cpu_count() // 2
    )
    parser.add_argument(
        "--path", type=str, help="path to the benchmark folders", default=file_directory
    )
    parser.add_argument(
        "--jobscripts",
        type=str,
        help="path to the jobscripts",
        default=os.path.join(file_directory, "jobscripts"),
    )
    parser.add_argument(
        "--output",
        type=str,
        help="path to the output",
        default=os.path.join(file_directory, "output"),
    )
    parser.add_argument(
        "--submit", help="optional batch submission of the scripts", action="store_true"
    )
    parser.add_argument(
        "--mail", type=str, help="mail address for job status messages", default=None, required=True
    )
    args = parser.parse_args()

    generate_jobscripts(args)
