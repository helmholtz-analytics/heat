# Docker images of Heat

There is some flexibility to building the Docker images of Heat.

Firstly, one can build from the released version taken from PyPI. This will either be
the latest release or the version set through the `--build-arg=HEAT_VERSION=X.Y.Z`
argument.

Secondly one can build a docker image from the GitHub sources, selected through
`--build-arg=INSTALL_TYPE=source`. The default branch to be built is main, other
branches can be specified using `--build-arg=HEAT_BRANCH=<branch-name>`.

## General build

### Docker

The [Dockerfile](./Dockerfile) guiding the build of the Docker image is located in this
directory. It is typically most convenient to `cd` over here and run the Docker build as:

```console
$ docker build --build-args HEAT_VERSION=X.Y.Z --PYTORCH_IMG=<nvcr-tag> -t heat .
```

We also offer prebuilt images in our [Package registry](https://github.com/helmholtz-analytics/heat/pkgs/container/heat) from which you can pull existing images:

```console
$ docker pull ghcr.io/helmholtz-analytics/heat:<version-tag>
```

### Building for HPC

With Heat being a native HPC library, one would naturally want to build the container
image also for HPC systems, such as the ones available at [Jülich Supercomputing Centre
(JSC)](https://www.fz-juelich.de/jsc/ "Juelich Supercomputing Centre"). We show two ways to convert the existing images from the registry into singularity containers.

#### Apptainer (formerly singularity)

To use one of the existing images from our registry:

	$ apptainer build heat.sif docker://ghcr.io/helmholtz-analytics/heat:<version-tag>

Building the image can require root access in some systems. If that is the case, we recommend building the image on a local machine, and then upload it to the desired HPC system.

If you see an error indicating that there is not enough space, use the --tmpdir flag of the build command. [Apptainer docs](https://apptainer.org/docs/user/latest/build_a_container.html)

#### SIB (Singularity Image Builder) for Apptainer images

A simple `Dockerfile` (in addition to the one above) to be used with SIB could look like
this:

	FROM ghcr.io/helmholtz-analytics/heat:<version-tag>

The invocation to build the image would be:

	$ sib upload ./Dockerfile heat
	$ sib build --recipe-name heat
	$ sib download --recipe-name heat

However, SIB is capable of using just about any available Docker image from any
registry, such that a specific Singularity image can be built by simply referencing the
available image. SIB is thus used as a conversion tool.

## Running on HPC

	$ apptainer run --nv heat /bin/bash
	$ python
	Python 3.8.13 (default, Mar 28 2022, 11:38:47)
	[GCC 7.5.0] :: Anaconda, Inc. on linux
	Type "help", "copyright", "credits" or "license" for more information.
	>>> import heat as ht
	...

The `--nv` argument to `apptainer` enables NVidia GPU support, which is desired for
Heat.

### Multi-node example

The following file can be used as an example to use the apptainer file together with SLURM, which allows heat to work in a multi-node environment.

```bash
#!/bin/bash
#SBATCH --time 0:10:00
#SBATCH --nodes 2
#SBATCH --tasks-per-node 2

...

srun --mpi="pmi2" apptainer exec --nv heat_1.2.0_torch.11_cuda11.5_py3.9.sif bash -c "cd ~/code/heat/examples/lasso; python demo.py"
```
