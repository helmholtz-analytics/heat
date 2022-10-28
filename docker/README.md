# Docker images of HeAT

There is some flexibility to building the Docker images of HeAT.

Firstly, one can build from the released version taken from PyPI. This will either be
the latest release or the version set through the `--build-arg=HEAT_VERSION=1.2.0`
argument.

Secondly one can build a docker image from the GitHub sources, selected through
`--build-arg=INSTALL_TYPE=source`. The default branch to be built is main, other
branches can be specified using `--build-arg=HEAT_BRANCH=branchname`.

## General build

The [Dockerfile](./Dockerfile) guiding the build of the Docker image is located in this
directory. It is typically most convenient to `cd` over here and run the Docker build as:

	$ docker build .

The resulting image (ID) should then be tagged for subsequent upload (push) to a
repository, for example:

	$ docker tag ea0a1040bf8a ghcr.io/helmholtz-analytics/heat:1.2.0_torch1.11_cuda11.5_py3.9
	$ docker push ghcr.io/helmholtz-analytics/heat:1.2.0_torch1.11_cuda11.5_py3.9

Please ensure that you push the same tag that you just created.

## Building for HPC

With HeAT being a native HPC library, one would naturally want to build the container
image also for HPC systems, such as the ones available at [Juelich Supercomputing Centre
(JSC)](https://www.fz-juelich.de/jsc/ "Juelich Supercomputing Centre").

HPC centres may run a choice of Apptainer or Singularity, which may incur limitations to
the flexibility of building images. For instance, the Singularity Image Builder (SIB)
does not work with the arguments mentioned above, such that these will have to be
avoided.

However, SIB is capable of using just about any available Docker image from any
registry, such that a specific Singularity image can be built by simply referencing the
available image. SIB is thus used as a conversion tool.

A simple `Dockerfile` (in addition to the one above) to be used with SIB could look like
this:

	FROM ghcr.io/helmholtz-analytics/heat:1.2.0_torch1.11_cuda11.5_py3.9

The invocation to build the image would be:

	$ sib upload ./Dockerfile heat_1.2.0_torch.11_cuda11.5_py3.9
	$ sib build --recipe-name heat_1.2.0_torch.11_cuda11.5_py3.9
	$ sib download --recipe-name heat_1.2.0_torch.11_cuda11.5_py3.9

## Running on HPC

	$ singularity run --nv heat_1.2.0_torch.11_cuda11.5_py3.9.sif /bin/bash
	$ python
	Python 3.8.13 (default, Mar 28 2022, 11:38:47)
	[GCC 7.5.0] :: Anaconda, Inc. on linux
	Type "help", "copyright", "credits" or "license" for more information.
	>>> import heat as ht
	...

The `--nv` argument to `singularity`enables NVidia GPU support, which is desired for
HeAT.
