# Docker images of HeAT

There is some flexibility to building the Docker images of HeAT.

Firstly, one can build from the released version taken from PyPI. This will either be
the latest release or the version set through the `--build-arg=HEAT_VERSION=1.2.0`
argument.

Secondly one can build a docker image from the GH sources, selected through
`--build-arg=INSTALL_TYPE=source`. The default branch to be built is main, other
branches can be specified using `--build-arg=HEAT_BRANCH=branchname`.
