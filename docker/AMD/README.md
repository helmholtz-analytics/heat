# Not functional yet


### AMD/ROCM Specific Information

Template found in: [ROCm Docker Github Link](github.com/ROCm/ROCm-docker/blob/master/rocm-terminal/Dockerile)

Check out the following [installation guide](https://github.com/ROCm/ROCm-docker/blob/master/quick-start.md)

### AMD Docker file sources
#### Basic Image
[Radeon Repository](https://repo.radeon.com/rocm/manylinux/)

#### Pytorch Image
[ROCm docker hub](https://hub.docker.com/r/rocm/pytorch)


#### General Things I realized
You might run into an error that no space is left on your device when trying to build the pytorch container. Within Docker Desktop you can specify a limit for file sizes. Since containers can take up to 100GB, it might take some free storage.

