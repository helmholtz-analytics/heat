#!/bin/bash
### As the name suggests, this script is meant for the HeAT developers to quickly build a new Docker image with the specified HeAT version, and Pytorch IMG version. The arguments TORCH_VERSION, CUDA_VERSION, and PYTHON_VERSION should indicated the versions of thouse libraries found on the pytorch image from nvidia, and used only to create the image tag.
# If you want to upload the image to the github package registry, use the '--upload' option. You need be logged in to the registry. Instructions here: https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry

GHCR_UPLOAD=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --heat-version)
      HEAT_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    --pytorch-img)
      PYTORCH_IMG="$2"
      shift # past argument
      shift # past value
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    --cuda-version)
      CUDA_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    --upload)
      GHCR_UPLOAD=true
      shift
      shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
  esac
done

echo "HEAT_VERSION=$HEAT_VERSION"
echo "PYTORCH_IMG=$PYTORCH_IMG"
echo "TORCH_VERSION=$TORCH_VERSION"
echo "CUDA_VERSION=$CUDA_VERSION"
echo "PYTHON_VERSION=$PYTHON_VERSION"


ghcr_tag="ghcr.io/helmholtz-analytics/heat:${HEAT_VERSION}_torch${TORCH_VERSION}_cu${CUDA_VERSION}_py${PYTHON_VERSION}"

echo "Building image $ghcr_tag"

docker build --file ../Dockerfile.release \
              --build-arg HEAT_VERSION=$HEAT_VERSION \
              --build-arg PYTORCH_IMG=$PYTORCH_IMG \
              --tag $ghcr_tag \
              .

if [ $GHCR_UPLOAD = true ]; then
  echo "Push image"
  echo "You might need to log in into ghcr.io (https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry)"
  docker push $ghcr_tag
fi
