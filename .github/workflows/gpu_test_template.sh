# This script will be used to submit a batch job
# `STATUS_TOKEN` and `CODECOV_TOKEN` are set in the `.bashrc` file on the machine
# The first one is used to authorize on GitHub to submit to the status API
# The second one is used for uploading the coverage report to codecov
# $SHA, $1 and $2 are by the GitHub workflow file and can be customized
# $1 defines the number of processes that will be spawned by srun
# $2 defines the number of GPUS that will be used in this job
# $SHA is the SHA of the current commit to allow to post messages to the GitHub status API

curl -H "Content-Type: application/json" -H "Authorization: token $STATUS_TOKEN" -X POST -d '{"state": "pending", "description": "GPU Test Status", "context": "continuous-integration/gpu"}' https://api.github.com/repos/helmholtz-analytics/heat/statuses/$SHA

STATUS="success"

source ~/.virtualenvs/heat/bin/activate
module load Python/3.6.8
module load GCC
module load Intel
module load ParaStationMPI
python -m pip install --user codecov pytest coverage
python -m pip install --user -e .
export CUDA_VISIBLE_DEVICES=$2
echo "NUMBER GPUS $CUDA_VISIBLE_DEVICES AND $1 AND $2"
# This will make heat run the unittests on GPU
export DEVICE="gpu"
srun -n "$1" echo "import torch; print('NUMBER GPUS 2', torch.cuda.device_count())" | python
python -m coverage combine
python -m coverage report
python -m coverage xml
python -m codecov -t $CODECOV_TOKEN

curl -H "Content-Type: application/json" -H "Authorization: token $STATUS_TOKEN" -X POST -d "{\"state\": \"$STATUS\", \"description\": \"GPU Test Status\", \"context\": \"continuous-integration/gpu\"}" https://api.github.com/repos/helmholtz-analytics/heat/statuses/$SHA

cd ~
