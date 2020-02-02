# This script will be submitted as a batch job
# `STATUS_TOKEN` and `CODECOV_TOKEN` are set in the `.bashrc` file on the machine
# The first one is used to authorize on GitHub to submit to the status API
# The second one is used for uploading the coverage report to codecov
# `NUM_PROC` is set by the GitHub workflow file and can be customized

curl -H "Content-Type: application/json" -H "Authorization: token $STATUS_TOKEN" -X POST -d '{"state": "pending", "description": "GPU Test Status", "context": "continuous-integration/gpu"}' https://api.github.com/repos/helmholtz-analytics/heat/statuses/$SHA

STATUS="success"
echo "GPUS $SLURM_GPUS"

source ~/.virtualenvs/heat/bin/activate
module load Python/3.6.8
module load GCC
module load Intel
module load ParaStationMPI
DEVICE="gpu"
export DEVICE
python -m pip install --user codecov pytest coverage
python -m pip install --user -e .
srun -n $NUM_PROC -G=$NUM_GPU python -m coverage run --source=heat --parallel-mode -m pytest || STATUS="failure"
srun -n $NUM_PROC -G=$NUM_GPU echo "GPUS $SLURM_GPUS"
python -m coverage combine
python -m coverage report
python -m coverage xml
python -m codecov -t $CODECOV_TOKEN

curl -H "Content-Type: application/json" -H "Authorization: token $STATUS_TOKEN" -X POST -d "{\"state\": \"$STATUS\", \"description\": \"GPU Test Status\", \"context\": \"continuous-integration/gpu\"}" https://api.github.com/repos/helmholtz-analytics/heat/statuses/$SHA

cd ~
