#!/bin/bash -x
#SBATCH --account=haf
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:000

curl -H "Content-Type: application/json" -H "Authorization: token $STATUS_TOKEN" -X POST -d '{"state": "pending", "description": "GPU Test Status", "context": "continuous-integration/gpu"}' https://api.github.com/repos/helmholtz-analytics/heat/statuses/$SHA

STATUS="success"

source ~/.virtualenvs/heat/bin/activate
module load Python/3.6.8
module load GCC
module load Intel
module load ParaStationMPI
DEVICE="gpu"
export DEVICE
python -m pip install --user codecov pytest coverage
python -m pip install --user -e .
srun -n $NUM_PROC python -m coverage run --source=heat --parallel-mode -m pytest || STATUS="failure"
python -m coverage combine
python -m coverage report
python -m coverage xml
python -m codecov -t $CODECOV_TOKEN

curl -H "Content-Type: application/json" -H "Authorization: token $STATUS_TOKEN" -X POST -d "{\"state\": \"$STATUS\", \"description\": \"GPU Test Status\", \"context\": \"continuous-integration/gpu\"}" https://api.github.com/repos/helmholtz-analytics/heat/statuses/$SHA

cd ~
