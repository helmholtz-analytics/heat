#!/usr/bin/env bash
echo "GITHUB var ${SHA}"
echo "secret var ${TEST}"
curl -H "Content-Type: application/json" -H "Authorization: token ${TOKEN}" -X POST -d '{"state": "pending", "description": "GPU Test Status", "context": "continuous-integration/gpu"}' https://api.github.com/repos/helmholtz-analytics/heat/statuses/${SHA}
{
source ~/.virtualenvs/heat/bin/activate
module load Python/3.6.8
module load GCC
module load Intel
module load ParaStationMPI
python -m pip install --user codecov pytest coverage
python -m pip install --user -e .
python -m coverage run --source=heat -m pytest
python -m coverage report
curl -H "Content-Type: application/json" -H "Authorization: token ${TOKEN}" -X POST -d '{"state": "success", "description": "GPU Test Status", "context": "continuous-integration/gpu"}' https://api.github.com/repos/helmholtz-analytics/heat/statuses/${SHA}
} || {
curl -H "Content-Type: application/json" -H "Authorization: token ${TOKEN}" -X POST -d '{"state": "failure", "description": "GPU Test Status", "context": "continuous-integration/gpu"}' https://api.github.com/repos/helmholtz-analytics/heat/statuses/${SHA}
}
