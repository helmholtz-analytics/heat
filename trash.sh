#!/bin/bash

uv run pytest tests/utils/data/test_distributed_data.py::TestDatatools && \
mpirun -n 1 uv run pytest tests/utils/data/test_distributed_data.py::TestDatatools && \
mpirun -n 2 uv run pytest tests/utils/data/test_distributed_data.py::TestDatatools && \
mpirun -n 3 uv run pytest tests/utils/data/test_distributed_data.py::TestDatatools && \
mpirun -n 4 uv run pytest tests/utils/data/test_distributed_data.py::TestDatatools && \
mpirun -n 5 uv run pytest tests/utils/data/test_distributed_data.py::TestDatatools
