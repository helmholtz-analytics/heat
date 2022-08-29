
# Benchmarking with Pytest

The purpose of Benchmarking with pytest is to provide a scaling pipeline which can assist in rapid detection of performance degradation in HeAT via a CI/CD system. The basis of this
project is based on the three related pilars to performance measurement. The first is that 
a given benchmark can be represented in a function call, This function call can then be tested
against various forms of datasets as parametes using a mature python testing library called pytest.

## CI overview

The  `benchmarking-test` workflow calls the script `pytest-benchmarks/run_benchmarks.sh`. so manually triggering
the script is also possible. 





## run_benchmarks.sh

```python
#calls benchmark for all the pytest encapsulated workloads in folder
pytest --benchmark-json

#Support for passing working directory as flags is being added for manual activation 

run_benchmarks -d 'dirto/localdata-repository'
```

## aggregate.py

```python
#responsible for parsing generated results and create/update aggregate.
#json of the specified benchmark 

python aggregate.py

```

## buildlist.py

```python
#responsible for collecting results location and their names for ginkgo performance explorer

./build-list . > list.json


```
