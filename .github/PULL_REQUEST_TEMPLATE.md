## Due diligence
<!--- Please address the following points before setting your PR "ready for review". --->
- General:
    - [ ] **title** follows `[component] Brief description` and is suitable to appear in the [Release Notes](https://github.com/helmholtz-analytics/heat/releases/latest)
- Distributed & implementation:
    - [ ] no raw input gathering (`resplit_(None)`, `Allgather`, `Gather` of full operands); per-rank memory scales with `lshape` and output `split` is preserved where applicable
    - [ ] unit tests: all valid `split` axes and uneven MPI process counts tested (e.g., 1, 2, 3, and 5 ranks) with assertions on `.split` and `.lshape`
    - [ ] unit tests: multiple dtypes tested
    - [ ] unit tests: tests for correct device behavior (CPU, GPU, MPS)
    - [ ] benchmarks: created for new functionality or improved/maintained for existing functionality (where applicable)
    - [ ] documentation updated where needed
    - [ ] non-general-Heat licensed material specified in REUSE.toml

## Description

<!--- Include a summary of the change/s.
Please also include relevant motivation and context. List any dependencies that are required for this change.
--->

Issue/s resolved: #

## Changes proposed:

-
-
-
-

## Type of change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation / build / CI update

## Memory requirements
<!--- Compare per-rank memory requirements to previous implementation / relevant torch operations if applicable:
- confirm per-rank allocations scale with local shard size (`lshape`) rather than global shape (`shape`)
- in distributed and non-distributed mode
- with `split=None` and `split not None`

This can be done using https://github.com/pythonprofilers/memory_profiler for CPU memory measurements,
GPU measurements can be done with https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html.
These tools only profile the memory used by each process, not the entire function.
--->

## Performance
<!--- Compare performance to previous implementation / relevant torch operations if applicable:
- in distributed and non-distributed mode
- with `split=None` and `split not None`

Python has an embedded profiler: https://docs.python.org/3/library/profile.html
Again, this will only profile the performance on each process. Printing the results with many processes
may be illegible. It may be easiest to save the output of each to a file.
--->

#### Does this change modify the behaviour of other functions? If so, which?
yes / no

<!--- If AI tools were used to author or review this PR, add a blank line below and state:
AI Support 🦾 (<exact model and version>)
--->
