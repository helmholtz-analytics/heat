## Due Diligence
<!--- Please address the following points before setting your PR "ready for review".
--->
- General:
    - [ ]  **title** of the PR is suitable to appear in the [Release Notes](https://github.com/helmholtz-analytics/heat/releases/latest)
- Implementation:
    - [ ] unit tests: all split configurations tested
    - [ ] unit tests: multiple dtypes tested
    - [ ] documentation updated where needed

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
<!--
i.e.
- Bug fix (non-breaking change which fixes an issue)
- New feature (non-breaking change which adds functionality)
- Breaking change (fix or feature that would cause existing functionality to not work as expected)
- Documentation update
--->

## Memory requirements
<!--- Compare memory requirements to previous implementation / relevant torch operations if applicable:
- in distributed and non-distributed mode
- with `split=None` and `split not None`

This can be done using https://github.com/pythonprofilers/memory_profiler for CPU memory measurements,
GPU measurements can be done with https://pytorch.org/docs/master/generated/torch.cuda.max_memory_allocated.html.
These tools only profile the memory used by each process, not the entire function.
--->

## Performance
<!--- Compare performance to previous implementation / relevant torch operations if applicable:
- in distributed and non-distributed mode
- with `split=None` and `split not None`

Python has an embedded profiler: https://docs.python.org/3.9/library/profile.html
Again, this will only profile the performance on each process. Printing the results with many processes
may be illegible. It may be easiest to save the output of each to a file.
--->

#### Does this change modify the behaviour of other functions? If so, which?
yes / no
