# Releasing a new Heat version

These are basic instructions for internal use. Will be expanded as need arises.

### Major or minor version update
(e.g. 1.2 --> 1.3, or 1.3 --> 2.0)

In the following, we assume we are about to release Heat v1.3.0.

1. Starting from  `main`, create a new release branch, e.g.:
```bash
git checkout main
git pull
git checkout -b release/1.3.x
```

2. Update `heat/core/version.py` like this:
```python
"""This module contains Heat's version information."""

major: int = 1
"""Indicates Heat's main version."""
minor: int = 2 # update to 3
"""Indicates feature extension."""
micro: int = 2 # update to 0
"""Indicates revisions for bugfixes."""
extension: str = "dev" # set to ""
"""Indicates special builds, e.g. for specific hardware."""
```


### Patch release
(e.g. 1.3.1 --> 1.3.2)
