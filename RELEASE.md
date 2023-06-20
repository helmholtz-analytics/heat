# Releasing a new Heat version

These are basic instructions for internal use. Will be expanded as need arises.

### Major or minor version update
(e.g. 1.2 --> 1.3, or 1.3 --> 2.0)

In the following, we assume we are about to release Heat v1.3.0.

**PRE-REQUISITES:**
- You need [PyPi](https://pypi.org/), [Test.PyPi](https://test.pypi.org/) account
- all intended PRs are merged, all tests have passed, and the `main` branch is ready for release.

1. We will release all new features in the development branch `main`. Branch off  `main` to create a new release branch, e.g.:
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
minor: int = 2 # <-- update to 3
"""Indicates feature extension."""
micro: int = 2 # <-- update to 0
"""Indicates revisions for bugfixes."""
extension: str = "dev" # <-- set to None
"""Indicates special builds, e.g. for specific hardware."""
```

3. Commit and push new `version.py` in `release/1.3.x`

4. Draft release notes:
  - Go to the GitHub repo's [Releases](https://github.com/helmholtz-analytics/heat/releases) page.
  - The release notes draft is automated. Click on Edit Draft
  - Select new tag: v1.3.0. Modify Target branch: `release/1.3.x`
  - Edit release notes as needed (see older releases)
  - Click on Save but do not publish yet

5. Build wheel in your local `heat/` directory, make sure you are on branch `release/1.3.x`.
   ```bash
   rm -f dist/*
   python -m build
   ```
   You might have to install the `build` package first (i.e. with `conda install build` or `pip install build`)
 6. Upload to Test PyPI and verify things look right. You need to install `twine` first.
    ```bash
     twine upload -r testpypi dist/*
     ```
    `twine` will prompt for your username and password.
 7. When everything works, upload to PyPI:
   ```bash
   twine upload dist/*
   ```
 8. Go back to the Release Notes draft and publish them. The new release is out!
 9. Now we want to update `main` to the latest version, and we want the version on `main` to be `1.3.0-dev`.
    ```bash
    git checkout main
    git pull
    git checkout -b workflows/update-version-main
    git merge release/1.3.x --no-ff --no-commit
    ```
    Modify `version.py` so that `extension` is `"dev"`. Commit and push the changes.
 11. Create a PR with `main` as base branch.
 12. Get approval and merge. You're done!


### Patch release
(e.g. 1.3.1 --> 1.3.2)
