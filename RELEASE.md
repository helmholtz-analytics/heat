# Releasing a new Heat version

These are basic instructions for internal use. Will be expanded as the need arises.

### Table of Contents
- [Major or minor release](#major-or-minor-release)
- [Patch release](#patch-release)
- [conda-forge build](#conda-forge-build)

### Major or minor release

(e.g. 1.4 --> 1.5, or 1.5 --> 2.0)

In the following, we assume we are about to release Heat v1.5.0.

**PRE-REQUISITES:**

- You need [PyPi](https://pypi.org/), [Test.PyPi](https://test.pypi.org/) account
- All intended PRs are merged, all tests have passed, and the `main` branch is ready for release.

1. We will release all new features in the development branch `main`. Branch off  `main` to create a new release branch, e.g.:

```bash
git checkout main
git pull
git checkout -b release/1.5.x
```

2. Create a new branch off `release/1.5.x` and update the version number in `heat/core/version.py`:

```bash
git checkout release/1.5.x
git pull
git checkout -b workflows/version-update
```



Update `heat/core/version.py` like this:

```python
"""This module contains Heat's version information."""

major: int = 1
"""Indicates Heat's main version."""
minor: int = 4 # <-- update to 5
"""Indicates feature extension."""
micro: int = 2 # <-- update to 0
"""Indicates revisions for bugfixes."""
extension: str = "dev" # <-- set to None
"""Indicates special builds, e.g. for specific hardware."""
```

3. Commit and push new `version.py` in `workflows/version-update`.

4. If necessary, also update the Requirements section on README.md to reflect the latest version of the dependencies.

5. Update `CITATION.cff` if needed, i.e. add names of non-core contributors (they are included in the Release notes draft you just created). Push to `workflows/version-update`.

6. Create a pull request from `workflows/version-update` to `release/1.5.x`

  - Remember to get a reviewers approval.
  - Wait for the tests to finish.
  - Squash and merge.

7. Draft release notes:

  - Go to the GitHub repo's [Releases](https://github.com/helmholtz-analytics/heat/releases) page.
  - The release notes draft is automated. Click on Edit Draft
  - Select the new tag: v1.5.0. Modify Target branch: `release/1.5.x`
  - Edit release notes as needed (see older releases)
  - Click on Save **but do not publish yet**

8. Build wheel in your local `heat/` directory, make sure you are on branch `release/1.5.x`.

   ```bash
   rm -f dist/*
   python -m build
   ```

   You might have to install the `build` package first (i.e. with `conda install -c conda-forge build` or `pip install build`)

9. Upload to Test PyPI and verify things look right. You need to install `twine` first.

    ```bash
     twine upload -r testpypi dist/*
     ```

    `twine` will prompt for your username and password.

10. When everything works, upload to PyPI:

   ```bash
   twine upload dist/*
   ```

11. Go back to the Release Notes draft and publish them. The new release is out!

  - Make sure the CHANGELOG.md got updated, if not, call @JuanPedroGHM.
  - Check our [Zenodo page](https://zenodo.org/doi/10.5281/zenodo.2531472) to make sure a DOI was created for the release.

12. On branch `main`,  we want to modify the version so that `minor` is increased by 1, and  `extension` is "dev".  In this example we want the version on `main` to be:`1.6.0-dev`. We need to create a new branch from `main`:

    ```bash
    git checkout main
    git pull
    git checkout -b workflows/update-version-main
    git branch
    ```

    On branch `workflows/update-version-main`, modify `version.py` so that `minor` is increased by 1, and `extension` is `"dev"`. Commit and push the changes.

13. Create a PR with `main` as the base branch.

14. Get approval and merge. You're done! Except if you're a conda-forge maintainer, then see [conda-forge build](#conda-forge-build).


### Patch release

(e.g. 1.5.0 --> 1.5.1)

1. Check that all intended branches have been merged to the release branch you want to upgrade, in this example `release/1.5.x`. Create a new branch off `release/1.5.x`, e.g.:

```bash
git checkout release/1.5.x
git pull
git checkout -b workflows/version-update
```

2. Update `heat/core/version.py` like this:

```python
"""This module contains Heat's version information."""

major: int = 1
"""Indicates Heat's main version."""
minor: int = 5
"""Indicates feature extension."""
micro: int = 0 # <-- update to 1
"""Indicates revisions for bugfixes."""
extension: str = None
"""Indicates special builds, e.g. for specific hardware."""
```

3. Follow steps 3-14 from the [Major or minor release section](#major-or-minor-release).


## conda-forge build
After releasing, the conda-forge automation will create a new PR on https://github.com/conda-forge/heat-feedstock. It's normal if this takes hours. conda-forge maintainers will review the PR and merge it if everything is correct.
  - Changes can be pushed to the PR.
    - Make sure the version number is correct.
    - Make sure the SHA points to the correct PyPI release.
    - Make sure dependencies match.
  - Once the PR is done, wait for the CI checks to finish and merge.
  - Refer to the conda-forge docs if there are any issues: https://conda-forge.org/docs/maintainer/updating_pkgs.html#pushing-to-regro-cf-autotick-bot-branch
