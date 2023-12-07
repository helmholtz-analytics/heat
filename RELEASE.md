# Releasing a new Heat version

These are basic instructions for internal use. Will be expanded as the need arises.

### Major or minor version update

(e.g. 1.2 --> 1.3, or 1.3 --> 2.0)

In the following, we assume we are about to release Heat v1.3.0.

**PRE-REQUISITES:**

- You need [PyPi](https://pypi.org/), [Test.PyPi](https://test.pypi.org/) account
- All intended PRs are merged, all tests have passed, and the `main` branch is ready for release.

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
  - Select the new tag: v1.3.0. Modify Target branch: `release/1.3.x`
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

8. Update conda-forge recipe (Need to be listed as maintainer, either @ClaudiaComito, @mrfh92, @JuanPedroGHM)
  - Go to https://github.com/conda-forge/heat-feedstock
  - A new PR should have been automatically created.
  - Changes can be pushed to the PR.
    - Make sure the version number is correct.
    - Make sure the SHA points to the correct PyPI release.
    - Make sure dependencies match.
  - Once the PR is done, wait for the CI checks to finish and merge.
  - Refer to the conda-forge docs if there are any issues: https://conda-forge.org/docs/maintainer/updating_pkgs.html#pushing-to-regro-cf-autotick-bot-branch

9. Go back to the Release Notes draft and publish them. The new release is out!

  - Make sure the CHANGELOG.md got updated, if not, call @JuanPedroGHM.

10. Now we want to update `main` to include the latest release,  we want to modify the version on main so that `minor` is increased by 1, and  `extension` is "dev".  In this example we want the version on `main` to be:`1.4.0-dev`.

    ```bash
    git checkout main
    git pull
    git checkout -b workflows/update-version-main
    git merge release/1.3.x --no-ff --no-commit
    ```

    Modify `version.py` so that `extension` is `"dev"`. Commit and push the changes.

12. Create a PR with `main` as the base branch.


13. Get approval and merge. You're done!

### Patch release

(e.g. 1.3.0 --> 1.3.1)

1. Check that all intended branches have been merged to the release branch you want to upgrade, in this example `release/1.3.x`. Branch off from `release/1.3.x` and create a new branch:

```bash
git checkout release/1.3.x
git pull
git checkout -b minor-version-update
```

2. Update `heat/core/version.py` like this:

```python
"""This module contains Heat's version information."""

major: int = 1
"""Indicates Heat's main version."""
minor: int = 3
"""Indicates feature extension."""
micro: int = 0 # <-- update to 1
"""Indicates revisions for bugfixes."""
extension: str = None
"""Indicates special builds, e.g. for specific hardware."""
```

3. Commit and push new `version.py` in `minor-version-update`

4. Create a pull request from `minor-version-update` to `release/1.3.x`

  - Remember to get a reviewers approval.
  - Wait for the test to finish.
  - Squash and merge.


5. Draft release notes:

  - Go to the GitHub repo's [Releases](https://github.com/helmholtz-analytics/heat/releases) page.
  - The release notes draft is automated. Click on Edit Draft
  - Select the new tag: `v1.3.1`
  - Edit release notes as needed (see older releases)

6. Build wheel in your local `heat/` directory, make sure you are on branch `release/1.3.x`.

   ```bash
   rm -f dist/*
   python -m build
   ```

   You might have to install the `build` package first (i.e. with `conda install build` or `pip install build`)

7. Upload to Test PyPI and verify things look right. You need to install `twine` first.

    ```bash
     twine upload -r testpypi dist/*
     ```

    `twine` will prompt for your username and password.

    - Look at the testpypi repository and make sure everything is correct (version number, readme, etc.)

8. When everything works, upload to PyPI:

   ```bash
   twine upload dist/*
   ```

9. Update conda-forge recipe (Need to be listed as maintainer, either @ClaudiaComito, @mrfh92, @JuanPedroGHM)
  - Go to https://github.com/conda-forge/heat-feedstock
  - A new PR should have been automatically created.
  - Changes can be pushed to the PR.
    - Make sure the version number is correct.
    - Make sure the SHA points to the correct PyPI release.
    - Make sure dependencies match.
  - Once the PR is done, wait for the CI checks to finish and merge.
  - Refer to the conda-forge docs if there are any issues: https://conda-forge.org/docs/maintainer/updating_pkgs.html#pushing-to-regro-cf-autotick-bot-branch


10. Go back to the Release Notes draft and publish them. The new release is out!
  - Make sure the CHANGELOG.md got updated in the release branch, in this case `release/1.3.x`, if not, call @JuanPedroGHM.

11. Now we want to update `main` to the latest version, and we want the version on `main` to be `1.4.0-dev`.
  - Create a new branch from `release/1.3.x`, for example `merge-latest-release-into-main`
  - Merge `main` into `merge-latest-release-into-main`, resolve conflicts and push.
  - Create a PR from `merge-latest-release-into-main`, base branch must be `main`
  - Make sure the version number in `merge-latest-release-into-main` is correct (i.e., it matches that in `main`).
  - Make sure the CHANGELOG.md in `merge-latest-release-into-main` matches that in the latest release branch, in this example`release/1.3.x`.
  - Get a reviewers approval, wait for the CI checks to pass, and merge.
