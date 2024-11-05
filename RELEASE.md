# Releasing a new Heat version

These are basic instructions for internal use. Will be expanded as the need arises.

### Table of Contents
- [GitHub and PyPi release](#github-and-pypi-release)
- [conda-forge build](#conda-forge-build)

### GitHub and PyPi release

**PRE-REQUISITES:**

- You have accounts on [PyPi](https://pypi.org/), [Test.PyPi](https://test.pypi.org/)
- All intended PRs are merged, all tests have passed.

1. Trigger the release preparation workflow by creating a new issue with the title `x.y.z` (stick to the format `x.y.z`), and label it with `release-prep`. You can use the "Release prep"  template (TODO: ADD LINK).

Submit the issue. This will create a new PR with the version number updated in `heat/core/version.py`.


2. Go through the checklist in the PR, get approval, and merge.

3. Draft release notes:

  - Go to the GitHub repo's [Releases](https://github.com/helmholtz-analytics/heat/releases) page.
  - The release notes draft is automated. Click on Edit Draft
  - Select the new tag and modify the target branch according to the release version (e.g. `release/1.5.x`)
  - Edit release notes as needed (see older releases)
  - Click on Save **but do not publish yet**

4. Build wheel in your local `heat/` directory (make sure you are on the release branch, i.e. `release/1.5.x`!)

   ```bash
   rm -f dist/*
   python -m build
   ```

   You might have to install the `build` package first (i.e. with `conda install -c conda-forge build` or `pip install build`)

5. Upload to Test PyPI and verify things look right (you may need to install `twine` first).

    ```bash
     twine upload -r testpypi dist/*
     ```

    `twine` will prompt for your username and password.

6. When you're sure everything works, release to PyPI:

   ```bash
   twine upload dist/*
   ```

7. Go back to the Release Notes draft and publish them. The GitHub release is out!

  - Make sure the CHANGELOG.md got updated, if not, call @JuanPedroGHM.
  - Check our [Zenodo page](https://zenodo.org/doi/10.5281/zenodo.2531472) to make sure a DOI was created for the release.

8. On branch `main`,  we want to modify the version so that `minor` is increased by 1, and  `extension` is "dev".  In this example we want the version on `main` to be:`1.6.0-dev`. We need to create a new branch from `main`:

    ```bash
    git checkout main
    git pull
    git checkout -b workflows/update-version-main
    git branch
    ```

    On branch `workflows/update-version-main`, modify `version.py` so that `minor` is increased by 1, and `extension` is `"dev"`. Commit and push the changes.

9. Create a PR with `main` as the base branch.

10. Get approval and merge. You're done! Except if you're a conda-forge maintainer, then see [conda-forge build](#conda-forge-build).

## conda-forge build
After releasing, the conda-forge automation will create a new PR on https://github.com/conda-forge/heat-feedstock. It's normal if this takes hours. conda-forge maintainers will review the PR and merge it if everything is correct.
  - Changes can be pushed to the PR.
    - Make sure the version number is correct.
    - Make sure the SHA points to the correct PyPI release.
    - Make sure dependencies match.
  - Once the PR is done, wait for the CI checks to finish and merge.
  - Refer to the conda-forge docs if there are any issues: https://conda-forge.org/docs/maintainer/updating_pkgs.html#pushing-to-regro-cf-autotick-bot-branch

## easybuild @ JSC
TBD

## spack
TBD

## Docker
TBD
