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

**Create Pre-release Branch**

1. Got to [this GH Action](https://github.com/helmholtz-analytics/heat/actions/workflows/release-prep.yml) and start a new manual workflow.

    a. `Use workflow from` should always be `main`.
    b. Change the version number to the next release (1.5.0) in this case.
    c. Because this is a major or minor release, the base branch should be `main`.
    d. Change the title, if you want to give the release a special name.
    e. Run the workflow.

When the workflow is done, you should see two new pull requests. One targeting `main`, the other one targeting `stable`. Both should be created for the same branch, `pre-release/x.y.z`. The new branch should include changes with the new version number on `version.py`, and an up-to-date `CHANGELOG.md`. For now, ignore the PR targeting main. That PR should only be merged after the release has been merged to `stable`.

2. Ensure that the changes to `version.py` and `CHANGELOG.md` are correct, if not, fix them.

4. If necessary, also update the Requirements section on README.md to reflect the latest version of the dependencies.

5. Update `CITATION.cff` if needed, i.e. add names of non-core contributors (they are included in the Release notes draft you just created).

6. Once the changes are done, merge the pull request.
  - ONLY MERGE THE PR FOR `stable`
  - DO NOT DELETE THE BRANCH AFTERWARDS.
  - Remember to get a reviewers approval.
  - Wait for the tests to finish.
  - Squash and merge.

Go to the main repo page, and then to releases (right panel). There should be a draft release with the changes made by the latest release.

7. Draft release notes:

  - Go to the GitHub repo's [Releases](https://github.com/helmholtz-analytics/heat/releases) page.
  - The release notes draft is automated. Click on Edit Draft
  - Select the new tag: v1.5.0. Modify Target branch: `release/1.5.x`
  - Edit release notes as needed (see older releases)
  - Click on Save **but do not publish yet**

8. On your local machine, fetch all the changes from origin, checkout the `stable` branch.
9. Build wheel in your local `heat/` directory.

   ```bash
   rm -f dist/*
   python -m build
   ```

   You might have to install the `build` package first (i.e. with `conda install -c conda-forge build` or `pip install build`)

10. Upload to Test PyPI and verify things look right. You need to install `twine` first.

    ```bash
     twine upload -r testpypi dist/*
     ```

    `twine` will prompt for your username and password.

11. When everything works, upload to PyPI:

   ```bash
   twine upload dist/*
   ```

12. Go back to the Release Notes draft and publish them. The new release is out!

  - Make sure the CHANGELOG.md got updated, if not, call @JuanPedroGHM.
  - Check our [Zenodo page](https://zenodo.org/doi/10.5281/zenodo.2531472) to make sure a DOI was created for the release.

13. On branch `main`, we want to modify the version so that `minor` is increased by 1, and `extension` is "dev". We also want to merge any changes to the changelog, and overall make sure it is up to date with the latest release changes. That is what the second PR is for. In this example we want the version on `main` to be:`1.6.0-dev`. We go to the left over PR, and change the version number accordingly. Make sure to also fix any merge conflicts.

14. Get approval and merge. You're done! Except if you're a conda-forge maintainer, then see [conda-forge build](#conda-forge-build).


### Patch release

(e.g. 1.5.0 --> 1.5.1)

1. Check that all intended branches have been merged to the `stable` branch.

2. Got to [this GH Action](https://github.com/helmholtz-analytics/heat/actions/workflows/release-prep.yml) and start a new manual workflow.

  1. Use workflow from should always be main.
  2. Change the version number to the next release (1.5.1) in this case.
  3. Because this is a patch release, the base branch should be `stable`.
  4. Change the title, if you want to give the release a special name.
  5. Run the workflow.

When the workflow is done, you should see two new pull requests. One targeting `main`, the other one targeting `stable`. Both should be created for the same branch, `pre-release/x.y.z`. The new branch should include changes with the new version number on `version.py`, and an update `CHANGELOG.md`. For now, ignore the PR targeting main. That PR should only be merged after the release has been merged to `stable`.

3. Follow steps 3-14 from the [Major or minor release section](#major-or-minor-release).


## conda-forge build
After releasing, the conda-forge automation will create a new PR on https://github.com/conda-forge/heat-feedstock. It's normal if this takes hours. conda-forge maintainers will review the PR and merge it if everything is correct.
  - Changes can be pushed to the PR.
    - Make sure the version number is correct.
    - Make sure the SHA points to the correct PyPI release.
    - Make sure dependencies match.
  - Once the PR is done, wait for the CI checks to finish and merge.
  - Refer to the conda-forge docs if there are any issues: https://conda-forge.org/docs/maintainer/updating_pkgs.html#pushing-to-regro-cf-autotick-bot-branch
