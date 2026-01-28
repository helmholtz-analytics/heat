# Releasing a new Heat version

These are instructions for Heat release management, including both automated and manual processes.

### Table of Contents
- [Automated Release Schedule](#automated-release-schedule)
- [Major or minor release](#major-or-minor-release)
- [Patch release](#patch-release)
- [conda-forge build](#conda-forge-build)

## Automated Release Schedule

Heat follows a bi-annual release schedule with automated preparation workflows:

- **Spring Release**: Target around the end of March or before the Easter break
- **Fall Release**: Target around the end of September

### Automated Timeline

The release process is largely automated with the following schedule:

#### 4 Weeks Before Release (March 1st / December 1st)
**Automated workflows create issues for:**
- NEP 29 compliance check and bug report template update
- Release highlights selection
- CITATION.cff update
- PR merge decisions
- Blog post drafting
- All open PRs are labeled with `pr-talk` for discussion

#### 2 Weeks Before Release
**Code freeze begins:**
- Automated code freeze issue is created
- Only critical bug fixes accepted
- Release preparation workflow should be triggered
- All open PRs are labeled and notified of code freeze

#### 1 Week Before Release
**Final preparations:**
- Release notes finalization issue created
- Blog post review issue created
- All release materials should be complete

### Manual Override

You can manually trigger these workflows at any time using the workflow dispatch feature in GitHub Actions:

- [Release Schedule Workflow](https://github.com/helmholtz-analytics/heat/actions/workflows/release-schedule.yml)
- [Code Freeze Workflow](https://github.com/helmholtz-analytics/heat/actions/workflows/release-code-freeze.yml)
- [Release Notes Finalization](https://github.com/helmholtz-analytics/heat/actions/workflows/release-notes-finalization.yml)

### Labels and Organization

The automated workflow uses several labels to organize release preparation:

- `release-prep` - All automated release preparation issues
- `pr-talk` - PRs that need discussion for release inclusion
- `code-freeze-review` - PRs under code freeze review
- `compliance`, `highlights`, `citation`, `blog-post` - Specific task types
- `high-priority`, `critical` - Priority levels

### Integration with Manual Process

The automated workflow handles the **preparation phase** of releases. The actual release creation, testing, and publication remain manual processes that require human oversight and follow the existing procedures below.

### Major or minor release

(e.g. 1.4 --> 1.5, or 1.5 --> 2.0)

In the following, we assume we are about to release Heat v1.5.0.

**PRE-REQUISITES:**

- You need [PyPi](https://pypi.org/), [Test.PyPi](https://test.pypi.org/) account
- All intended PRs are merged, all tests have passed, and the `main` branch is ready for release.
- **If following automated schedule**: All automated issues have been addressed (NEP 29 compliance, highlights selected, CITATION updated, etc.)

**Create Pre-release Branch**

1. Got to [this GH Action](https://github.com/helmholtz-analytics/heat/actions/workflows/release-prep.yml) and start a new manual workflow.

    a. `Use workflow from` should always be `main`.

    b. Set the version number to the next release (1.5.0) in this case.

    c. Because this is a major or minor release, the base branch should be `main`.

    d. Change the title, if you want to give the release a special name.

    e. Run the workflow.

When the workflow is done, you should see two new pull requests. One targeting `main`, the other one targeting `stable`. Both should be created for the same branch, `pre-release/x.y.z`. The new branch should include changes with the new version number on `version.py`, and an up-to-date `CHANGELOG.md`. For now, **ignore the PR targeting `main`**. That PR should only be merged after the release has been merged to `stable`.

2. Ensure that the changes to `version.py` and `CHANGELOG.md` are correct, if not, fix them.

3. If necessary, also update the Requirements section on README.md to reflect the latest version of the dependencies.

4. Update `CITATION.cff` if needed, i.e. add names of non-core contributors (they are included in the Release notes draft you just created).

5. Once the changes are done:
  - Get a reviewers approval.
  - ONLY MERGE THE PR FOR `stable`
  - DO NOT DELETE THE BRANCH AFTERWARDS.
  - Wait for the tests to finish.
  - Squash and merge.

Go to the main repo page, and then to releases (right panel). There should be a draft release with the changes made by the latest release.

6. Draft release notes:

  - Go to the GitHub repo's [Releases](https://github.com/helmholtz-analytics/heat/releases) page.
  - The release notes draft is automated. Click on Edit Draft
  - Select the new tag: v1.5.0. Modify Target branch: `stable`
  - Edit release notes as needed (see older releases)
  - Click on Save **but do not publish yet**

7. On your local machine, fetch all the changes from origin, checkout the `stable` branch.
8. Build wheel in your local `heat/` directory.

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

  - Attach the tar and wheel file genereated on step 8 in the dist folder.
  - Make sure the CHANGELOG.md got updated, if not, call @JuanPedroGHM.
  - Check our [Zenodo page](https://zenodo.org/doi/10.5281/zenodo.2531472) to make sure a DOI was created for the release.

12. On branch `main`, we want to modify the version so that `minor` is increased by 1, and `extension` is "dev". We also want to merge any changes to the changelog, and overall make sure it is up to date with the latest release changes. That is what the second PR is for. In this example we want the version on `main` to be:`1.6.0-dev`. We go to the left over PR, and change the version number accordingly. Make sure to also fix any merge conflicts.

13. Get approval and merge. You're done! Except if you're a conda-forge maintainer, then see [conda-forge build](#conda-forge-build).


### Patch release

(e.g. 1.5.0 --> 1.5.1)

1. Check that all intended branches have been merged to the `stable` branch.

2. Got to [this GH Action](https://github.com/helmholtz-analytics/heat/actions/workflows/release-prep.yml) and start a new manual workflow.

  1. `Use workflow from` should always be `main`.
  2. Set the version number to the next release (1.5.1 in this case).
  3. Because this is a patch release, the base branch should be `stable`.
  4. Change the title, if you want to give the release a special name.
  5. Run the workflow.

When the workflow is done, you should see two new pull requests. One targeting `main`, the other one targeting `stable`. Both should be created for the same branch, `pre-release/x.y.z`. The new branch should include changes with the new version number on `version.py`, and an update `CHANGELOG.md`. For now, **ignore the PR targeting `main`**. That PR should only be merged after the release has been merged to `stable`.

3. Follow steps 2-14 from the [Major or minor release section](#major-or-minor-release).


## conda-forge build
After releasing, the conda-forge automation will create a new PR on https://github.com/conda-forge/heat-feedstock. It's normal if this takes hours. conda-forge maintainers will review the PR and merge it if everything is correct.
  - Changes can be pushed to the PR.
    - Make sure the version number is correct.
    - Make sure the SHA points to the correct PyPI release.
    - Make sure dependencies match.
  - Once the PR is done, wait for the CI checks to finish and merge.
  - Refer to the conda-forge docs if there are any issues: https://conda-forge.org/docs/maintainer/updating_pkgs.html#pushing-to-regro-cf-autotick-bot-branch
