## Contributing to Heat

Thank you for your interest in contributing to Heat, we really appreciate your time and effort!

 * If you want to report a bug, or propose a new feature, you can file an [Issue](https://github.com/helmholtz-analytics/heat/issues/new/choose).
 * You can also get in touch with us on [Mattermost](https://mattermost.hzdr.de/signup_user_complete/?id=3sixwk9okpbzpjyfrhen5jpqfo). You can sign up with your GitHub credentials. Once you log in, you can introduce yourself on the `Town Square` channel.
 * To set up your environment for Heat development, follow these [instructions](README.md#Hacking).
 * We strongly recommend getting in touch with the core developers, either here on GitHub (by filing and/or commenting on an Issue) or on [Mattermost](https://mattermost.hzdr.de/signup_user_complete/?id=3sixwk9okpbzpjyfrhen5jpqfo), before starting to work on a contribution. We are a small team and it's good to know who is currently working on what.
 * Our git workflow is described in a lot of detail [below](#developing-contributions).
 * **TL;DR for experts:** (Also check out [Quick Start](quick_start.md#new-contributors))

 1. `git add`, `pre-commit run --all-files` and `git commit` as needed;
 2. `git rebase -i main` to rebase and tidy up your commits;
 3. `git push` to publish to the remote repository.

*The following is based on the SciPy community's [Contributing to NumPy](https://numpy.org/doc/stable/dev/) guidelines.*

#### Getting the Source Code

* Go to [https://github.com/helmholtz-analytics/heat/](https://github.com/helmholtz-analytics/heat/) and click on the “fork” button to create your own copy of the project.

* Clone the repository to your computer by running:

```
git clone https://github.com/<YOUR-USERNAME>/heat.git
```

* Change your working directory to the cloned repository:

```
cd heat
```

* Add the original Heat repository as your upstream:

```
git remote add upstream https://github.com/helmholtz-analytics/heat.git
```

* Now, `git remote -v` will show two remote repositories named:
    * `upstream`, which refers to the main Heat repository
    * `origin`, which refers to your personal fork of Heat

#### Developing Contributions

* Pull the latest changes from upstream:

```
git checkout main
git pull upstream main
```

* Install Heat from the checked out sources with:

```
pip install .[hdf5,netcdf,dev]
```

* The extra `dev` dependency pulls in additional tools to support the enforcement
of coding conventions ([Black](https://github.com/psf/black)) and to support a
pre-commit hook to do the same. In order to fully use this framework, please
also install the pre-commit hook with

```
pre-commit install
````

* **NEW** As of Aug 2023, as soon as an issue is assigned, a branch is created and its name is posted in a comment under the original issue. **Do adopt this branch** for your development, it is guaranteed to have the correct source branch - `release/...` for bug fixes, `main` for new features, docs updates, etc.

* Commit locally as you progress:

```
git add
pre-commit run --all-files
git commit
```

Use a properly formatted commit message, write tests that fail before your change and pass afterward, run all the tests locally and in parallel for different process counts (`mpirun -np <PROCESSES>`). Be sure to document any changed behavior in docstrings, keeping to Heat's [docstring standard](https://github.com/helmholtz-analytics/heat/blob/main/doc/source/documentation_howto.rst).

#### Publishing your Contributions

* Before publishing your changes, you might want to rebase to the main branch and tidy up your list of commits, keeping only the most relevant ones and "fixing up" the others. This is done with interactive rebase or `git rebase -i`. Here's an excellent [tutorial](https://www.atlassian.com/git/tutorials/merging-vs-rebasing). This should only be done **before** pushing anything to the remote repository!

* Push your changes back to your fork on GitHub:

```
git push origin features/123-boolean-operators
```

* Enter your GitHub username and password (advanced users can remove this step by connecting to GitHub with SSH.

* Go to GitHub. The new branch will show up with a green Pull Request button. Make sure the title and message are clear, concise, and self-explanatory. Then click the button to submit it.

* If your commit introduces a new feature or changes functionality, **please explain your changes and the thinking behind them**. This greatly simplifies the review process. For bug fixes, documentation updates, etc., this is generally not necessary, though if you do not get any reaction, do feel free to ask for a review.

* Phrase the PR title as a changelog message and make sure the PR is properly tagged ('enhancement', 'bug', 'ci/cd', 'chore', 'documentation').

#### Review Process

* Reviewers (the other developers and interested community members) will write inline and/or general comments on your Pull Request (PR) to help you improve its implementation, documentation, and style. Every single developer working on the project has their code reviewed, and we’ve come to see it as a friendly conversation from which we all learn and the overall code quality benefits. Therefore, please don’t let the review discourage you from contributing: its only aim is to improve the quality of the project, not to criticize (we are, after all, very grateful for the time you’re donating!).

* To update your PR, make your changes on your local repository, commit, run tests, and push to your fork. As soon as those changes are pushed up (to the same branch as before) the PR will update automatically. If you have no idea how to fix the test failures, you may push your changes anyway and ask for help in a PR comment.

* Various continuous integration (CI) services are triggered after each PR update to build the code, run unit tests, measure code coverage, and check the coding style of your branch. The CI tests must pass before your PR can be merged. If CI fails, you can find out why by clicking on the “failed” icon (red cross) and inspecting the build and test log. To avoid overuse and waste of this resource, test your work locally before committing.

* There might also be a "failed" red cross, if the test coverage (i.e. the test code lines) is not high enough. There might be good reasons for this that should be properly described in the PR message. In most cases however, a sufficient test coverage should be achieved through adequate unit tests.

* A PR must be approved by at least one core team member before merging. Approval means the core team member has carefully reviewed the changes, and the PR is ready for merging.

* If the PR relates to any issues, you can add the text `#<ISSUE-NUMBER>` to insert a link to the original issue and/or another PR. Please do so for all relevant topics known to you.

#### Document Changes

* Make sure to reflect changes in the code in the functions docstring and possible description in the general documentation.

* If your change introduces a deprecation, make sure to discuss this first on GitHub and what the appropriate deprecation strategy is.

#### Divergence between upstream/main and your feature branch

If GitHub indicates that the branch of your PR can no longer be merged automatically, you have to incorporate changes that have been made since you started into your branch. Our recommended way to do this is to rebase on `main`.

## Guidelines

* All code should have tests (see test coverage below for more details).

* All code should be documented in accordance with Heat's [docstring standard](https://github.com/helmholtz-analytics/heat/blob/main/doc/source/documentation_howto.rst).

* No changes are ever merged without review and approval by a core team member. Feel free to ping us on the PR if you get no response to your pull request within a week.

## Stylistic Guidelines

* Set up your editor to follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) (remove trailing white space, no tabs, etc.).

* Use the following import conventions:
    * `import heat as ht`
    * `import numpy as np`
    * Have Python standard library and third-party dependencies imported before Heat modules.

* Sort functions alphabetically in files (leading underscores are ignored).

* Expose only necessary functions to modules via the `__all__` variable.

## Test Coverage

* Pull requests (PRs) that modify code should either have new tests, or modify existing tests to fail before the PR and pass afterwards. You should run the tests before pushing a PR.

* Tests for a module should ideally cover all code in that module, i.e., statement coverage should be at 100%.

* To measure the test coverage, install [codecov](https://github.com/codecov/codecov-python) and then run:

```
mpirun -np <PROCESSES coverage run --source=heat --parallel-mode -m pytest heat/ && \
    coverage combine && \
    coverage report && \
    coverage xml'
```
