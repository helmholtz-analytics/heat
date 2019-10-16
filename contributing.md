## Contributing to HeAT

*These contribution guidelines are inspired by NumPy's contribution guidelines of the SciPy community.*

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

* Add the original HeAT repository as your upstream:

```
git remote add upstream https://github.com/helmholtz-analytics/heat.git
```

* Now, `git remote -v` will show two remote repositories named:
    * `upstream`, which refers to the main HeAT repository
    * `origin`, which refers to your personal fork of HeAT

#### Developing Contributions

* Pull the latest changes from upstream:

```
git checkout master
git pull upstream master
```

* Create a branch for the feature you want to work on. Since the branch name will appear in the merge message, use a sensible name. The naming scheme is as follows `<KIND-OF-CONTRIBUTION/<ISSUE-NR>-<NAME>`, where the kind of the contribution should be *features* for an entirely new feature, *bug* for, well, a bug and *enhancement* for things like performance optimizations for example. Please make sure that *NAME* very briefly summarizes the content of your contribution.

```
git checkout -b features/123-boolean-operators
```

* Commit locally as you progress (`git add` and `git commit`) Use a properly formatted commit message, write tests that fail before your change and pass afterward, run all the tests locally and in parallel for different process counts (`mpirun -np <PROCESSES>`). Be sure to document any changed behavior in docstrings, keeping to HeAT's docstring standard (explained below).

#### Publishing your Contributions

* Push your changes back to your fork on GitHub:

```
git push origin features/123-boolean-operators
```

* Enter your GitHub username and password (advanced users can remove this step by connecting to GitHub with SSH.

* Go to GitHub. The new branch will show up with a green Pull Request button. Make sure the title and message are clear, concise, and self- explanatory. Then click the button to submit it.

* If your commit introduces a new feature or changes functionality, please explain your changes. For bug fixes, documentation updates, etc., this is generally not necessary, though if you do not get any reaction, do feel free to ask for review.

#### Review Process

* Reviewers (the other developers and interested community members) will write inline and/or general comments on your Pull Request (PR) to help you improve its implementation, documentation and style. Every single developer working on the project has their code reviewed, and we’ve come to see it as friendly conversation from which we all learn and the overall code quality benefits. Therefore, please don’t let the review discourage you from contributing: its only aim is to improve the quality of project, not to criticize (we are, after all, very grateful for the time you’re donating!).

* To update your PR, make your changes on your local repository, commit, run tests and push to your fork. As soon as those changes are pushed up (to the same branch as before) the PR will update automatically. If you have no idea how to fix the test failures, you may push your changes anyway and ask for help in a PR comment.

* Various continuous integration (CI) services are triggered after each PR update to build the code, run unit tests, measure code coverage and check coding style of your branch. The CI tests must pass before your PR can be merged. If CI fails, you can find out why by clicking on the “failed” icon (red cross) and inspecting the build and test log. To avoid overuse and waste of this resource, test your work locally before committing.

* There might also be a "failed" red cross, if the test coverage (i.e. the test code lines) is not high enough. There might be good reasons for this that should be properly described in the PR message. In most cases however, a sufficient test coverage should be achieved through adequate unit tests.

* A PR must be approved by at least one core team member before merging. Approval means the core team member has carefully reviewed the changes, and the PR is ready for merging.

* If the PR relates to any issues, you can add the text `#<ISSUE-NUMBER>` to insert a link to the original issue and/or another PR. Please do so for all relevant topics known to you.

#### Document Changes

* Make sure to reflect changes in the code in the functions docstring and possible description in the general documentation.

* If your change introduces a deprecation, make sure to discuss this first on GitHub and what the appropriate deprecation strategy is.

#### Divergence between upstream/master and your feature branch

If GitHub indicates that the branch of your PR can no longer be merged automatically, you have to incorporate changes that have been made since you started into your branch. Our recommended way to do this is to rebase on master.

## Guidelines

* All code should have tests (see test coverage below for more details).

* All code should be documented.

* No changes are ever merged without review and approval by a core team member. Please ask politely on the PR if you get no response to your pull request within a week.

## Stylistic Guidelines

* Set up your editor to follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) (remove trailing white space, no tabs, etc.).

* Use the following import conventions:
    * `import heat as ht`
    * `import numpy as np`
    * Have Python standard library and third-party dependencies imported before HeAT modules

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
