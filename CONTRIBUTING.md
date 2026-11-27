# Contributing to Heat

Thank you for your interest in contributing. To maintain project quality and support our distributed (no pun intended!) team, please follow this structured workflow.

---

## Contribution Workflow

```mermaid
graph TD
    %% Define Brand Style
    classDef heatStyle fill:#FFE9D6,stroke:#F07E26,stroke-width:2px,color:#56656B;
    A(["Found a bug?"]) --> B["Write an Issue"]
    C(["Need functionality?"]) --> B
    D["Want to contribute code?"] --> E["Comment on Issue and wait for assignment"]
    E --> F[["Branch created on assignment"]]

    F -- "Core team / Students (Internal)" --> G["Clone Main Repository"]
    F -- "External Contributors" --> H["Fork & Clone your Fork"]

    G --> I{"Environment Setup"}
    H --> I
    I -- "Conda / Mamba" --> J[/See Quick Start: Method A/]
    I -- "Standard Pip" --> K[/See Quick Start: Method B/]

    J --> L(["Check out your branch and start coding!"])
    K --> L
    L --> M["Test locally with various numbers of tasks using `mpirun -n <PROCESSES> pytest`"]
    M --> N["Commit and push to your branch"]
    N --> O["Create (Draft) Pull Request and let maintainers know"]

    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P heatStyle;
```

## Getting Started
The best way to start a contribution is to open an [Issue](https://github.com/helmholtz-analytics/heat/issues/new/choose). This way, the maintainers can give you some pointers before you start and hopefully save you some time.

Don't hesitate to reach out, for instance by creating a draft PR if you need help with anything!

## Environment Setup

Follow the [Quick Start](https://heat.readthedocs.io/en/stable/quick_start.html) to set up your development environment and install the mandatory `pre-commit` hooks.

## Developing & Testing
* **Branching:** For any new development, create a new branch from the latest main or, if applicable and possible, use the specific branch created for you by the project bot.
* **Testing:** Heat is a distributed framework; all code must be verified in parallel. You can run the suite e.g. with: `mpirun -n <PROCESSES> pytest`.
If you add new functionality, you also have to add a test for this.
Keep in mind that the code must run on diverse hardware including GPUs. Don't worry if you don't have the resources to test on all necessary devices, this will happen in the continuous integration pipeline once you open the pull request and potential failures can be fixed at that time. If you do have GPUs available, use `HEAT_TEST_USE_DEVICE="gpu" mpirun -n <PROCESSES> pytest` to run the tests there.
* **Stay synchronized:** Make sure to keep your local repository / fork synchronized with the upstream main! Regularly run `git fetch upstream; git merge upstream` if working on your fork or `git pull; git merge main` if working directly on the main repository. This way you can resolve merge conflicts early on and save a lot of work later.

* **Review process:** All contributions will be reviewed and must receive approval before being merged. In order to ensure a smooth merge process, please make your PR easy to review by (1) coding expressively with well chosen names or comments (2) describing the changes you are proposing in sufficient detail and (3) keeping the diff as small as possible, potentially by splitting up the PR if it contains multiple separable features.

*Thank you for your time!* Open source codes like Heat rely on your contributions and we appreciate your effort!

## Stylistic Guidelines
* **Python Standards:** The pre-commit hook will enforce [PEP 8](https://www.python.org/dev/peps/pep-0008/) compliance.
* **Imports:** Use `import heat as ht` and `import numpy as np`.
* **Documentation:** All public functions must follow the [Heat docstring standard](https://heat.readthedocs.io/en/stable/documentation_howto.html).

## LLM and AI Usage

We embrace the use of LLMs and coding agents when they save time across the whole lifecycle, where `total_time = development_time + review_time`. You are responsible for every line you submit—unverified AI-generated code that shifts debugging onto reviewers will be closed.

* **Point your tools to our agent docs:** If you use an AI assistant or coding agent, make sure it reads `AGENTS.md` and `ARCHITECTURE.md` first. These files enforce Heat's memory-scaling rules (no gathering of raw distributed inputs), PyTorch kernel usage, and incremental PR review edits.
* **Verify in parallel:** Always run the test suite across multiple MPI ranks (including uneven counts such as `mpirun -n 3` and `mpirun -n 5`) and compare against NumPy/SciPy/PyTorch before opening a PR.
* **Disclose AI assistance:** If AI tools were used to author or review your PR, add a blank line at the end of the PR description followed by `AI Support 🦾 (<exact model and version>)`.

## Next Steps

Once your PR is open, monitor the CI status. If the red cross appears, check the logs to resolve failures before requesting a final manual review. Notably:

- `codebase` CI runs on CUDA and ROCm runners and checks for test failures on CPU and GPU, and code coverage.

- the `codecov` CI  will fail if the coverage drops below the required threshold.
