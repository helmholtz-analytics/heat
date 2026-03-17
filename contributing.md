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
    L --> M["Test locally with `mpirun -n PROCESSES`"]
    M --> N["Commit and push to your branch"]
    N --> O["Create (Draft) Pull Request and let maintainers know"]
    O --> P["Check for failures in CI (tests, code coverage)"]

    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P heatStyle;
```

## Getting Started
All contributions must start with an [Issue](https://github.com/helmholtz-analytics/heat/issues/new/choose).

* **Assignment:** Before writing code, pick an issue and comment to let the maintainers know you are interested. **Wait for assignment**; a bot will create a dedicated branch for you once you are assigned.

## Environment Setup
To ensure consistency across the project, we use a unified technical setup.

Follow the [New Contributors section of the Quick Start](quick_start.md#new-contributors) to set up your environment and install the mandatory `pre-commit` hooks.

## Developing & Testing
* **Branching:** Use the specific branch created for you by the project bot.
* **Testing:** Heat is a distributed framework; all code must be verified in parallel. Run the suite with: `mpirun -n <PROCESSES> python -m unittest`.
* **Commits:** We prefer a clean, logical history. Use `git rebase -i main` to tidy your commits before pushing.

## Stylistic Guidelines
* **Python Standards:** The pre-commit hook will enforce [PEP 8](https://www.python.org/dev/peps/pep-0008/) compliance.
* **Imports:** Use `import heat as ht` and `import numpy as np`.
* **Documentation:** All functions must follow the [Heat docstring standard](https://github.com/helmholtz-analytics/heat/blob/main/doc/source/documentation_howto.rst).

## LLM and AI Usage

We embrace the use of LLMs only if they save our time, where `total_time = development_time + review_time`.

### Instructions for AI and Contributors
If you are using an LLM to generate or review code for Heat, be aware of the following project-specific limitations:

* **Distributed logic:** LLMs consistently struggle with memory-distributed algorithms and MPI communication primitives. Make sure the correctness tests (result comparison to equivalent numpy, scipy, scikit-learn functionality) pass in parallel as well, not just on a single process.
* **Kernel implementation:** Do not reimplement kernels from scratch. Heat is designed to rely on **PyTorch's optimized kernels**; AI suggestions that attempt to bypass PyTorch are usually inefficient and will be rejected.
* **Incremental reviews:** When using an LLM to incorporate PR review feedback, make sure the model does not rewrite entire functions from scratch. "Total rewrites" make follow-up reviews excruciating.

But also:
* **Go for it!** LLMs are great for things like:
    * Code vectorization and refactoring
    * Documentation and docstring generation
    * Understanding and explaining the existing codebase
    * Drafting tests, etc.
    * creating flowcharts and diagrams to illustrate concepts, like this one:

```mermaid
    graph TD
    %% Define Brand Style
    classDef heatStyle fill:#FFE9D6,stroke:#F07E26,stroke-width:2px,color:#56656B;
    classDef automation fill:#56656B,stroke:#56656B,stroke-width:2px,color:#FFFFFF;

    Start(["Using an LLM for Heat?"]) --> Choice{Task Type}

    %% The "Go For It" Path
    Choice -- "Refactoring / Docs / Tests" --> Good["Go for it!"]
    Good --> G1["Vectorization & Refactoring"]
    Good --> G2["Docstring Generation"]
    Good --> G3["Drafting Unit Tests"]
    Good --> G4["Explaining Codebase"]

    %% The "Caution" Path
    Choice -- "New Logic / PR Reviews" --> Caution["Exercise Caution"]
    Caution --> C1[["Distributed & MPI Logic"]]
    Caution --> C2[["Process-local Pytorch ops"]]
    Caution --> C3[["Incremental Reviews"]]

    %% Constraints
    C1 --> R1["Verify results in parallel vs. NumPy/SciPy"]
    C2 --> R2["Do NOT reimplement ops in pure Python, use PyTorch optimized kernels!"]
    C3 --> R3["Avoid 'Total Rewrites' after receiving feedback"]

    %% Conclusion
    R1 & R2 & R3 & G1 & G2 & G3 & G4 --> Final(["Verify & Push"])

    %% Apply Classes
    class Start,Choice,Good,Caution,G1,G2,G3,G4,C1,C2,C3,R1,R2,R3,Final heatStyle;
    class C1,C2,C3 automation;
```

## Next Steps

Once your PR is open, monitor the CI status. If the red cross appears, check the logs to resolve failures before requesting a final manual review. Notably:

- `codebase` CI runs on CUDA and ROCm runners and checks for test failures on CPU and GPU, and code coverage.

- the `codecov` CI  will fail if the coverage drops below the required threshold.
