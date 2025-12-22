# Writing Heat Documentation

Heat’s documentation is now built entirely with **MkDocs** and the Material theme, with API reference pages generated from the source code.  This guide explains how to build the docs locally and how to write consistent, high‑quality docstrings.

## Prerequisites

The documentation stack consists of:

- MkDocs with the Material theme for the static site.
- pdoc for auto‑generated API reference pages under `doc/api/heat/...`.
- Standard Markdown (plus a few MkDocs/Material extensions such as admonitions and fenced code blocks).

Install the documentation dependencies into your virtual environment:

```bash
pip install -e .
pip install -r doc/requirements.txt
```

Typical requirements include `mkdocs`, `mkdocs-material`, `mkdocstrings-python`, `mkdocs-git-revision-date-localized-plugin`, and `pdoc`.

All MkDocs configuration lives in `mkdocs.yml`, and the Markdown sources are under `doc/`.

## Building the documentation

There are two steps: regenerate the API reference and build the MkDocs site.

  I. From the project root, regenerate the API docs:

   ```bash
   PYTHONPATH=. pdoc --skip-errors --force --output-dir doc/api heat
   ```

   This recreates `doc/api/heat/...` and produces one Markdown page per module, class, and function, including subpackages and tests.

  II. Build or serve the MkDocs site:

   ```bash
   mkdocs serve        # local preview at http://127.0.0.1:8000
   # or
   mkdocs build        # static site output in site/
   ```

MkDocs uses the navigation specified in `mkdocs.yml` to organize tutorials, guides, and the API Reference sidebar, fully replacing the previous Sphinx + autoapi setup.

- The API navigation is maintained manually in `mkdocs.yml`. When new modules are added or removed, update the `API Reference` section so the sidebar matches the generated `doc/api/heat/...` pages.

## Docstring guidelines

Docstrings continue to follow the **NumPy** style, with reStructuredText‑like section headings (Parameters, Returns, Notes, Examples, …), but the surrounding site is now pure Markdown.  pdoc renders these docstrings into the API pages, so clarity and consistency matter.

### Docstring content

- Write clear, concise descriptions that explain behavior and intent, not just types.
- Use type hints for all parameters and return types whenever possible.
- Cross‑reference major Heat classes (`DNDarray`, `Communication`, `Device`, `data_type`) by importing them in the module and referring to them by name in the text (pdoc will link them where possible).
- In narrative text, refer to Heat arrays as “array” and reserve “tensor” for PyTorch tensors.
- Use code formatting for function names, parameters, literals, and exceptions, for example `add`, `dtype`, `None`, `True`, `NotImplementedError`.
- Use math formatting (LaTeX inside `\( … \)` or `\[ … \]`) for formulas in docstrings or Markdown pages.

### Docstring format

A standard function should look like this:

```python
def foo(x: DNDarray, y: str, k: int = 0) -> DNDarray:
    """
    One-line summary of what the function does.

    A longer description can explain details, edge cases, or provide a short narrative
    about how the function should be used.

    Parameters
    ----------
    x : DNDarray
        Description of x.
    y : str
        Description of y. Can be either 'a', 'b' or 'c'.
    k : int, optional
        Description of k. Default is 0.

    Notes
    -----
    Additional background, algorithmic details, or caveats.

    References
    ----------
    [1] Webpage or paper reference.
    [2] Additional literature as needed.

    Warnings
    --------
    Important usage warnings or behavioral quirks.

    Raises
    ------
    ValueError
        Describe when this is raised.
    RuntimeError
        Describe when this is raised.

    See Also
    --------
    other_function : Brief explanation of the relationship.

    Examples
    --------
    >>> import heat as ht
    >>> T = ht.array([[1, 2], [3, 4]], dtype=ht.float32)
    >>> ht.add(T, 2)
    DNDarray([[3., 4.],
              [5., 6.]], dtype=ht.float32, device=cpu:0)
    """
```

For classes, place the docstring directly under the `class` definition rather than in `__init__`, so that initialization parameters and attributes are captured correctly.

### Parameter and example conventions

- Define default values in the **Parameters** section (for example, “Default is 0”) rather than in separate notes.
- Shape information goes at the end of the parameter description, e.g. `Shape = (x, y, ...)`.
- For classes, describe initialization parameters in an **Attributes** section.
- When listing alternative types, separate them with `or`, not commas (for example, `int or None`).
- For complex type hints (`Union`, `List`, `Tuple`, etc.), follow the standard `typing` module conventions.

Examples:

- Group related examples into a single **Examples** block; use blank lines only when there is a clear distinction between examples.
- Do not add a colon after the **Examples** heading.
- Avoid inline comments inside doctest blocks; move explanatory text into **Notes** instead.

## Writing Markdown pages

All narrative documentation (tutorials, guides, case studies) is now written in Markdown under `doc/`.

- Use standard Markdown headings (`#`, `##`, `###`) and fenced code blocks (```python).
- Prefer Markdown links for internal navigation, for example `[API Reference](../api/heat/core/arithmetics.md)`, with correct relative paths from the current page.
- HTML is allowed for advanced layout (tooltips, custom cards), but ensure all tags are properly closed and paths remain relative so they work on Read the Docs.
- Images should use repository‑relative paths under `doc/`, not raw GitHub URLs, to keep builds portable.

By keeping docstrings NumPy‑style and Markdown pages consistent with these guidelines, Heat’s MkDocs site remains readable, maintainable, and fully synchronized with the source code and API surface.
