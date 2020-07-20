.. role:: orangemarker
.. role:: greymarker
.. role:: bluemarker

Writing HeAT Documentation
==========================

In order to maintain proper, uniform and understandable API documentation of HeAT, a few style guidelines are
enforced. The following sections summarize the key features of HeATs API documentation.

Prerequisites
-------------

The heat documentation is build using Sphinx (Version 3.0.3), with a custom derivation of the Sphinx-RTD-Theme
(defined in `_static/css/custom.css`).
There are three main colors available for formatting:

* :orangemarker:`Orange: RGB(240, 120, 30)`
* :greymarker:`Grey: RGB(90, 105, 110)`
* :bluemarker:`Blue: RGB(0, 90, 160)`

All configurations regarding the documentation build are set in `doc/source/conf.py`.
API Documentation is generated using the `sphinx-autoapi extension <https://sphinx-autoapi.readthedocs.io>`_ . This is
done via custom templates, defined in `source/_templates/autoapi/python`.

Docstring Guidelines
--------------------

Dostrings are written using the NumPy Documentation style (see sphinx-contributions `napoleon
<https://sphinxcontrib-napoleon.readthedocs.io>`_ ).
Apart from that, formatting happens via reStructuredText (reST). For a full reference on reST see `here <https://www
.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_

Docstring Content
^^^^^^^^^^^^^^^^^

* Write clear on concise docstrings, especially in the function and parameter descriptions
* Use type hints for:

  * Parameters
  * Return types

* Cross-referencing of major HeAT classes (``DNDarray``, ``Communication``, ``Device``, ``data_type``)

  * Import major classes directly (e.g.  ``from .dndarray import DNDarray``, ``from .devices import Device``)
  * In descriptions, use cross references when useful (full module path with tilde):  ``:class:`~heat.core.dndarray.DNDarray```
    or ``:function:`~heat.core.arithmetics.add```
  * use ``from __future__ import annotations`` for module internal crossreferencing: see e.g.
    `naive_bayes/gaussianNB.py: partial_fit

* In narrative form always refer to DNDarrays as array and not as tensor. The latter is exclusively reserved for PyTorch tensors.
* Use code-style markdown to annotate functions, parameters or globally defined variables (e.g. ``None``, ``True``, ``NotImplementedError`` etc.) in description texts.
* Math-style markdown can be used to typeset formulas


Docstrings Format
^^^^^^^^^^^^^^^^^

Method Template
    The following example shows the standard formatting of a function docstring ::

        def foo(x: DNDarray, y: str, k: int = 0) -> DNDarray
        """
        A description of the function behaviour and return value (not the type): What does the function do?
        Any additional information can be given here, either in narrative form or in bullet points like such:
         * Item 1 \n
         * Item 2 \n

        Parameters
        -----------
        x : DNDarray
            Parameter desription of x
        y : str
            Parameter description of y. Can be either 'a', 'b' or 'c'
        k : int, optional

        Notes
        -----------
        Notes on the function should be given in the "Notes" section (not in the function description

        References
        -----------
        [1] Webpage references \n
        [2] Paper references.  \n
        [3] Do not use indentations at linebreaks for a reference

        Warnings
        -----------
        Warnings on the function should be given in the "Warnings" section (not in the function description

        Raises
        -----------
        If the function raises any "unexpected" Errors/Exceptions that the user might not be aware of, these should be
        mentioned here. This does not include standard exceptions like type errors from input sanitation or similar

        See Also
        -----------
        Referencencs to other functions can be given here (e.g for aliasing)

        Examples
        ----------
        >>> import heat as ht
        >>> T = ht.array([[1,2],[3,4]], dtype=ht.float)
        >>> ht.add(T, 2)
        tensor([[3., 4.],
                [5., 6.]])
        >>> T + 2
        tensor([[3., 4.],
                [5., 6.]])
        """

For classes, the docstring goes right under the class definition (as opposed to in the __init__ function). This
way, all attributes that are passed for class initialization are documented properly, with type and default
value annotation

Parameter Definitions
    * Defaults are defined in the function Parameters
    * Shape definitions go at the very end of the Parameter description in the following format: `Shape = (x, y, ...)`
    * For classes, the initialization parameters are defined as section ``Attributes``
    * Different Parameter types are separated by `or`, not commas
    * For detailed instructions on type hints for parameter and return type annotation (such as ``Union``, ``List``,
      ``Tuple``, etc.)
      See `typing <https://docs.python.org/3/library/typing.html>`_ (PEP 484)

Examples
    * Examples should only be separated by empty lines, if there is a clear distinction between the two example types.
      Note that every empty line in the examples will create a new example code block. This is fine for 2-3 separated
      blocks, but do not separate 15 different examples into individual blocks.
    * There must not be a colon after Examples
    * No comments in the examples (on number of processes or what the example shows). Put these in coding examples
      under ``Notes``
