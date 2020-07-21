Feel the HeAT - A 30 Minutes Welcome
====================================

HeAT is a flexible and seamless open-source software for high performance data analytics and machine learning in Python.

Our main audience are




Getting Started
---------------

Make sure that you have HeAT installed on your system. For detailed description of your options, see our full :ref:`Getting Started <Installation>` section. For now, this should suffice:

.. code:: bash

    pip install heat

DNDarrays
---------

DNDarrays mimick NumPy’s ndarrays interface as close as possible. On top of that they can also be used to accelerate computations with GPUs or MPI in distributed cluster systems. Let's try it out:

.. code:: python

    import heat as ht

The following creates a :math:`3\times 4` matrix with zeros.

.. code:: python

    ht.zeros((3,4,))

Output:

.. code:: output

    DNDarray([[0., 0., 0., 0.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=None)


Or a vector with the numbers from :math:`0-9` ascending.

.. code:: python

    ht.arange(10)

Output:

.. code:: output

    DNDarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=ht.int32, device=cpu:0, split=None)


The following snippet creates a column vector with :math:`5` element, each position filled with the value :math:`9` and the ``ht.int64`` data type.

.. code:: python

    ht.full((1, 5,), fill_value=9, dtype=ht.int64)

Output:

.. code:: output

    DNDarray([[9, 9, 9, 9, 9]], dtype=ht.int64, device=cpu:0, split=None)


Finally, let's load some user defined data.

.. note::

    HeAT takes care of automatically inferring the shape, i.e. the tensor dimensions, and data types from the user provided input.

.. code:: python

    ht.array([[0, 1, 2], [0.1, 0.2, 3]])

Output:

.. code:: output

    DNDarray([[0.0000, 1.0000, 2.0000],
              [0.1000, 0.2000, 3.0000]], dtype=ht.float32, device=cpu:0, split=None)

Operations
----------

HeAT supports several mathematical operations, ranging from simple element-wise functions, binary arithmetic operations, and linear algebra, to more powerful reductions. In the following example we add a two matrices of same size.

.. code:: python

    ht.full((3, 4,), fill_value=9) + ht.ones((3, 4,))

Output:

.. code:: output

    DNDarray([[10., 10., 10., 10.],
              [10., 10., 10., 10.],
              [10., 10., 10., 10.]], dtype=ht.float32, device=cpu:0, split=None)

Instead of operators, we can also use a functional approach.

.. code:: python

    ht.add(ht.full((3, 4,), fill_value=9), ht.ones((3, 4,)))

Output:

.. code:: output

    DNDarray([[10., 10., 10., 10.],
              [10., 10., 10., 10.],
              [10., 10., 10., 10.]], dtype=ht.float32, device=cpu:0, split=None)


If there is no obvious operator for a function, you can also call a method on the ``DNDarray``.

.. code:: python

    ht.arange(5).sin()

Output:

.. code:: output

    DNDarray([ 0.0000,  0.8415,  0.9093,  0.1411, -0.7568], dtype=ht.float32, device=cpu:0, split=None)

Just like other numerical computation libraries, HeAT supports broadcasting. It describes how two ``DNDarrays`` with different dimensions (also called shape) can still be combined in arithmetic operations given certain constraints. For example, we can add a scalar to a matrix.

.. code:: python

    ht.zeros((3, 4,)) + 5.0

Output:

.. code:: output

    DNDarray([[5., 5., 5., 5.],
              [5., 5., 5., 5.],
              [5., 5., 5., 5.]], dtype=ht.float32, device=cpu:0, split=None)

The scalar has been element-wise repeated for every entry within the matrix. We can do the same with matrices and vectors as well


.. code:: python

    ht.zeros((3, 4,)) + ht.arange(4)

Output:

.. code:: output

    DNDarray([[0., 1., 2., 3.],
              [0., 1., 2., 3.],
              [0., 1., 2., 3.]], dtype=ht.float32, device=cpu:0, split=None)

The vector has been repeated for every row of the left-hand side matrix. A full description of broadcasting rules can be found in `NumPy's manual <https://numpy.org/devdocs/user/theory.broadcasting.html>`_. While talking about it, HeAT is designed as seamless drop-in replacement for NumPy. There still might be cases, e.g. working with native Python code, when you want to convert a ``DNDarray`` to an ``ndarray`` instead.


.. code:: python

    ht.arange(5).numpy()

Output:

.. code:: output

    array([0, 1, 2, 3, 4], dtype=int32)

And vice versa:

.. code:: python

    import numpy as np
    ht.array(np.arange(5))

Output:

.. code:: output

    DNDarray([0, 1, 2, 3, 4], dtype=ht.int64, device=cpu:0, split=None)

.. seealso::
    Read up more later on hundreds of other functions in our `API reference <autoapi/index.html>`_. Or find out about them interactively by using the ``help()`` function in your Python interpreter.
