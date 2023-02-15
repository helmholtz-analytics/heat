Distributed Functionalities as Numpy
====================================

Inference
---------
Heat is a Python package for accelerated and distributed tensor computations. Internally, it is based on PyTorch. Heat has various functionalities similar to that of numpy/scipy and scikit-learn. What more is offered is that it supports distributed tensor computations for accelerated computations. The implementations allows us to tackle use cases that would otherwise exceed memory limits of a single node. Below listed are the various operatiosn that are performed on heat, just like any other tensor library:

Features
--------

Importing Heat:

.. code:: python

    import heat as ht

Array Initialization:

.. code:: python

    ht.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]], split=1)

Basic Numpy operations performed in Heat:

.. code:: python

    #Various Array initilization menthods
    ht.zeros((3, 4))
    ht.ones((3, 4))
    ht.random.randn(3)
    ht.linspace(3, 8)
    ht.arange(10)
    ht.full(3, 9)
    ht.eye(4)


    #Basic Mathematical Functions
    ht.sin(ht.array([1, 2, 3]))
    ht.cos(ht.array([3, 4, 5]))
    ht.log(ht.array([5, 6, 7]))
    ht.exp(ht.array([7, 8, 9]))
    ht.sqrt(ht.array([9, 1, 2]))
    ht.min(ht.array([4, 8, 9]))
    ht.max(ht.array([1, 1, 7]))
    ht.unique(ht.array([3, 6, 6]))
    ht.mean(ht.array([9, 1, 2]))
    ht.median(ht.array([5, 1, 5]))



Matrix Operaterations:

.. code:: python

    a = ht.array([1, 2, 3])
    b = ht.array([4, 5, 6])

    #Matrix Multiplication
    ht.matmul(a, b, True)
    #boolean expression represents whether to distribute a in the case that both a.split is None and b.split is None

    #Matrix Norm
    ht.matrix_norm(ht.array([[1, 2], [3, 4]]))

    #Transpose of a matrix
    ht.transpose(a)

    #Dot product
    ht.dot(a, b)

    #Cross Product
    ht.cross(a, b)

    #Reshape a Matrix
    ht.reshape(a, (3, 1))


Data science and Machine Learning
---------------------------------

.. code:: python

    X = ht.random.randn(10, 4, split=0)
    Y = ht.random.randn(10, 1, split=0)

    #Linear Regression (a linear model with L1 regularization)
    e = ht.regression.lasso.Lasso(max_iter=10)
    e.fit(X, Y)

    #K nearest neighbour
    knn = ht.classification.kneighborsclassifier.KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, Y)

    #Naive Bayes
    #Gaussian Naive Bayes
    clf=ht.naive_bayes.GaussianNB()
    clf.fit(X, Y)
