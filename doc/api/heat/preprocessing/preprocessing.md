Module heat.preprocessing.preprocessing
=======================================
Module implementing basic data preprocessing techniques

Classes
-------

`MaxAbsScaler(*, copy: bool = True)`
:   MaxAbsScaler: scale each feature of a given data set linearly by its maximum absolute value. The underyling data set to be scaled is
    assumed to be stored as a 2D-`DNDarray` of shape (n_datapoints, n_features); this routine is similar to
    `sklearn.preprocessing.MaxAbsScaler`.

    Each feature is scaled individually such that the maximal absolute value of each feature after transformation will be 1.0.
    No shifting/centering is applied.

    Parameters
    ----------
    copy : bool, default=True
        ``copy=False`` enables in-place transformation.

    Attributes
    ----------
    scale_ : DNDarray of shape (n_features,)
        Per feature relative scaling of the data.

    max_abs_ : DNDarray of shape (n_features,)
        Per feature maximum absolute value of the input data.

    ### Ancestors (in MRO)

    * heat.core.base.TransformMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, X: heat.core.dndarray.DNDarray) ‑> Self`
    :   Fit MaxAbsScaler to input data ``X``: compute the parameters to be used for later scaling.

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            The data set to which the scaler shall be fitted.

    `inverse_transform(self, Y: heat.core.dndarray.DNDarray) ‑> Self | heat.core.dndarray.DNDarray`
    :   Apply the inverse of :meth:``transform``, i.e. scale the input data ``Y`` back to the original representation.

        Parameters
        ----------
        Y : DNDarray of shape (n_datapoints, n_features)
            The data set to be transformed back.

    `transform(self, X: heat.core.dndarray.DNDarray) ‑> Self | heat.core.dndarray.DNDarray`
    :   Scale the data with the MaxAbsScaler.

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            The data set to be scaled.

`MinMaxScaler(feature_range: Tuple[float, float] = (0.0, 1.0), *, copy: bool = True, clip: bool = False)`
:   Min-Max-Scaler: transforms the features by scaling each feature (affine) linearly to the prescribed range;
    similar to `sklearn.preprocessing.MinMaxScaler`.
    The data set to be scaled must be stored as 2D-`DNDarray` of shape (n_datapoints, n_features).

    Each feature is scaled and translated individually such that it is in the given range on the input data set,
    e.g. between zero and one (default).

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed features.

    copy : bool, default=True
        ``copy = False`` means in-place transformations whenever possible.

    clip : Not yet supported.
        raises ``NotImplementedError``.

    Attributes
    ----------
    min_ : DNDarray of shape (n_features,)
        translation required per feature

    scale_ : DNDarray of shape (n_features,)
        scaling required per feature

    data_min_ : DNDarray of shape (n_features,)
        minimum per feature in the input data set

    data_max_ : DNDarray of shape (n_features,)
        maximum per feature in the input data set

    data_range_ : DNDarray of shape (n_features,)
        range per feature in the input data set

    ### Ancestors (in MRO)

    * heat.core.base.TransformMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, X: heat.core.dndarray.DNDarray) ‑> Self`
    :   Fit the MinMaxScaler: i.e. compute the parameters required for later scaling.

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            data set to which scaler shall be fitted.

    `inverse_transform(self, Y: heat.core.dndarray.DNDarray) ‑> Self | heat.core.dndarray.DNDarray`
    :   Apply the inverse of :meth:``fit``.

        Parameters
        ----------
        Y : DNDarray of shape (n_datapoints, n_features)
            Data set to be transformed back.

    `transform(self, X: heat.core.dndarray.DNDarray) ‑> Self | heat.core.dndarray.DNDarray`
    :   Transform input data with MinMaxScaler: i.e. scale features of ``X`` according to feature_range.

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            Data set to be transformed.

`Normalizer(norm: str = 'l2', *, copy: bool = True)`
:   Normalizer: each data point of a data set is scaled to unit norm independently.
    The data set to be scaled must be stored as 2D-`DNDarray` of shape (n_datapoints, n_features); therefore
    the Normalizer scales each row to unit norm. This object is similar to `sklearn.preprocessing.Normalizer`.

    Parameters
    ----------
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize the data points. ``norm='max'`` refers to the :math:`\ell^\infty`-norm.

    copy : bool, default=True
        ``copy=False`` enables in-place normalization.

    Attributes
    ----------
    None


    Notes
    -----
    Normalizer is :term:`stateless` and, consequently, :meth:``fit`` is only a dummy that does not need to be called before :meth:``transform``.
    Since :meth:``transform`` is not bijective, there is no back-transformation :meth:``inverse_transform``.

    ### Ancestors (in MRO)

    * heat.core.base.TransformMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, X: heat.core.dndarray.DNDarray) ‑> Self`
    :   Since :object:``Normalizer`` is stateless, this function is only a dummy.

    `transform(self, X: heat.core.dndarray.DNDarray) ‑> Self | heat.core.dndarray.DNDarray`
    :   Apply Normalizer trasformation: scales each data point of the input data set ``X`` to unit norm (w.r.t. to ``norm``).

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            The data set to be normalized.

        copy : bool, default=None
            ``copy=False`` enables in-place transformation.

`RobustScaler(*, with_centering: bool = True, with_scaling: bool = True, quantile_range: Tuple[float, float] = (25.0, 75.0), copy: bool = True, unit_variance: bool = False, sketched: bool = False, sketch_size: float | None = 1.0)`
:   Scales the features of a given data set making use of statistics
    that are robust to outliers: it removes the median and scales the data according to
    the quantile range (defaults to IQR: Interquartile Range); this routine is similar
    to ``sklearn.preprocessing.RobustScaler``.

    Per default, the "true" median and IQR of the entire data set is computed; however, the argument
    `sketched` allows to switch to a faster but inaccurate version that computes
    median and IQR only on behalf of a random subset of the data set ("sketch") of size `sketch_size`.

    The underyling data set to be scaled must be stored as a 2D-`DNDarray` of shape (n_datapoints, n_features).
    Each feature is centered and scaled independently.

    Parameters
    ----------
    with_centering : bool, default=True
        If `True`, data are centered before scaling.

    with_scaling : bool, default=True
        If `True`, scale the data to prescribed interquantile range.

    quantile_range : tuple (q_min, q_max), 0.0 <= q_min < q_max <= 100.0,         default=(25.0, 75.0)
        Quantile range used to calculate `scale_`; default is the so-called
        the IQR given by ``q_min=25`` and ``q_max=75``.

    copy : bool, default=True
        ``copy=False`` enable in-place transformations.

    unit_variance : not yet supported.
        raises ``NotImplementedError``

    sketched : bool, default=False
        If `True`, use a sketch of the data set to compute the median and IQR.
        This is faster but less accurate. The size of the sketch is determined by the argument `sketch_size`.

    sketch_size : float, default=1./ht.MPI_WORLD.size
        Fraction of the data set to be used for the sketch if `sketched=True`. The default value is 1/N, where N is the number of MPI processes.
        Ignored if `sketched=False`.

    Attributes
    ----------
    center_ : DNDarray of shape (n_features,)
        Feature-wise median value of the given data set.

    iqr_ : DNDarray of shape (n_features,)
        length of the interquantile range for each feature.

    scale_ : array of floats
        feature-wise inverse of ``iqr_``.

    ### Ancestors (in MRO)

    * heat.core.base.TransformMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, X: heat.core.dndarray.DNDarray) ‑> Self`
    :   Fit RobustScaler to given data set, i.e. compute the parameters required for transformation.

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            Data to which the Scaler should be fitted.

    `inverse_transform(self, Y: heat.core.dndarray.DNDarray) ‑> Self | heat.core.dndarray.DNDarray`
    :   Apply inverse of :meth:``transform``.

        Parameters
        ----------
        Y : DNDarray of shape (n_datapoints, n_features)
            Data to be back-transformed

    `transform(self, X: heat.core.dndarray.DNDarray) ‑> Self | heat.core.dndarray.DNDarray`
    :   Transform given data with RobustScaler

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            Data set to be transformed.

`StandardScaler(*, copy: bool = True, with_mean: bool = True, with_std: bool = True)`
:   Standardization of features to mean 0 and variance 1 by affine linear transformation; similar to `sklearn.preprocessing.StandardScaler`.
    The data set to be scaled must be stored as 2D-`DNDarray` of shape (n_datapoints, n_features).
    Shifting to mean 0 and scaling to variance 1 is applied to each feature independently.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.

    with_mean : bool, default=True
        If True, center the data (i.e. mean = 0) before scaling.

    with_std : bool, default=True
        If True, scale the data to variance = 1.

    Attributes
    ----------
    scale_ : DNDarray of shape (n_features,) or None
        Per feature relative scaling of the data to achieve unit
        variance. Set to ``None`` (no variance scaling applied) if ``var = None`` or ``var`` below machine precision.

    mean_ : DNDarray of shape (n_features,) or None
        The mean value for each feature. Equal to ``None`` when ``with_mean=False``.

    var_ : DNDarray of shape (n_features,) or None
        Featurewise variance of the given data. Equal to ``None`` when ``with_std=False``.

    ### Ancestors (in MRO)

    * heat.core.base.TransformMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, X: heat.core.dndarray.DNDarray, sample_weight: heat.core.dndarray.DNDarray | None = None) ‑> Self`
    :   Fit ``StandardScaler`` to the given data ``X``, i.e. compute mean and standard deviation of ``X`` to be used for later scaling.

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features).
            Data used to compute the mean and standard deviation used for later featurewise scaling.

        sample_weight : Not yet supported.
            Raises ``NotImplementedError``.

    `inverse_transform(self, Y: heat.core.dndarray.DNDarray) ‑> Self | heat.core.dndarray.DNDarray`
    :   Scale back the data to the original representation, i.e. apply the inverse of :meth:``transform`` to the input ``Y``.

        Parameters
        ----------
        Y : DNDarray of shape (n_datapoints, n_features)
            Data to be scaled back.
        copy : bool, default=None
            Copy the input ``Y`` or not.

    `transform(self, X: heat.core.dndarray.DNDarray) ‑> Self | heat.core.dndarray.DNDarray`
    :   Applies standardization to input data ``X`` by centering and scaling w.r.t. mean and std previously computed and saved in ``StandardScaler`` with :meth:``fit``.

        Parameters
        ----------
        X : DNDarray (n_datapoints, n_features)
            The data set to be standardized.
        copy : bool, default=None
            Copy the input ``X`` or not.
