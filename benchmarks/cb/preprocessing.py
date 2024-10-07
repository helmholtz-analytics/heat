# flake8: noqa
import heat as ht
from mpi4py import MPI
from perun import monitor
from sizes import GSIZE_TS_L, GSIZE_TS_S

"""
Benchmarks in this file:
- StandardScaler and inverse_transform
- MinMaxScaler and inverse_transform
- MaxAbsScaler and inverse_transform
- RobustScaler and inverse_transform
- Normalizer (without inverse, of course)
All of them are both fit_transform and inverse_transform (together); data is split along the data axis.
"""


@monitor()
def apply_inplace_standard_scaler_and_inverse(X):
    scaler = ht.preprocessing.StandardScaler(copy=False)
    scaler.fit_transform(X)
    scaler.inverse_transform(X)


@monitor()
def apply_inplace_min_max_scaler_and_inverse(X):
    scaler = ht.preprocessing.MinMaxScaler(copy=False)
    scaler.fit_transform(X)
    scaler.inverse_transform(X)


@monitor()
def apply_inplace_max_abs_scaler_and_inverse(X):
    scaler = ht.preprocessing.MaxAbsScaler(copy=False)
    scaler.fit_transform(X)
    scaler.inverse_transform(X)


@monitor()
def apply_inplace_robust_scaler_and_inverse(X):
    scaler = ht.preprocessing.RobustScaler(copy=False)
    X = scaler.fit_transform(X)
    X = scaler.inverse_transform(X)


@monitor()
def apply_inplace_normalizer(X):
    scaler = ht.preprocessing.Normalizer(copy=False)
    scaler.fit_transform(X)


def run_preprocessing_benchmarks():
    n_data_points = GSIZE_TS_L
    n_features = GSIZE_TS_S
    X = ht.random.randn(n_data_points, n_features, split=0)

    apply_inplace_standard_scaler_and_inverse(X)
    apply_inplace_min_max_scaler_and_inverse(X)
    apply_inplace_max_abs_scaler_and_inverse(X)
    apply_inplace_robust_scaler_and_inverse(X)
    apply_inplace_normalizer(X)

    del X
