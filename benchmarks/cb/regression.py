import heat as ht
from perun import monitor

@monitor()
def lasso_fit(estimator, X, y):
    """
    Benchmark the fitting process of LASSO.
    """
    estimator.fit(X, y)

@monitor()
def lasso_predict(estimator, X):
    """
    Benchmark the prediction process of LASSO.
    """
    estimator.predict(X)

def run_regression_benchmarks():
    # Benchmark parameters
    n_data_points = 2000
    n_features = 200
    max_iter = 50

    # 1. Distributed Samples (split=0)
    # Common in large-scale data analytics where observations are partitioned
    X_s0 = ht.random.randn(n_data_points, n_features, split=0)
    true_w = ht.random.randn(n_features, 1)
    y_s0 = (X_s0 @ true_w) + 0.1 * ht.random.randn(n_data_points, 1, split=0)

    lasso_s0 = ht.regression.Lasso(max_iter=max_iter, tol=1e-6)

    print("Benchmarking LASSO Fit (split=0)...")
    lasso_fit(lasso_s0, X_s0, y_s0)

    print("Benchmarking LASSO Predict (split=0)...")
    lasso_predict(lasso_s0, X_s0)

    del X_s0, y_s0, lasso_s0

    # 2. Distributed Features (split=1)
    # Relevant for the coordinate descent approach used in your Lasso implementation
    X_s1 = ht.random.randn(n_data_points, n_features, split=1)
    y_s1 = (X_s1 @ true_w) + 0.1 * ht.random.randn(n_data_points, 1, split=None)

    lasso_s1 = ht.regression.Lasso(max_iter=max_iter, tol=1e-6)

    print("Benchmarking LASSO Fit (split=1)...")
    lasso_fit(lasso_s1, X_s1, y_s1)

    print("Benchmarking LASSO Predict (split=1)...")
    lasso_predict(lasso_s1, X_s1)

    del X_s1, y_s1, lasso_s1
