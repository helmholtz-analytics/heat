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
    n_data_points = 20000
    n_features = 10
    max_iter = 100

    # Distributed Samples (split=0)
    X = ht.random.randn(n_data_points, n_features, split=0)
    ground_truth = ht.random.randn(n_features, 1)
    # define y = X @ ground_truth + noise
    y = (X @ ground_truth) + 0.1 * ht.random.randn(n_data_points, 1, split=0)

    lasso = ht.regression.Lasso(max_iter=max_iter, tol=1e-6)

    print("Benchmarking LASSO Fit (split=0)...")
    lasso_fit(lasso, X, y)

    print("Benchmarking LASSO Predict (split=0)...")
    lasso_predict(lasso, X)

    del X, y, lasso
