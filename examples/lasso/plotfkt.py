from matplotlib import pyplot as plt


# plotting function for lasso paths
def plot_lasso_path(lamda, theta_lasso, feature_names, title="Lasso Paths - Numpy implementation"):
    # Plot results
    n, _ = theta_lasso.shape

    for i in range(n):
        plt.plot(lamda, theta_lasso[i], label=feature_names[i])

    plt.xscale("log")
    plt.xlabel("Log($\\lambda$)")
    plt.ylabel("Coefficients")
    plt.title(title)
    plt.axis("tight")
