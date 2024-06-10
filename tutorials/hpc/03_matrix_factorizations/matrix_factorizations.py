# # Matrix factorizations
#
# ### Refresher
#
# Using PyTorch as compute engine and mpi4py for communication, Heat implements a number of array operations and algorithms that are optimized for memory-distributed data volumes. This allows you to tackle datasets that are too large for single-node (or worse, single-GPU) processing.
#
# As opposed to task-parallel frameworks, Heat takes a data-parallel approach, meaning that each "worker" or MPI process performs the same tasks on different slices of the data. Many operations and algorithms are not embarassingly parallel, and involve data exchange between processes. Heat operations and algorithms are designed to minimize this communication overhead, and to make it transparent to the user.
#
# In other words:
# - you don't have to worry about optimizing data chunk sizes;
# - you don't have to make sure your research problem is embarassingly parallel, or artificially make your dataset smaller so your RAM is sufficient;
# - you do have to make sure that you have sufficient **overall** RAM to run your global task (e.g. number of nodes / GPUs).

# In the following, we will demonstrate the usage of Heat's truncated SVD algorithm.

# ### SVD and its truncated counterparts in a nutshell
#
# Let $X \in \mathbb{R}^{m \times n}$ be a matrix, e.g., given by a data set consisting of $m$ data points $\in \mathbb{R}^n$ stacked together. The so-called **singular value decomposition (SVD)** of $X$ is given by
#
# $$
# 	X = U \Sigma V^T
# $$
#
# where $U \in \mathbb{R}^{m \times r_X}$ and $V \in \mathbb{R}^{n \times r_X}$ have orthonormal columns, $\Sigma = \text{diag}(\sigma_1,...,\sigma_{r_X}) \in \mathbb{R}^{r_X \times r_X}$ is a diagonal matrix containing the so-called singular values $\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_{r_X} > 0$, and $r_X \leq \min(m,n)$ denotes the rank of $X$ (i.e. the dimension of the subspace of $\mathbb{R}^m$ spanned by the columns of $X$). Since $\Sigma = U^T X V$ is diagonal, one can imagine this decomposition as finding orthogonal coordinate transformations under which $X$ looks "linear".

# ### SVD in data science
#
# In data science, SVD is more often known as **principle component analysis (PCA)**, the columns of $U$ being called the principle components of $X$. In fact, in many applications **truncated SVD/PCA** suffices: to reduce $X$ to the "essential" information, one chooses a truncation rank $0 < r \leq r_X$ and considers the truncated SVD/PCA given by
#
# $$
# X \approx X_r := U_{[:,:r]} \Sigma_{[:r,:r]} V_{[:,:r]}^T
# $$
#
# where we have used `numpy`-like notation for selecting only the first $r$ columns of $U$ and $V$, respectively. The rationale behind this is that if the first $r$ singular values of $X$ are much larger than the remaining ones, $X_r$ will still contain all "essential" information contained in $X$; in mathematical terms:
#
# $$
# \lVert X_r - X \rVert_{F}^2 = \sum_{i=r+1}^{r_X} \sigma_i^2,
# $$
#
# where $\lVert \cdot \rVert_F$ denotes the Frobenius norm. Thus, truncated SVD/PCA may be used for, e.g.,
# * filtering away non-essential information in order to get a "feeling" for the main characteristics of your data set,
# * to detect linear (or "almost" linear) dependencies in your data,
# * to generate features for further processing of your data.
#
# Moreover, there is a plenty of more advanced data analytics and data-based simulation techniques, such as, e.g., Proper Orthogonal Decomposition (POD) or Dynamic Mode Decomposition (DMD), that are based on SVD/PCA.

# ### Truncated SVD in Heat
#
# In Heat we have currently implemented an algorithm for computing an approximate truncated SVD, where truncation takes place either w.r.t. a fixed truncation-rank (`heat.linalg.hsvd_rank`) or w.r.t. a desired accuracy (`heat.linalg.hsvd_rtol`). In the latter case it can be ensured that it holds for the "reconstruction error":
#
# $$
# \frac{\lVert X - U U^T X \rVert_F}{\lVert X \rVert_F} \overset{!}{\leq} \text{rtol},
# $$
#
# where $U$ denotes the approximate left-singular vectors of $X$ computed by `heat.linalg.hsvd_rtol`.
#

# To demonstrate the usage of Heat's truncated SVD algorithm, we will load the data set from the last example and then compute its truncated SVD. As usual, first we need to gain access to the MPI environment.


import heat as ht

X = ht.load_hdf5("~/mydata.h5", dataset="mydata", split=0).T


# Note that due to the transpose, `X` is distributed along the columns now; this is required by the hSVD-algorithm.

# Let's first compute the truncated SVD by setting the relative tolerance.


# compute truncated SVD w.r.t. relative tolerance
svd_with_reltol = ht.linalg.hsvd_rtol(X, rtol=1.0e-2, compute_sv=True, silent=False)
print("relative residual:", svd_with_reltol[3], "rank: ", svd_with_reltol[0].shape[1])


# Alternatively, you can compute a truncated SVD with a fixed truncation rank:

# compute truncated SVD w.r.t. a fixed truncation rank
svd_with_rank = ht.linalg.hsvd_rank(X, maxrank=3, compute_sv=True, silent=False)
print("relative residual:", svd_with_rank[3], "rank: ", svd_with_rank[0].shape[1])

# Once we have computed the truncated SVD, we can use it to approximate the original data matrix `X` by the truncated matrix `X_r`.
#
# Check out https://helmholtz-analytics.github.io/heat/2023/06/16/new-feature-hsvd.html to see how Heat's truncated SVD algorithm scales with the number of MPI processes and size of the dataset.

# ### Other factorizations
#
# Other common factorization algorithms are supported in Heat, such as:
# - QR decomposition (`heat.linalg.qr`)
# - Lanczos algorithm for computing the largest eigenvalues and corresponding eigenvectors (`heat.linalg.lanczos`)
#
# Check out our [`linalg` PRs](https://github.com/helmholtz-analytics/heat/pulls?q=is%3Aopen+is%3Apr+label%3Alinalg) to see what's in progress.
#

# **References for hierarchical SVD**
#
# 1. Iwen, Ong. *A distributed and incremental SVD algorithm for agglomerative data analysis on large networks.* SIAM J. Matrix Anal. Appl., **37** (4), 2016.
# 2. Himpe, Leibner, Rave. *Hierarchical approximate proper orthogonal decomposition.* SIAM J. Sci. Comput., **4** (5), 2018.
# 3. Halko, Martinsson, Tropp. *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions.* SIAM Rev. 53, **2** (2011)
