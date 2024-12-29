import numpy as np
from numpy import cov, trace, iscomplexobj
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn import metrics

def calculate_inception_score(p_yx, eps=1E-16):
    """
    Calculate the Inception Score (IS) for generated samples.

    Arguments:
    - p_yx: Probability distributions of classes for each image (n_samples x n_classes).
    - eps: A small value to avoid numerical instability in logarithms (default: 1E-16).

    Returns:
    - is_score: The calculated Inception Score.
    """
    # Calculate the marginal class probabilities, p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)

    # Compute KL divergence for each image between p(y|x) and p(y)
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))

    # Sum over the class probabilities
    sum_kl_d = kl_d.sum(axis=1)

    # Average the KL divergence over all images
    avg_kl_d = np.mean(sum_kl_d)

    # Exponentiate the result to obtain the Inception Score
    is_score = np.exp(avg_kl_d)
    return is_score

def calculate_fid(act1, act2):
    """
    Calculate the Frechet Inception Distance (FID) between two sets of activations.

    Arguments:
    - act1: Activations from the first dataset (n_samples x n_features).
    - act2: Activations from the second dataset (n_samples x n_features).

    Returns:
    - fid: The calculated FID score.
    """
    # Compute the mean and covariance of activations for both datasets
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

    # Compute the sum of squared differences between the means
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # Compute the square root of the product of covariance matrices
    covmean = sqrtm(sigma1.dot(sigma2))

    # Correct numerical instability by discarding imaginary components if present
    if iscomplexobj(covmean):
        covmean = covmean.real

    # Compute the FID score using the formula
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def mmd_linear(X, Y):
    """
    Compute the Maximum Mean Discrepancy (MMD) using a linear kernel.

    Arguments:
    - X: Samples from the first dataset (n_samples x n_features).
    - Y: Samples from the second dataset (n_samples x n_features).

    Returns:
    - mmd_value: The calculated MMD value.
    """
    # Convert inputs to numpy arrays
    X, Y = np.array(X), np.array(Y)

    # Compute pairwise dot products for X and Y
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)

    # Calculate the linear kernel MMD value
    mmd_value = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd_value

def mmd_rbf(X, Y, gamma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) using an RBF (Gaussian) kernel.

    Arguments:
    - X: Samples from the first dataset (n_samples x n_features).
    - Y: Samples from the second dataset (n_samples x n_features).

    Keyword Arguments:
    - gamma: Kernel parameter controlling the width of the Gaussian kernel (default: 1.0).

    Returns:
    - mmd_value: The calculated MMD value.
    """
    # Compute pairwise RBF kernels for X and Y
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)

    # Calculate the RBF kernel MMD value
    mmd_value = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd_value

def wass_distance(x, y):
    """
    Compute the average Wasserstein distance between corresponding rows of two datasets.

    Arguments:
    - x: Samples from the first dataset (n_samples x n_features).
    - y: Samples from the second dataset (n_samples x n_features).

    Returns:
    - avg_distance: The average Wasserstein distance.
    """
    # Initialize a variable to accumulate the distances
    total_distance = 0

    # Loop through each row in the datasets
    for i in range(x.shape[0]):
        total_distance += wasserstein_distance(x[i], y[i])

    # Calculate the average distance
    avg_distance = total_distance / x.shape[0]
    return avg_distance

