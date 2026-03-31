import numpy as np
from numpy.typing import NDArray
from typing import Callable

def sample_gaussian_measure(
    kernel: Callable[[NDArray, NDArray], NDArray],
    mean: Callable[[NDArray], NDArray] | NDArray,
    N: int,
    grid: NDArray,
    weights: NDArray,
    n_samples: int = 1,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """
    Sample xi ~ N(m, K) on L^2(Omega) via the truncated Karhunen-Loeve expansion:

        xi = m + sum_{k=1}^{N} sqrt(lambda_k) * gamma_k * e_k,  gamma_k ~ N(0,1)

    Args:
        kernel:    Symmetric PD kernel k(x, y), broadcastable over NDArrays.
        mean:      Mean function m(x) (callable) or pre-evaluated array of shape (M,).
        N:         Number of KL modes to retain (spectral truncation).
        grid:      Spatial quadrature points, shape (M,).
        weights:   Quadrature weights, shape (M,). E.g. trapezoidal.
        n_samples: Number of independent samples. Default 1.
        rng:       Random generator for reproducibility.

    Returns:
        samples:   NDArray of shape (n_samples, M).
    """
    M = len(grid)
    if not 1 <= N <= M:
        raise ValueError(f"Require 1 <= N <= M, got N={N}, M={M}.")

    rng = rng or np.random.default_rng()

    # Kernel matrix
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    K_mat = kernel(X, Y)

    # Similarity transform: B = W^{1/2} K W^{1/2}, elementwise since W is diagonal
    sqrt_w = np.sqrt(weights)
    B = sqrt_w[:, None] * K_mat * sqrt_w[None, :]

    # Top-N eigenpairs of B
    eigenvalues, evecs_B = np.linalg.eigh(B)
    eigenvalues = np.maximum(eigenvalues[::-1][:N], 0.0)
    evecs_B = evecs_B[:, ::-1][:, :N]

    # Recover W-orthonormal eigenvectors: e_k = W^{-1/2} v_k
    evecs = evecs_B / sqrt_w[:, None]

    # Mean vector
    m_vec = mean(grid) if callable(mean) else np.asarray(mean, dtype=float)
    if m_vec.shape != (M,):
        raise ValueError(f"mean must have shape ({M},), got {m_vec.shape}.")

    # KL samples: xi = m + (gamma * sqrt(lambda)) @ E^T
    gamma = rng.standard_normal(size=(n_samples, N))
    return m_vec + (gamma * np.sqrt(eigenvalues)) @ evecs.T


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    M = 256
    grid = np.linspace(0.0, 1.0, M)
    h = grid[1] - grid[0]
    weights = np.full(M, h); weights[0] *= 0.5; weights[-1] *= 0.5 # Trapezoid rule
    
    se_kernel = lambda x, y: np.exp(-0.5 * (x - y)**2 / 0.05**2)
    mean = lambda x: np.zeros_like(x)
    samples = sample_gaussian_measure(
        kernel=se_kernel,
        mean=mean,
        N=50, grid=grid, weights=weights, n_samples=10,
        rng=np.random.default_rng(42),
    )

    for s in samples:
        plt.plot(grid, s, lw=0.8, alpha=0.7)
    plt.title(r"Samples from $\mathcal{N}(0, K)$")
    plt.xlabel("$x$"); plt.ylabel(r"$\xi(x)$")
    plt.tight_layout(); plt.show()