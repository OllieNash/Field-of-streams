import math
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.hermite import hermgauss

def compute_hermites(grid: NDArray, N: int) -> NDArray:
    """
    Compute the first N+1 orthonormal probabilists' Hermite polynomials on a 1D grid.

    Args:
        grid: Spatial points, shape (M,).
        N: Highest Hermite index to compute.

    Returns:
        values: NDArray of shape (N+1, M), where values[k, i] = H_k(grid[i]).
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError(f"grid must be one-dimensional, got shape {grid.shape}.")
    if N < 0:
        raise ValueError(f"N must be nonnegative, got N={N}.")

    M = grid.shape[0]
    hermites = np.zeros((N + 1, M), dtype=float)
    hermites[0] = 1.0

    if N >= 1:
        hermites[1] = grid

    # Hermite recurrence:
    # h_{k+1}(x) = x/sqrt(k+1) h_k(x) - sqrt(k/(k+1)) h_{k-1}(x).
    for k in range(1, N):
        kp1 = k + 1
        hermites[k + 1] = (grid / math.sqrt(kp1)) * hermites[k] - math.sqrt(k / kp1) * hermites[k - 1]
   
    return hermites


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    K = 100 # Spatial discreization 
    x, w = hermgauss(K) # Gaussian quadrature nodes plus weights
    
    # Convert from physicists' Gauss-Hermite rule (weight exp(-t^2))
    # to the standard normal rule (weight (2*pi)^(-1/2) exp(-x^2/2)).
    x = np.sqrt(2) * x
    w = w / np.sqrt(np.pi)


    N = 10  # Spectral truncation
    hermite_values = compute_hermites(x, N)

    # Gram matrix G_ij = <H_i, H_j> under the standard normal measure.
    gram = (hermite_values * w) @ hermite_values.T
    error = gram - np.eye(N + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(gram, origin="lower", cmap="plasma")
    axes[0].set_title("Gram Matrix G")
    axes[0].set_xlabel("j")
    axes[0].set_ylabel("i")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    vmax = np.max(np.abs(error))
    im1 = axes[1].imshow(error, origin="lower", cmap="plasma", vmin=-vmax, vmax=vmax)
    axes[1].set_title("G - I")
    axes[1].set_xlabel("j")
    axes[1].set_ylabel("i")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.show()

