import numpy as np
from numpy.typing import NDArray


def compute_hermites(grid: NDArray, N: int) -> NDArray:
	"""
	Compute the first N+1 orthonormal Hermite polynomials on a 1D grid.

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

	for k in range(1, N):
		hermites[k + 1] = (grid * hermites[k] - np.sqrt(k) * hermites[k - 1]) / np.sqrt(k + 1)

	return hermites


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	grid = np.linspace(-4.0, 4.0, 400)
	N = 10
	hermite_values = compute_hermites(grid, N)

	for k, values in enumerate(hermite_values):
		plt.plot(grid, values, lw=1.0, label=fr"$H_{k}$")

	plt.title(r"Orthonormal Hermite polynomials")
	plt.xlabel(r"$x$")
	plt.ylabel(r"$H_k(x)$")
	plt.legend(ncol=2, fontsize=9)
	plt.tight_layout()
	plt.show()
