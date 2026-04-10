import numpy as np
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt

from hermite import compute_hermites
from sample_gaussian import sample_gaussian_measure

# Spatial domain discretization.
m_points = 256
grid = np.linspace(0.0, 1.0, m_points)
h = grid[1] - grid[0]
weights = np.full(m_points, h)
weights[0] *= 0.5
weights[-1] *= 0.5

# Gaussian field and Monte Carlo settings.
se_kernel = lambda x, y: np.exp(-0.5 * (x - y) ** 2 / 0.05 ** 2)
mean = lambda x: np.zeros_like(x)
kl_modes = 10
n_samples = 1000
total_degree = 1

rng = np.random.default_rng(42)
samples, gammas = sample_gaussian_measure(
    kernel=se_kernel,
    mean=mean,
    N=kl_modes,
    grid=grid,
    weights=weights,
    n_samples=n_samples,
    rng=rng,
)

if samples.ndim != 2:
    raise ValueError(f"samples must have shape (S, M), got {samples.shape}.")

sample_count, point_count = samples.shape
if point_count != m_points:
    raise ValueError(
        f"Sample/grid size mismatch: samples has M={point_count}, expected {m_points}."
    )
if gammas.shape != (sample_count, kl_modes):
    raise ValueError(
        f"Gamma shape mismatch: got {gammas.shape}, expected ({sample_count}, {kl_modes})."
    )

# Multi-index set A = {alpha in N_0^N : |alpha| <= total_degree}.
multi_indices = []
for d in range(total_degree + 1):
    for combo in combinations_with_replacement(range(kl_modes), d):
        alpha = np.zeros(kl_modes, dtype=int)
        for idx in combo:
            alpha[idx] += 1
        multi_indices.append(tuple(alpha.tolist()))
n_basis = len(multi_indices)
print(n_basis)

# 1D Hermites h_0,...,h_total_degree for each gamma_k sample list.
herm_1d = np.empty((kl_modes, total_degree + 1, sample_count), dtype=float)
for k in range(kl_modes):
    herm_1d[k] = compute_hermites(gammas[:, k], total_degree)

# Tensor-product Hermite chaos basis Psi_alpha(gamma).
psi = np.ones((sample_count, n_basis), dtype=float)
for a_idx, alpha in enumerate(multi_indices):
    for k, degree in ((k, degree) for k, degree in enumerate(alpha) if degree > 0):
        psi[:, a_idx] *= herm_1d[k, degree, :]

# Chaos Gram matrix in probability space: <Psi_a, Psi_b>_Omega.
chaos_gram = (psi.T @ psi) / sample_count

identity = np.eye(n_basis)
abs_error = np.abs(chaos_gram - identity)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

im0 = axes[0].imshow(chaos_gram, origin="lower", cmap="plasma")
axes[0].set_title("Gram Matrix G")
axes[0].set_xlabel("j")
axes[0].set_ylabel("i")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(abs_error, origin="lower", cmap="plasma")
axes[1].set_title("|G - I|")
axes[1].set_xlabel("j")
axes[1].set_ylabel("i")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.show()


