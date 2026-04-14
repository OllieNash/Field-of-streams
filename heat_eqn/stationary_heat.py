"""Stationary distribution demo for the 1D stochastic heat equation.

## Goal
Approximate the stationary distribution of an observable phi(u) by:
1. Sampling an initial field from the Gaussian field model.
2. Running many independent stochastic heat trajectories.
3. Discarding burn-in steps.
4. Evaluating phi(u) on each post-burn-in sample.
5. Plotting a histogram of phi(u).

## Usage
Run directly from the repository root:

    python heat_eqn/stationary_heat.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample_gaussian import sample_gaussian_measure
from stochastic_heat import simulate_spde


def sample_stationary_observable(
    n_trajectories: int = 400,
    burn_in_steps: int = 250,
    dt: float = 1e-3,
    beta: float = 0.1,
    nu: float = 0.5,
    grid_size: int = 256,
    observable: Callable[[np.ndarray, float], float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample an empirical stationary distribution of an observable phi(u).

    Args:
        n_trajectories: Number of independent trajectories.
        burn_in_steps: Number of stochastic heat steps to discard.
        dt: Time step size.
        beta: Noise-strength parameter.
        nu: Diffusion coefficient.
        grid_size: Number of spatial grid points.
        observable: Scalar observable phi(u, dx). If None, uses the first marginal u[0].
        rng: Random number generator for reproducibility.

    Returns:
        grid: Spatial grid.
        initial_state: Sampled initial condition on the grid.
        observable_samples: Array of shape (n_trajectories,) with phi(u) samples.
    """
    rng = rng or np.random.default_rng()

    # Spatial grid and quadrature weights for the Gaussian sampler.
    grid = np.linspace(0.0, 1.0, grid_size, endpoint=False)
    dx = grid[1] - grid[0]
    weights = np.full(grid_size, dx)

    if observable is None:
        # Default observable: first spatial marginal.
        observable = lambda u, dx_val: float(u[0])

    # Sample one initial condition from the Gaussian field model.
    se_kernel = lambda x, y: np.exp(-0.5 * (x - y) ** 2 / 0.05 ** 2)
    mean = lambda x: np.zeros_like(x)
    initial_samples, _ = sample_gaussian_measure(
        kernel=se_kernel,
        mean=mean,
        N=50,
        grid=grid,
        weights=weights,
        n_samples=1,
        rng=rng,
    )
    initial_state = initial_samples[0]

    # Run many independent trajectories and evaluate phi(u) after burn-in.
    observable_samples = np.empty(n_trajectories, dtype=float)
    for j in range(n_trajectories):
        trajectory_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        state = initial_state.copy()
        for _ in range(burn_in_steps):
            state = simulate_spde(
                state,
                dt=dt,
                dx=dx,
                beta=beta,
                nu=nu,
                rng=trajectory_rng,
            )
        observable_samples[j] = observable(state, dx)

    return grid, initial_state, observable_samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Experiment settings.
    dt = 1e-3
    burn_in_steps = 250
    n_trajectories = 1000
    beta = 0.1
    nu = 0.5

    # Example observable: first spatial marginal.
    phi_first_marginal = lambda u, dx_val: float(u[5])

    grid, u0, phi_samples = sample_stationary_observable(
        n_trajectories=n_trajectories,
        burn_in_steps=burn_in_steps,
        dt=dt,
        beta=beta,
        nu=nu,
        grid_size=256,
        observable=phi_first_marginal,
        rng=np.random.default_rng(42),
    )

    # Visualize initial condition and empirical law of phi(u).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axes[0].plot(grid, u0, lw=1.5, color="tab:blue")
    axes[0].set_title("Sampled initial condition")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(phi_samples, bins=35, density=True, alpha=0.8, color="tab:orange", edgecolor="white")
    axes[1].set_title("Empirical stationary distribution of phi(u)")
    axes[1].set_xlabel(r"$u_0$")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.3)

    plt.show()
