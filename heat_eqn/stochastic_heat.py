r"""One-step stochastic heat equation solver in 1D.

This module advances a spatial field by one timestep under the stochastic
heat equation

    du = nu * u_xx dt + sigma dW,

using a semi-implicit FFT update on a periodic grid.

The main entry point is `simulate_spde`, which takes an initial condition and
returns the field one step later. When run as a script, this file samples an
initial condition from the Gaussian field utilities and plots the state before
and after one stochastic heat step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heat import heat_step_implicit
from sample_gaussian import sample_gaussian_measure
from Plotter import plot_spde_path


def simulate_spde(
    initial_state: np.ndarray,
    dt: float = 1e-3,
    dx: float | None = None,
    beta: float = 1,
    nu: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""
    Advance one timestep of the 1D stochastic heat equation.

    The update is

        u^{n+1} = (I - dt * nu * \Delta)^{-1}(u^n + sigma * dW),

    where `dW` is spatially discretized Gaussian noise and

        sigma = sqrt(2 / beta).

    Args:
        initial_state: Initial field with shape (N,) or (M, N).
        dt: Time step size.
        dx: Spatial grid spacing. If omitted, uses 1 / N.
        beta: Noise-strength parameter.
        nu: Diffusion coefficient.
        rng: Random number generator for reproducibility.

    Returns:
        The field one stochastic heat step later, with the same shape as
        `initial_state`.
    """
    state = np.asarray(initial_state, dtype=float)
    if state.ndim not in (1, 2):
        raise ValueError(f"initial_state must be 1D or 2D, got shape {state.shape}.")
    if dt < 0 or beta <= 0 or nu < 0:
        raise ValueError(f"Require dt>=0, beta>0, nu>=0; got dt={dt}, beta={beta}, nu={nu}.")

    rng = rng or np.random.default_rng()
    dx = float(dx) if dx is not None else 1.0 / state.shape[-1]
    sigma = np.sqrt(2.0 / beta)

    noise = rng.normal(0.0, np.sqrt(dt/dx), size=state.shape)
    rhs = state + sigma * noise
    return heat_step_implicit(rhs, h=dx, dt=dt, nu=nu)

def multi_sim(
    T_steps,
    u_0, 
    dt,
    dx,
    beta,
    nu,
    rng,
):
    """
    Simulate the 1D stochastic heat equation over multiple timesteps.

    Iteratively advances the stochastic heat equation from an initial condition
    for a specified number of steps, storing the solution at each timestep.

    Args:
        T_steps: Number of timesteps to simulate.
        u_0: Initial field condition with shape (N,) or (M, N).
        dt: Time step size.
        dx: Spatial grid spacing.
        beta: Noise-strength parameter.
        nu: Diffusion coefficient.
        rng: Random number generator for reproducibility.

    Returns:
        List of fields at each timestep, starting with u_0 and including
        T_steps additional states.
    """
    u = [] 
    u.append(u_0)
    u_prev =u_0
    for t in range(0,T_steps):
            u_next = simulate_spde(u_prev, dt=dt, dx=dx, beta=0.1, nu=0.5, rng=rng)
            u.append(u_next)
            u_prev = u_next
    return u



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Demo settings.
    M = 256
    grid = np.linspace(0.0, 1.0, M, endpoint=False)
    dx = 1/M
    weights = np.full(M, dx)

    # Sample an initial condition from the Gaussian field model.
    periodic_kernel = lambda x, y: np.exp((-2 * np.sin(np.pi*(x - y)) ** 2) / 0.2**2)
    mean = lambda x: np.zeros_like(x)
    u0_samples = sample_gaussian_measure(
        kernel=periodic_kernel,
        mean=mean,
        N=50,
        grid=grid,
        weights=weights,
        n_samples=1,
        rng=np.random.default_rng(42),
    )
    u0 = u0_samples[0]
    # Take one stochastic heat step.
    dt = 1e-4
    T_steps = 1000
    u_prev = u0
    rng=np.random.default_rng(seed=2345)
    u=  multi_sim(T_steps,u0,dt,dx,beta=1,nu=0.5,rng=rng)

    print(dt)
    print((dx**2)/0.5)
    print((0.5*dt)/(dx**2))
    # Plot the initial and updated states.
    plot_spde_path(u,grid,T_steps,u0)
