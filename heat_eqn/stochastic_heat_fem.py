r"""Finite element stochastic heat equation solver in 1D.

This module advances a spatial field under the stochastic heat equation

    du = nu * u_xx dt + sigma dW,

using continuous piecewise-linear (P1) finite elements on a uniform periodic
mesh and a theta-scheme Euler-Maruyama time step. The derivation is in
`heat_eqn/maths/Maths_Firedrake.md`.

The main entry points are `simulate_spde_fem` (one step) and `multi_sim_fem`
(many steps), which have the same interface as `simulate_spde` / `multi_sim`
in `stochastic_heat.py`: fields are arrays on the grid x_j = j * dx, and noise
is drawn as `rng.normal(0, sqrt(dt/dx), size=...)`. This means the FEM solver
can be passed directly to `Convergence.run_dt_refinement` and driven by the
same noise increments as the spectral solver.

When run as a script, this file compares the FEM and spectral solvers on the
same noise and checks the FEM stationary variance against theory, saving the
figures to `figures/`.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def assemble_p1_periodic(N: int, L: float = 1.0) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    r"""
    Assemble the P1 mass and stiffness matrices on a uniform periodic mesh.

    The mesh has N cells [x_e, x_{e+1}] with x_e = e * L / N, and node N is
    identified with node 0. On each cell of width h the local matrices are

        M_loc = h/6 * [[2, 1], [1, 2]],    K_loc = 1/h * [[1, -1], [-1, 1]],

    and these are summed into the global matrices.

    Args:
        N: Number of cells (= number of nodes / DOFs).
        L: Domain length.

    Returns:
        (M, K): sparse mass and stiffness matrices, shape (N, N).
    """
    h = L / N
    M_loc = h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    K_loc = 1.0 / h * np.array([[1.0, -1.0], [-1.0, 1.0]])

    rows, cols, m_vals, k_vals = [], [], [], []
    for e in range(N):
        nodes = (e, (e + 1) % N)
        for a in range(2):
            for b in range(2):
                rows.append(nodes[a])
                cols.append(nodes[b])
                m_vals.append(M_loc[a, b])
                k_vals.append(K_loc[a, b])

    # coo -> csr sums duplicate entries, which is exactly the assembly step.
    M = sp.coo_matrix((m_vals, (rows, cols)), shape=(N, N)).tocsr()
    K = sp.coo_matrix((k_vals, (rows, cols)), shape=(N, N)).tocsr()
    return M, K


class FEMStochasticHeat:
    r"""
    P1 finite element theta-scheme for the periodic 1D stochastic heat equation.

    Each step solves

        (M + theta*dt*nu*K) U^{n+1} = (M - (1-theta)*dt*nu*K) U^n + sigma * b^n,

    where M, K are the mass and stiffness matrices and b^n is the lumped-mass
    white-noise load vector, b^n_i = m_i * zeta_i with m_i = sum_j M_ij and
    zeta_i ~ N(0, dt/dx). The left-hand side is LU-factorised once.

    Args:
        N: Number of mesh cells (= number of grid points / DOFs).
        dt: Time step size.
        beta: Noise-strength parameter, sigma = sqrt(2 / beta).
        nu: Diffusion coefficient.
        theta: 0 for forward Euler-Maruyama, 1 for semi-implicit (default).
        L: Domain length.
    """

    def __init__(
        self,
        N: int,
        dt: float,
        beta: float = 1,
        nu: float = 0.5,
        theta: float = 1.0,
        L: float = 1.0,
    ):
        if dt < 0 or beta <= 0 or nu < 0 or not 0 <= theta <= 1:
            raise ValueError(
                f"Require dt>=0, beta>0, nu>=0, 0<=theta<=1; "
                f"got dt={dt}, beta={beta}, nu={nu}, theta={theta}."
            )
        self.N, self.dt, self.L = N, dt, L
        self.dx = L / N
        self.sigma = np.sqrt(2.0 / beta)

        M, K = assemble_p1_periodic(N, L)
        self.lu = splu((M + theta * dt * nu * K).tocsc())
        self.B = (M - (1.0 - theta) * dt * nu * K).tocsr()
        self.m_lumped = np.asarray(M.sum(axis=1)).ravel()

    def step(self, state: np.ndarray, rng) -> np.ndarray:
        """
        Advance one time step.

        Args:
            state: Field with shape (N,) or a batch with shape (R, N).
            rng: Random number generator (or `Convergence.PreloadedGenerator`).

        Returns:
            The field one step later, same shape as `state`.
        """
        noise = rng.normal(0.0, np.sqrt(self.dt / self.dx), size=state.shape)
        # Work with nodes along axis 0 so batches are solved in one call.
        U, Z = state.T, noise.T
        load = self.m_lumped[:, None] * Z if U.ndim == 2 else self.m_lumped * Z
        return self.lu.solve(self.B @ U + self.sigma * load).T


@lru_cache(maxsize=16)
def _get_problem(N, dt, beta, nu, theta, L):
    """Build (and cache) the FEM problem so repeated runs reuse the LU factorisation."""
    return FEMStochasticHeat(N, dt, beta=beta, nu=nu, theta=theta, L=L)


def _domain_length(N: int, dx: float | None) -> float:
    return 1.0 if dx is None else float(dx) * N


def simulate_spde_fem(
    initial_state: np.ndarray,
    dt: float = 1e-3,
    dx: float | None = None,
    beta: float = 1,
    nu: float = 0.5,
    rng: np.random.Generator | None = None,
    theta: float = 1.0,
) -> np.ndarray:
    """
    Advance one timestep of the 1D stochastic heat equation with P1 FEM.

    Args:
        initial_state: Initial field with shape (N,) or (R, N) on the grid x_j = j * dx.
        dt: Time step size.
        dx: Spatial grid spacing. If omitted, uses 1 / N.
        beta: Noise-strength parameter.
        nu: Diffusion coefficient.
        rng: Random number generator for reproducibility.
        theta: 0 for forward Euler-Maruyama, 1 for semi-implicit.

    Returns:
        The field one stochastic heat step later, same shape as `initial_state`.
    """
    state = np.asarray(initial_state, dtype=float)
    if state.ndim not in (1, 2):
        raise ValueError(f"initial_state must be 1D or 2D, got shape {state.shape}.")
    rng = rng or np.random.default_rng()
    N = state.shape[-1]
    problem = _get_problem(N, dt, beta, nu, theta, _domain_length(N, dx))
    return problem.step(state, rng)


def multi_sim_fem(
    T_steps,
    u_0,
    dt,
    dx,
    beta,
    nu,
    rng,
    theta=1.0,
):
    """
    Simulate the 1D stochastic heat equation over multiple timesteps with P1 FEM.

    Same interface as `stochastic_heat.multi_sim`, so it can be used with
    `Convergence.run_dt_refinement` and the plotting helpers.

    Args:
        T_steps: Number of timesteps to simulate.
        u_0: Initial field with shape (N,) or a batch with shape (R, N).
        dt: Time step size.
        dx: Spatial grid spacing.
        beta: Noise-strength parameter.
        nu: Diffusion coefficient.
        rng: Random number generator for reproducibility.
        theta: 0 for forward Euler-Maruyama, 1 for semi-implicit.

    Returns:
        List of fields at each timestep, starting with u_0 and including
        T_steps additional states.
    """
    u_0 = np.asarray(u_0, dtype=float)
    N = u_0.shape[-1]
    problem = _get_problem(N, dt, beta, nu, theta, _domain_length(N, dx))

    u = [u_0]
    for _ in range(T_steps):
        u.append(problem.step(u[-1], rng))
    return u


def stationary_energy_theory(beta: float, nu: float, L: float = 1.0) -> float:
    r"""E||u - mean(u)||_L2^2 under the stationary law: sigma^2 L^2 / (24 nu)."""
    return (2.0 / beta) * L**2 / (24.0 * nu)


if __name__ == "__main__":
    from heat_eqn.Convergence import PreloadedGenerator
    from heat_eqn.stochastic_heat import multi_sim
    from sample_gaussian import sample_gaussian_measure
    from Plotter import plot_fem_vs_spectral, plot_stationary_energy

    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)

    # Demo settings (same as the spectral demo in stochastic_heat.py).
    M = 256
    grid = np.linspace(0.0, 1.0, M, endpoint=False)
    dx = 1 / M
    weights = np.full(M, dx)
    dt = 1e-4
    T_steps = 1000
    beta, nu = 1, 0.5

    periodic_kernel = lambda x, y: np.exp((-2 * np.sin(np.pi * (x - y)) ** 2) / 0.2**2)
    mean = lambda x: np.zeros_like(x)
    u0 = sample_gaussian_measure(
        kernel=periodic_kernel,
        mean=mean,
        N=50,
        grid=grid,
        weights=weights,
        n_samples=1,
        rng=np.random.default_rng(42),
    )[0]

    # 1. Drive the FEM and spectral solvers with the same noise increments.
    rng = np.random.default_rng(seed=2345)
    noise = list(rng.normal(0.0, np.sqrt(dt / dx), size=(T_steps, M)))
    u_fem = np.asarray(multi_sim_fem(T_steps, u0, dt, dx, beta, nu, rng=PreloadedGenerator(noise)))
    u_spec = np.asarray(multi_sim(T_steps, u0, dt, dx, beta, nu, rng=PreloadedGenerator(noise)))
    times = dt * np.arange(T_steps + 1)

    diff = np.sqrt(np.sum((u_fem[-1] - u_spec[-1]) ** 2) * dx)
    norm = np.sqrt(np.sum(u_spec[-1] ** 2) * dx)
    print(f"||u_fem(T) - u_spectral(T)||_L2 = {diff:.3e}  (relative {diff / norm:.3e})")
    plot_fem_vs_spectral(grid, times, u_fem, u_spec, save_path=figures / "fem_vs_spectral.png")

    # 2. Ensemble run from u = 0: E||u - mean||^2 should approach sigma^2/(24 nu).
    #    The ensemble is stepped as one (R, M) batch; only the energy is stored.
    R, dt_ens, steps_ens = 400, 1e-4, 2000
    ens_rng = np.random.default_rng(7)
    u = np.zeros((R, M))
    energy = np.zeros((steps_ens + 1, R))
    for n in range(1, steps_ens + 1):
        u = simulate_spde_fem(u, dt=dt_ens, dx=dx, beta=beta, nu=nu, rng=ens_rng)
        energy[n] = np.sum((u - u.mean(axis=-1, keepdims=True)) ** 2, axis=-1) * dx
    plot_stationary_energy(
        dt_ens * np.arange(steps_ens + 1),
        energy,
        theory=stationary_energy_theory(beta, nu),
        save_path=figures / "fem_stationary_energy.png",
    )
