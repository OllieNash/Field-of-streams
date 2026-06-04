import numpy as np
import matplotlib.pyplot as plt

from sample_gaussian import sample_gaussian_measure


def heat_step_implicit(state: np.ndarray, h: float, dt: float, nu: float) -> np.ndarray:
    r"""
    Advance the 1D heat equation by one implicit-Euler step using FFT (periodic BCs).

    Solves one timestep of the heat equation on a periodic domain using the
    implicit Euler scheme:

        u^{n+1} = (I - dt * nu * \Delta)^{-1} u^n

    The inverse is computed efficiently via FFT by solving diagonally in
    Fourier space.

    Args:
        state: Initial field with shape (N,) or (M, N).
        h: Spatial grid spacing (positive).
        dt: Time step size (non-negative).
        nu: Diffusion coefficient (non-negative).

    Returns:
        The field one implicit Euler step later, with the same shape as `state`.

    Raises:
        ValueError: If state is not 1D or 2D, or if h <= 0, dt < 0, or nu < 0.
    """
    state = np.asarray(state, dtype=float)
    if state.ndim not in (1, 2):
        raise ValueError(f"state must be 1D or 2D, got shape {state.shape}.")
    if h <= 0 or dt < 0 or nu < 0:
        raise ValueError(f"Require h>0, dt>=0, nu>=0; got h={h}, dt={dt}, nu={nu}.")

    m = state.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(m, d=h)
    denom = 1.0 + dt * nu * (k**2)

    is_vector = state.ndim == 1
    state_2d = state[None, :] if is_vector else state
    state_hat = np.fft.fft(state_2d, axis=-1)
    next_hat = state_hat / denom[None, :]
    next_state = np.fft.ifft(next_hat, axis=-1).real

    return next_state[0] if is_vector else next_state


if __name__ == "__main__":
    M = 256
    grid = np.linspace(0.0, 1.0, M,endpoint=False)
    h = 1/M
    weights = np.full(M, h)

    se_kernel = lambda x, y: np.exp(-0.5 * (x - y) ** 2 / 0.05 ** 2)
    mean = lambda x: np.zeros_like(x)

    samples = sample_gaussian_measure(
        kernel=se_kernel,
        mean=mean,
        N=50,
        grid=grid,
        weights=weights,
        n_samples=5,
        rng=np.random.default_rng(42),
    )

    u0 = samples[3]
    u1 = heat_step_implicit(u0, h=h, dt=1e-3, nu=1.0)
    u2 = heat_step_implicit(u1, h=h, dt=1e-3, nu=1.0)

    plt.figure(figsize=(8, 4.5), constrained_layout=True)
    plt.plot(grid, u0, lw=1.5, label=r"$u(t=0)$")
    plt.plot(grid, u1, lw=1.5, label=r"$u(t=\Delta t)$")
    plt.plot(grid, u2, lw=1.5, label=r"$u(t=\Delta t)$")
    plt.title("Heat step demo")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()