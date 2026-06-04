
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.Plotter import plot_sde_path, plot_dist_sde

def drift(x, mu, theta):
    return theta * (mu - x)


def diffusion(beta):
    return np.sqrt(2 / beta)


def drift_sl(x, chi):
    return x * (chi - x ** 2)


def initial_positions(n_particles, num_steps, x0_std=0.5, rng=None):
    """Inputs: 
    n_particles - number of particles
    num_steps - number of timesteps
    x0_std - the standard deviation of the initial positions of the particles
    rng - rng regime chosen 
    
    Output:
    x - n_particles x num_steps array, randomised initial positions of every 
    particle and zero for future steps"""
    rng = np.random.default_rng() if rng is None else rng
    x = np.zeros((n_particles, num_steps + 1))
    x[:, 0] = rng.normal(0.0, x0_std, size=n_particles)
    return x


def non_markovian(
    n_particles,
    t_final,
    h,
    chi,
    beta,
    x0_std=0.5,
    rng=None,
):
    num_steps = int(t_final / h)
    rng = np.random.default_rng() if rng is None else rng
    x = initial_positions(n_particles, num_steps, x0_std=x0_std, rng=rng)
    sigma = diffusion(beta)

    for t in range(num_steps):
        dw_1 = rng.normal(0.0, np.sqrt(h), size=n_particles)
        dw_2 = rng.normal(0.0, np.sqrt(h), size=n_particles)
        x[:, t + 1] = x[:, t] + drift_sl(x[:, t], chi) * h + sigma * (dw_1 + dw_2) / 2.0

    return x


def euler_maruyama(
    n_particles,
    t_final,
    h,
    theta,
    mu=0.0,
    beta=50.0,
    x0_std=0.5,
    rng=None,
):
    num_steps = int(t_final / h)
    rng = np.random.default_rng() if rng is None else rng
    x = initial_positions(n_particles, num_steps, x0_std=x0_std, rng=rng)
    sigma = diffusion(beta)

    for t in range(num_steps):
        dw = rng.normal(0.0, np.sqrt(h), size=n_particles)
        x[:, t + 1] = x[:, t] + drift(x[:, t], mu, theta) * h + sigma * dw

    return x

def heun(
    n_particles,
    t_final,
    h,
    drift_fn,      
    diffusion_fn,   
    x0_std=0.5,
    rng=None,
):
    """Inputs: 
    n_particles - number of particles
    t_final - time the solver runs for
    h - size of time steps
    drift_fn - the drift function chosen (takes input of 1D array of positions and current time)
    diffusion_fn - the diffusion function chosen (takes input of 1D array of positions and current time)
    x0_std - the standard deviation of the initial positions of the particles
    rng - rng regime chosen 
    
    Output:
    x - n_particles x num_steps array, positions of particles at each timestep"""
    num_steps = int(t_final / h)
    rng = np.random.default_rng() if rng is None else rng
    x = initial_positions(n_particles, num_steps, x0_std=x0_std, rng=rng)

    for t in range(num_steps):
        t_now = t * h
        dw = rng.normal(0.0, np.sqrt(h), size=n_particles)

        # Predictor (Euler step)
        f0 = drift_fn(x[:, t], t_now)
        g0 = diffusion_fn(x[:, t], t_now)
        x_pred = x[:, t] + f0 * h + g0 * dw

        # Corrector
        f1 = drift_fn(x_pred, t_now + h)
        g1 = diffusion_fn(x_pred, t_now + h)
        x[:, t + 1] = x[:, t] + 0.5 * (f0 + f1) * h + 0.5 * (g0 + g1) * dw

    return x


if __name__ == "__main__":
    n_particles = 1000
    t_final = 100.0
    h = 0.01
    theta = 5.0
    mu = 0.0
    beta = 50.0

    x_em = euler_maruyama(
        n_particles=n_particles,
        t_final=t_final,
        h=h,
        theta=theta,
        mu=mu,
        beta=beta,
    )

    x_heun = heun(
        n_particles=n_particles,
        t_final=t_final,
        h=h,
        drift_fn=lambda x, t: drift(x, mu, theta),
        diffusion_fn=lambda x, t: np.full_like(x, diffusion(beta)),
    )

    plot_sde_path(path=x_em)
    plot_dist_sde(path=x_em)
    plot_sde_path(path=x_heun)
    plot_dist_sde(path=x_heun)

