import numpy as np


def drift(x, mu, theta):
    return theta * (mu - x)


def diffusion(beta):
    return np.sqrt(2 / beta)


def drift_sl(x, chi):
    return x * (chi - x ** 2)


def initial_positions(n_particles, num_steps, x0_std=0.5, rng=None):
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_particles = 3
    t_final = 100.0
    h = 0.01
    theta = 5.0
    mu = 0.0
    beta = 50.0

    x = euler_maruyama(
        n_particles=n_particles,
        t_final=t_final,
        h=h,
        theta=theta,
        mu=mu,
        beta=beta,
    )

    for i in range(x.shape[0]):
        plt.plot(x[i, :], alpha=0.7)
        
    plt.title("Trajectories of Particles over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Position")
    plt.show()   



