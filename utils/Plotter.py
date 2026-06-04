import numpy as np
import matplotlib.pyplot as plt

def plot_strong_convergence(results):
    """
    Plot strong convergence rates for time-step refinement study.

    Displays strong L2 errors on a log-log scale against timestep sizes,
    inverts the x-axis for visualization of refinement progression.

    Args:
        results: List of dicts with keys 'dt' (timestep) and 'strong_L2' (error).
    """
    dts = np.array([r["dt"] for r in results])
    plt.figure()
    plt.loglog(dts, [r["strong_L2"] for r in results], "o-", label="Strong L2")
    #plt.loglog(dts, [r["strong_H1"] for r in results], "s-", label="Strong H1")
    #plt.loglog(dts, [r["strong_space_time"] for r in results], "^-", label="Strong space-time L2")
    plt.gca().invert_xaxis()
    plt.grid(True, which='both')
    plt.xlabel("dt")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Strong Convergence Rates")
    plt.show()

  

def plot_sde_path(path):
    """
    Plot multiple sample paths from a stochastic differential equation.

    Displays trajectories of particles over time with transparency to show
    overlapping paths.

    Args:
        path: Array of particle trajectories with shape (num_particles, num_timesteps).
    """
    for i in range(path.shape[0]):
        plt.plot(path[i, :], alpha=0.7)    
    plt.title("Trajectories of Particles over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Position")
    plt.show()   


def plot_spde_path(u, grid, T_steps, u0):
    """
    Plot the stochastic heat equation solution at multiple timesteps.

    Visualizes the initial condition and solution at t=Δt, t=T/2, and t=T,
    with the final and mid-time solutions mean-centered to show oscillations.
    Note: No diffusion at k=0 mode, so mean performs a pure random walk.

    Args:
        u: List or array of fields at each timestep, shape (num_steps, num_spatial_points).
        grid: Spatial grid points for x-axis.
        T_steps: Total number of timesteps simulated.
        u0: Initial condition field.
    """
    u_mean_final = np.mean(u[-1])
    print(u_mean_final) #No diffusion at k =0 and mean performs a pure random walk, 
    plt.figure(figsize=(8, 4.5), constrained_layout=True)
    plt.plot(grid, u0, lw=1.6, label=r"$u(t=0)$")
    plt.plot(grid, u[1], lw=1.6, label=r"$u(t=\Delta t)$")
    plt.plot(grid, (u[int(T_steps/2)]-np.mean(u[int(T_steps/2)])), lw=1.6, label=r"$u(t=T/2)$")
    plt.plot(grid, (u[-1]-u_mean_final), lw=1.6, label=r"$u(t=T)$")
    plt.title("Stochastic heat equation")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

def plot_dist_sde(path):
    """
    Plot the empirical distribution of SDE solutions.

    Args:
        path: Array of particle trajectories with shape (num_particles, num_timesteps).
    """

    plt.hist(path[:,-1],bins=30,density=True,alpha=0.9,color="tab:orange",edgecolor="white")
    plt.title("Empirical distrubution")
    plt.xlabel("Position")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.show() 

def plot_dist_spde(grid, u0, phi_samples):
    """
    Plot initial condition and empirical stationary distribution of a functional.

    Displays a two-panel figure: (left) the sampled initial condition u0,
    (right) histogram of phi(u) samples from the stationary distribution.

    Args:
        grid: Spatial grid points.
        u0: Initial condition field with shape (num_spatial_points,).
        phi_samples: Array of functional values sampled from stationary distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axes[0].plot(grid, u0, lw=1.5, color="tab:blue")
    axes[0].set_title("Sampled initial condition")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(phi_samples, bins=30, density=True, alpha=0.8, color="tab:orange", edgecolor="white")
    axes[1].set_title("Empirical stationary distribution of phi(u)")
    axes[1].set_xlabel(r"$u_0$")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.3)

    plt.show()
