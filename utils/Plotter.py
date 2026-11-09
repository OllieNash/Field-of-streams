import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Shared chart styling: categorical series colours (used in this fixed order),
# ink/grid colours, and a blue <-> red diverging map with a grey midpoint.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9c2626"]
)


def _style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply the shared look: recessive grid and axes, ink-coloured text."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(which="both", colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="semibold")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2)


def _finish(fig, save_path=None):
    """Save the figure (if a path is given) and show it."""
    fig.patch.set_facecolor(SURFACE)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, facecolor=SURFACE)
        print(f"Saved {save_path}")
    plt.show()

def plot_strong_convergence(results, save_path=None, ref_order=0.25):
    """
    Plot strong convergence rates for time-step refinement study.

    Displays strong L2 errors on a log-log scale against timestep sizes, with
    the least-squares fitted order and a reference slope (the expected strong
    order 1/4 in time for the stochastic heat equation with space-time white
    noise).

    Args:
        results: List of dicts with keys 'dt' (timestep) and 'strong_L2' (error).
        save_path: Optional path to save the figure (PNG).
        ref_order: Order of the reference slope line.
    """
    dts = np.array([r["dt"] for r in results])
    errs = np.array([r["strong_L2"] for r in results])
    order = np.polyfit(np.log(dts), np.log(errs), 1)[0]

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ref = errs[0] * (dts / dts[0]) ** ref_order
    ax.loglog(dts, ref, "--", color=MUTED, lw=1.2)
    ax.annotate(rf"$\propto \Delta t^{{{ref_order:g}}}$", (dts[-1], ref[-1]),
                textcoords="offset points", xytext=(6, -12), color=INK_2)
    ax.loglog(dts, errs, "o-", color=SERIES[0], lw=2, ms=7)
    ax.text(0.03, 0.05, f"fitted order {order:.2f}", transform=ax.transAxes, color=SERIES[0])
    ax.invert_xaxis()
    _style_axes(ax, "Strong convergence in time", r"$\Delta t$", r"strong $L^2$ error at $T$")
    ax.grid(True, which="both", color=GRID, lw=0.6)
    _finish(fig, save_path)

  

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


def plot_fem_vs_spectral(grid, times, u_fem, u_spec, save_path=None):
    """
    Compare FEM and spectral solutions driven by the same noise.

    Four panels: (a) space-time map of the FEM solution, (b) space-time map of
    the difference FEM - spectral, (c) snapshots at t = 0, T/2, T with FEM
    solid and spectral dashed, (d) relative L2 difference over time.

    Args:
        grid: Spatial grid points, shape (N,).
        times: Times of the stored states, shape (T_steps + 1,).
        u_fem: FEM states, shape (T_steps + 1, N).
        u_spec: Spectral states, shape (T_steps + 1, N).
        save_path: Optional path to save the figure (PNG).
    """
    u_fem, u_spec = np.asarray(u_fem), np.asarray(u_spec)
    diff = u_fem - u_spec
    extent = [grid[0], grid[-1] + (grid[1] - grid[0]), times[0], times[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for ax, field, title, label in (
        (axes[0, 0], u_fem, "(a) FEM solution u(t, x)", "u"),
        (axes[0, 1], diff, "(b) FEM − spectral", "difference"),
    ):
        vmax = np.max(np.abs(field))
        im = ax.imshow(field, origin="lower", aspect="auto", extent=extent,
                       cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest")
        _style_axes(ax, title, "x", "t")
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label(label, color=INK_2)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)

    ax = axes[1, 0]
    n = len(times) - 1
    for colour, idx in zip(SERIES, (0, n // 2, n)):
        ax.plot(grid, u_fem[idx], color=colour, lw=2, label=f"t = {times[idx]:.3g}")
        ax.plot(grid, u_spec[idx], color=INK, lw=1, ls="--")
    ax.plot([], [], color=INK, lw=1, ls="--", label="spectral (dashed)")
    _style_axes(ax, "(c) Snapshots: FEM (solid) vs spectral (dashed)", "x", "u(x)")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9)

    ax = axes[1, 1]
    rel = np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(u_spec, axis=1), 1e-300)
    ax.plot(times, rel, color=SERIES[0], lw=2)
    _style_axes(ax, r"(d) Relative $L^2$ difference ‖FEM − spectral‖ / ‖spectral‖", "t", "relative difference")
    ax.set_ylim(bottom=0)

    fig.suptitle("Stochastic heat equation: P1 finite elements vs spectral, same noise",
                 color=INK, fontsize=13, fontweight="semibold")
    _finish(fig, save_path)


def plot_stationary_energy(times, energy, theory, save_path=None, n_paths_shown=20):
    r"""
    Plot E||u - mean(u)||^2 over time against its stationary value.

    Shows a few individual paths (grey), the ensemble mean with a 95%
    confidence band, and the theoretical stationary value as a dashed line.

    Args:
        times: Times, shape (T_steps + 1,).
        energy: ||u - mean(u)||_L2^2 per path, shape (T_steps + 1, R).
        theory: Theoretical stationary value.
        save_path: Optional path to save the figure (PNG).
        n_paths_shown: Number of individual paths drawn in the background.
    """
    energy = np.asarray(energy)
    mean = energy.mean(axis=1)
    half_width = 1.96 * energy.std(axis=1) / np.sqrt(energy.shape[1])

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(times, energy[:, :n_paths_shown], color=MUTED, lw=0.6, alpha=0.35)
    ax.fill_between(times, mean - half_width, mean + half_width, color=SERIES[0], alpha=0.2, lw=0)
    ax.plot(times, mean, color=SERIES[0], lw=2)
    ax.axhline(theory, color=INK, lw=1.2, ls="--")

    ax.annotate(f"ensemble mean, {energy.shape[1]} paths (95% band)", (times[-1], mean[-1]),
                textcoords="offset points", xytext=(-4, 8), ha="right", color=SERIES[0])
    ax.annotate(rf"theory $\sigma^2 L^2 / (24\nu)$ = {theory:.4f}", (times[0], theory),
                textcoords="offset points", xytext=(4, 6), color=INK)
    ax.annotate("individual paths", (times[len(times) // 3], energy[len(times) // 3, :n_paths_shown].max()),
                textcoords="offset points", xytext=(0, 4), color=MUTED, fontsize=9)
    ax.set_ylim(0, np.percentile(energy[:, :n_paths_shown], 99.5) * 1.1)
    _style_axes(ax, r"FEM stationary check: $\mathbb{E}\,\|u - \bar u\|_{L^2}^2$ over time",
                "t", r"$\|u - \bar u\|_{L^2}^2$")
    _finish(fig, save_path)

