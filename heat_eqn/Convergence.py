import numpy as np


def l2_norm(u, v, dx):
    """
    Compute the L2 norm of the difference between two fields.

    Calculates ||u - v||_L2 using midpoint quadrature on a uniform grid.

    Args:
        u: First field array with shape (N,).
        v: Second field array with shape (N,).
        dx: Spatial grid spacing.

    Returns:
        The L2 norm of the difference (u - v).
    """
    return np.sqrt(np.sum((u - v)**2) * dx)


def run_dt_refinement(
    simulate_fn,
    N=1024,
    T=1.0,
    beta=0.1,
    nu=0.5,
    n=50,
    base_dt=0.01,
    levels=5,
    rng=None,
):
    """
    Run a time-step refinement convergence study for stochastic PDE simulation.

    Evaluates strong convergence by computing strong L2 errors at different timesteps.
    Uses the same noise increments across simulations for variance reduction.
    Compares solutions at coarser timesteps to a fine reference solution.

    Args:
        simulate_fn: Simulation function (e.g., multi_sim) that takes parameters:
            T_steps, u_0, dt, dx, beta, nu, rng and returns trajectory.
        N: Number of spatial grid points.
        T: Final time for simulation (unused; controlled by base_dt and levels).
        beta: Noise strength parameter.
        nu: Diffusion coefficient.
        n: Number of independent samples/realizations.
        base_dt: Coarsest timestep size for refinement study.
        levels: Number of refinement levels (finer timesteps are dt / 2^ell).
        rng: Random number generator for reproducibility.

    Returns:
        List of dicts with keys 'dt' and 'strong_L2', one per refinement level,
        sorted from coarsest to finest timestep.
    """
    L = 1.0
    dx = L / N
    rng = rng or np.random.default_rng(42)
    u0 = np.zeros(N, dtype=float)

    class _PreloadedGenerator:
        """Internal generator that returns preloaded noise increments from a queue.
        
        Used to ensure reproducibility by controlling noise samples in refinement studies.
        """
        def __init__(self, noise_queue):
            self._noise_queue = list(noise_queue)

        def normal(self, loc=0.0, scale=1.0, size=None):
            """Pop and return the next preloaded noise increment.
            
            Args:
                loc: Location parameter (unused, for interface compatibility).
                scale: Scale parameter (unused, for interface compatibility).
                size: Expected shape of noise. Must match stored noise shape.
                
            Returns:
                The next noise increment shifted by loc.
                
            Raises:
                RuntimeError: If the noise queue is empty.
                ValueError: If requested size does not match stored noise shape.
            """
            if not self._noise_queue:
                raise RuntimeError("Ran out of preloaded noise increments.")
            noise = self._noise_queue.pop(0)
            if size is not None and tuple(size) != noise.shape:
                raise ValueError(
                    f"Expected noise shape {size}, got {noise.shape} from preloaded queue."
                )
            return loc + noise

    def _simulate_with_preloaded_noise(noise_queue, steps, dt, outer_rng):
        """Simulate a trajectory using preloaded noise increments for reproducibility.
        
        Temporarily patches np.random.default_rng to inject preloaded noise,
        ensuring the same noise is used across simulations with different timesteps.
        
        Args:
            noise_queue: List of noise increments to use in simulation.
            steps: Number of timesteps to simulate.
            dt: Timestep size.
            outer_rng: Random number generator (unused due to noise patching).
            
        Returns:
            Array of field values at each timestep.
        """
        original_default_rng = np.random.default_rng
        next_index = 0

        def patched_default_rng(*args, **kwargs):
            nonlocal next_index
            if next_index != 0:
                raise RuntimeError("Only one trajectory should be simulated per preloaded noise queue.")
            next_index += 1
            return _PreloadedGenerator(noise_queue)

        np.random.default_rng = patched_default_rng
        try:
            trajectory = simulate_fn(
                T_steps=steps,
                u_0=u0,
                dt=dt,
                dx=dx,
                beta=beta,
                nu=nu,
                rng=outer_rng,
            )
        finally:
            np.random.default_rng = original_default_rng

        return np.asarray(trajectory)

    def _simulate_trajectory_from_noise(noise_increments, dt):
        steps = len(noise_increments)
        outer_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        return _simulate_with_preloaded_noise(noise_increments, steps, dt, outer_rng)

    dt_ref = base_dt / (2**levels)
    steps_ref = max(1, int(np.round(T / dt_ref)))
    dt_values = [base_dt / (2**ell) for ell in range(levels)]

    Xref_T = np.empty((n, N), dtype=float)
    X_levels = {dt: np.empty((n, N), dtype=float) for dt in dt_values}

    for i in range(n):
        noise_ref = rng.normal(0.0, np.sqrt(dt_ref / dx), size=(steps_ref, N))
        noise_ref -= noise_ref.mean(axis=1, keepdims=True)  # remove k=0 contribution

        reference_trajectory = _simulate_trajectory_from_noise(list(noise_ref), dt_ref)
        Xref_T[i] = reference_trajectory[-1]

        for dt in dt_values:
            ratio = int(round(dt / dt_ref))
            if ratio < 1 or steps_ref % ratio != 0:
                raise ValueError(
                    "dt values must be integer multiples of the reference dt_ref. "
                    f"Got dt_ref={dt_ref}, dt={dt}."
                )
            noise_coarse = [
                np.sum(noise_ref[j * ratio : (j + 1) * ratio], axis=0)
                for j in range(steps_ref // ratio)
            ]
            coarse_trajectory = _simulate_trajectory_from_noise(noise_coarse, dt)
            X_levels[dt][i] = coarse_trajectory[-1]

    results = []
    for dt in dt_values:
        XT = X_levels[dt]
        squared_l2_errors = np.array([l2_norm(XT[i], Xref_T[i], dx)**2 for i in range(n)])
        strong_L2 = np.sqrt(np.mean(squared_l2_errors))
        results.append({"dt": dt, "strong_L2": strong_L2})
        print(f"dt={dt:.5f}  strong L2={strong_L2:.4e}")

    return results