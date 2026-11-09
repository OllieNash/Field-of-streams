"""Strong convergence study for the stochastic heat equation.

Usage (from the repository root):

    python heat_eqn/main.py          # spectral solver
    python heat_eqn/main.py --fem    # P1 finite element solver
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heat_eqn.stochastic_heat import multi_sim  # noqa: E402
from heat_eqn.stochastic_heat_fem import multi_sim_fem  # noqa: E402
from heat_eqn.Convergence import run_dt_refinement  # noqa: E402
from Plotter import plot_strong_convergence  # noqa: E402

use_fem = "--fem" in sys.argv
name = "fem" if use_fem else "spectral"
figures = ROOT / "figures"
figures.mkdir(exist_ok=True)

errors = run_dt_refinement(multi_sim_fem if use_fem else multi_sim)
plot_strong_convergence(errors, save_path=figures / f"strong_convergence_{name}.png")
