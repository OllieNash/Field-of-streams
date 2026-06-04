import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heat_eqn.stochastic_heat import multi_sim  # noqa: E402
from heat_eqn.Convergence import run_dt_refinement  # noqa: E402
from Plotter import plot_strong_convergence  # noqa: E402

errors = run_dt_refinement(multi_sim)
plot_strong_convergence(errors)