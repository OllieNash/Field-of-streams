import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heat_eqn.si_ed_1d import simulate_spde
from heat_eqn.Convergence import run_dt_refinement
from Plotter import plot_convergence

errors = run_dt_refinement(simulate_spde)
plot_convergence(errors)