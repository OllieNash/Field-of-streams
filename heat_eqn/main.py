from si_ed_1d import simulate_spde
from Convergence import run_dt_refinement, plot_convergence

errors = run_dt_refinement(simulate_spde)
plot_convergence(errors)