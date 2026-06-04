# Field-of-streams

This repository contains reference code and experiments for simulating stochastic
partial differential equations (SPDEs) and stochastic differential equations
(SDEs) in 1D. The focus is on spectral and finite-element style schemes and on
empirically testing weak and strong convergence properties.

Quick overview
--------------
- Language: Python 3.8+
- Minimal dependencies: `numpy`, `matplotlib`

Install
-------
Create a virtual environment and install the basics:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install numpy matplotlib
```

Folder structure
----------------
- `heat_eqn/` — SPDE simulation code and experiments
	- `Convergence.py` — time-step refinement study utilities and strong-L2 error measurement
	- `main.py` — experiment runners (entry points for demo simulations)
	- `stochastic_heat.py` — one-step and multi-step stochastic heat solver (`simulate_spde`, `multi_sim`)
	- `stationary_heat.py` — scripts to explore stationary behaviour (utilities)
	- `maths/` — supporting math notes and derivations

- `utils/` — small helper utilities
	- `heat.py` — implicit FFT-based heat-step solver (`heat_step_implicit`)
	- `sample_gaussian.py` — Gaussian field sampling helpers
	- `hermite.py`, `hermite_inner_product.py` — Hermite-related utilities
    - `Plotter.py` — plotting helpers for convergence and sample paths

- Root-level scripts and notes
	- `simulations_sdes.py` — SDE simulation experiments
	- `Plan.md`, `papers.md` — project notes, plans and references

What each component does
-----------------------
- `heat_step_implicit` (in `utils/heat.py`): advances a spatial field one
	implicit-Euler timestep using FFT on a periodic grid.
- `simulate_spde` / `multi_sim` (in `heat_eqn/stochastic_heat.py`): construct a
	single-step (and multi-step) integrator for the stochastic heat equation and
	convenience functions for producing trajectories.
- `Convergence.run_dt_refinement`: runs a refinement experiment re-using the
	same noise increments across resolutions and reports strong-L2 errors.
- `Plotter.py`: convenience plotting functions used by notebooks and demo
	scripts to visualise paths, fields and convergence curves.

Running the demos
-----------------
- To run the heat-step demo embedded in `utils/heat.py` for deterministic heat equation to help with understanding.

```bash
python utils/heat.py
```

- To run other experiments, open `heat_eqn/main.py` or call the functions from a
	small driver script or an interactive session (Jupyter/REPL). The code is
	modular so you can import `simulate_spde`, `multi_sim`, `run_dt_refinement`,
	or plotting helpers directly.


License / contact
-----------------
This repo is a research codebase; add a license or contact details as
appropriate for your project.

