# Finite Element Discretization of the Stochastic Heat Equation

We consider the stochastic heat equation on the periodic domain $D = [0, L)$ with additive
space–time white noise:
$$du = \nu \nabla^2 u \, dt + \sigma \, dW, \qquad \sigma = \sqrt{2/\beta},$$
where $W$ is a cylindrical Wiener process on $L^2(D)$, so that formally $\dot W = \xi$ is
space–time white noise. This is implemented in `heat_eqn/stochastic_heat_fem.py` using
NumPy/SciPy (element-by-element assembly of the P1 matrices below, sparse LU solve), so
Firedrake is not required.

---

## Time Discretization

Let $t_n = n\Delta t$. The Brownian increment over one step is
$$\Delta W^{n} = W(t_{n+1}) - W(t_{n}),$$
which is a Gaussian field with $\mathbb{E}\langle \Delta W^n, f\rangle\langle \Delta W^n, g\rangle = \Delta t \, \langle f, g\rangle$.

The forward Euler–Maruyama scheme is
$$u^{n+1} = u^{n} + \Delta t \, \nu \nabla^2 u^{n} + \sigma \Delta W^{n}.$$
This is only stable when $\Delta t \lesssim h^2 / (2\nu)$ (with $h$ the mesh size), so we use
the more general $\theta$-scheme
$$u^{n+1} = u^{n} + \Delta t \, \nu \nabla^2 \big(\theta u^{n+1} + (1-\theta) u^{n}\big) + \sigma \Delta W^{n},$$
where $\theta = 0$ is forward Euler–Maruyama and $\theta = 1$ is the semi-implicit
(drift-implicit) scheme. The semi-implicit scheme is unconditionally stable and matches the spectral
solver in `stochastic_heat.py`.

---

## Weak Form (Integrated Against Test Function $v$)

Multiply by a test function $v$ and integrate over $D$. Integrating the Laplacian by parts
(the boundary term vanishes on a periodic domain) gives a minus sign:
$$\int_D \nabla^2 u \, v \, dx = -\int_D \nabla u \cdot \nabla v \, dx.$$

Find $u^{n+1} \in V_h$ such that for all $v \in V_h$
$$\int u^{n+1} v \, dx + \theta \Delta t \, \nu \int \nabla u^{n+1} \cdot \nabla v \, dx
= \int u^{n} v \, dx - (1-\theta) \Delta t \, \nu \int \nabla u^{n} \cdot \nabla v \, dx + \sigma \int v \, dW^{n}.$$

We take $V_h$ to be continuous piecewise-linear (P1) functions on a uniform periodic mesh
with $N$ cells, $h = L/N$.

---

## Matrix Form

Writing $u^n = \sum_j U^n_j \phi_j$ in the hat-function basis $\{\phi_j\}$, with mass and
stiffness matrices
$$M_{ij} = \int \phi_i \phi_j \, dx, \qquad K_{ij} = \int \nabla\phi_i \cdot \nabla\phi_j \, dx,$$
the scheme reads
$$(M + \theta \Delta t \, \nu K) \, U^{n+1} = (M - (1-\theta) \Delta t \, \nu K) \, U^{n} + \sigma \, b^{n},
\qquad b^n_i = \langle \Delta W^n, \phi_i \rangle.$$

---

## Sampling the Noise

The load vector $b^n$ is Gaussian with mean zero and covariance
$$\mathbb{E}[b^n_i b^n_j] = \Delta t \int \phi_i \phi_j \, dx = \Delta t \, M_{ij}.$$
Exact sampling would use $b^n = \sqrt{\Delta t}\, L z$ with $M = L L^\top$ and $z \sim N(0, I)$.
We instead **lump the mass matrix**, $M \approx \mathrm{diag}(m_i)$ with $m_i = \int \phi_i \, dx = h$,
which gives independent entries
$$b^n_i = \sqrt{\Delta t \, h} \; z_i = h \cdot \zeta_i, \qquad \zeta_i \sim N(0, \Delta t / h).$$
The nodal values $\zeta_i$ are exactly the noise used by the finite-difference/spectral solver,
so both solvers can be driven by the same noise increments.

---

## Stationary Law (Sanity Check)

Excluding the spatial mean (which performs a Brownian motion), the solution converges to a
centred Gaussian measure with
$$\mathbb{E}\, \| u - \bar u \|_{L^2}^2 = \sum_{k \neq 0} \frac{\sigma^2}{2 \nu \, (2\pi k / L)^2}
= \frac{\sigma^2 L^2}{24 \, \nu}.$$
