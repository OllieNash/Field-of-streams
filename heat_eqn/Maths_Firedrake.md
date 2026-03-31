# Discretization of the Stochastic Heat Equation

We consider the stochastic heat equation on space–time with additive space–time white noise:
$$\frac{\partial u}{\partial t} = \nabla^2 u + \xi$$
where $$\xi = \xi(t,x)$$ denotes space–time white noise.

---

## Time Discretization

Applying the forward Euler–Maruyama scheme in time with step size \(\Delta t\), we obtain
$$u^{n+1} = u^{n} + \Delta t \,\nabla^2 u^{n} + \Delta \xi^{n}$$
where the noise increment is
$$\Delta \xi^{n} = \xi^{n+1} - \xi^{n}$$

---

## Weak Form (Integrated Against Test Function \(v\))

Let $$v$$ be a test function. Multiply both sides by $$v$$ and integrate over the spatial domain:

$$\int u^{n+1} vdx = \int u^{n} vdx + \Delta t \int \nabla u^{n} \cdot \nabla vdx + \Delta t \int v \, \nabla \xi^{n}dx$$

This gives the weak formulation corresponding to the Euler–Maruyama time discretization of the stochastic heat equation.
