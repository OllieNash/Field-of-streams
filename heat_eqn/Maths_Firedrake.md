So we first discretize the Stochastic heat equation (white noise and u are dependent on time and space)
$$ \frac{\partial u}{\partial t} = \nabla^2 u + \xi $$ 
Where we apply a forward Euler-Maruyama in time as 
$$u^{n+1} = u^n + \Delta t \nabla^2 u^n + \Delta \xi^n$$
Where 
$$ \Delta \xi^n = \xi^{n+1} - \xi^{n} $$
So integrating against some test function v, we can write this as 
$$ \int u^{n+1}v dx = \int u^n v dx + \Delta t \int \nabla u^n \nabla v dx + \Delta t \int v \nabla \xi^n dx $$
