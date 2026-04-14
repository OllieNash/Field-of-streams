Recall Stochastic heat equation $$\partial_t U(t,x) = \nu \partial_{xx} U(t,x) + \sigma \dot B(t,x)$$

##Spatial discretisation via fourier##

We take the fourier transform as the fourier nodes are the eigenfunctions of the semi-group operator. 
So we can write $$U(t,x) = \sum_{k \in \mathbb{Z}} \hat U_k(t)e^{ikx}$$ 
Where for each node we have the fourier transform of the mild solution: 
$$ \hat U_k(t) = e^{-\nu |k|^2 t} \hat U_k(t) + \sigma \int_0^t e^{-\nu |k|^2(t-s)} d \hat B_k(s)$$
For each $$k \in \mathbb{Z}$$ this is a 1D Ornstein-Uhlenbeck process. We perform the update step on each of these nodes seperately and then apply an inverse fourier transform.


##Time discretisation - Semi-implicit Euler-Maruyama## 

Let $$h >0$$ be the time step and denote $$\hat X^{n}$$, then our update from $$t_n to t_{n+1}$$ is 
$$ \hat X^{n_1}(k_m) = \frac{\hat X^{n}(k_m) + \sigma \hat \Delta B^n(k_m)}{1 - h\nu \lambda_m} $$

We see the strength of using fourier as in physical space: 
$$(I - h\nu\partial_{xx}) X^{n+1} = X^n + \sigma \DeltaB^n $$ is the update and we have to solve a large linear operator. 
On tours, $$\partial_{xx}$$ diagonalises and makes computation much nicer. 

##Noise discretisation$$
In physical sapce, the noise i.i.d across space and time so $$\Delta B^n(x_j) \sim \mathcal{N}(0,h)$$
In fourier space, it is the same and we have $$\Delta \hat B^n(k_m) \sim \mathcal{N}(0,h)$$


