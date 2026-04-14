Recall Stochastic heat equation $$\partial_t U(t,x) = \nu \partial_{xx} U(t,x) + \sigma \dot B(t,x)$$

#Spatial discretisation via fourier

<p> We take the fourier transform as the fourier nodes are the eigenfunctions of the semi-group operator. <br>
So we can write <br> 
$$U(t,x) = \sum_{k \in \mathbb{Z}} \hat U_k(t)e^{ikx}$$ <br>
Where for each node we have the fourier transform of the mild solution: <br>
$$\hat U_k(t) = e^{-\nu |k|^2 t} \hat U_k(t) + \sigma \int_0^t e^{-\nu |k|^2(t-s)} d \hat B_k(s)$$ <br>
For each $$k \in \mathbb{Z}$$ this is a 1D Ornstein-Uhlenbeck process. We perform the time update step on each of these nodes seperately and then apply an inverse fourier transform. </p>


#Time discretisation - Semi-implicit Euler-Maruyama 

<p> Let $$h >0$$ be the time step and denote $$\hat X^{n}(k_m)$$ as the fourier node at time t for $$k_m$$, then our update from $$t_{n}$$ to $$t_{n+1}$$ is <br>
$$\hat X^{n_1}(k_m) = \frac{\hat X^{n}(k_m) + \sigma \hat \Delta B^n(k_m)}{1 - h\nu \lambda_m}$$ </p>

<p> We see the strength of using fourier as in physical space: <br>
$$(I - h\nu\partial_{xx}) X^{n+1} = X^n + \sigma \Delta B^n$$ is the update and we have to solve a large linear operator. <br>
On tours, $$\partial_{xx}$$ diagonalises and makes computation much nicer. </p>

##Noise discretisation
<p> In physical sapce, the noise i.i.d across space and time so $$\Delta B^n(x_j) \sim \mathcal{N}(0,h)$$ <br>
In fourier space, it is the same and we have $$\Delta \hat B^n(k_m) \sim \mathcal{N}(0,h)$$ </p>


