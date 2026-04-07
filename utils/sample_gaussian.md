# Sampling Gaussian Measures via Karhunen–Loève Expansion

Let $\mu$ be a Gaussian measure with mean $m$ and covariance operator $K$ in some real separable infinite-dimensional Hilbert space $\mathcal{H}$. Suppose that $K$ is a self-adjoint, positive trace-class operator with positive eigenvalues $\lambda_k$ and a corresponding set $e_k$ of orthonormal eigenvectors, $k\ge 1$.

Then, by the Karhunen–Loève theorem, any $\xi \sim \mathcal{N}(m, K)$ has the representation

$$
\xi = m + \sum_{k \ge 1} \sqrt{\lambda_k} \gamma_k e_k, \quad \gamma_k \sim \mathcal{N}(0, 1), \quad \text{a.s.}
$$

If we know the eigendecomposition of $K$, one can sample $\xi \sim \mathcal{N}(m, K)$ by sampling $\gamma_k \sim \mathcal{N}(0,1)$ i.i.d. and truncating the series above up to some desired resolution.

---

## Integral Operator Setting

Consider the setup where $\mathcal{H} = L^2(\Omega, \nu)$. Suppose the covariance operator  
$K : L^2(\Omega, \nu) \to L^2(\Omega, \nu)$ is defined via an integral kernel $k : \Omega \times \Omega \to \mathbb{R}$:

$$
(Kf)(x) = \int_\Omega k(x,y) f(y)\, d\nu(y), \quad f \in L^2(\Omega, \nu).
$$

We assume that $k$ is continuous, symmetric, and positive definite, so that $K$ is self-adjoint, positive, and trace-class (see Mercer's theorem).

A common choice of kernel is the squared-exponential kernel:

$$
k(x,y) = \exp\left(-\frac{|x-y|^2}{2\ell^2}\right), \quad x,y \in \Omega,
$$

where $\ell > 0$ is a length scale parameter.

---

## Algorithm: Sampling from $\mathcal{N}(m, K)$

**Input:** kernel $k$, mean $m$, points $\{x_i\}$, weights $\{w_i\}$, truncation $N$, samples $S$  
**Output:** samples $\{\xi^{(s)}\}$

1. Form kernel matrix $K_{ij} = k(x_i, x_j)$
2. Form $B_{ij} = \sqrt{w_i} K_{ij} \sqrt{w_j}$
3. Compute eigendecomposition $B = V \Lambda V^\top$
4. Keep top-$N$ eigenpairs
5. Compute $e_k = w^{-1/2} \odot v_k$
6. For $s = 1, \dots, S$:
   - Sample $\gamma_k \sim \mathcal{N}(0,1)$
   - Compute $\xi^{(s)} = m + \sum_{k=1}^N \sqrt{\lambda_k} \gamma_k e_k$
7. Return samples
