# Motivation

From classical Fourier analysis, we know that the Hilbert space $L^2([-\pi, \pi], dx)$ admits the orthonormal basis
$$
\left\{ \frac{1}{\sqrt{2\pi}} e^{inx} \;\middle|\; n \in \mathbb{Z} \right\}.
$$

Thus, every $f \in L^2([-\pi, \pi], dx)$ can be written as
$$
f(x) = \sum_{n \in \mathbb{Z}} \frac{c_n}{\sqrt{2\pi}} e^{inx}.
$$

If we remove compactness and instead consider $L^2(\mathbb{R}, dx)$, the situation becomes more subtle. Square integrability is now a much more restrictive condition, and the spectral decomposition is no longer countable. In this setting, Fourier series are replaced by the Fourier transform.

An alternative approach is to replace the Lebesgue measure $dx$ with the Gaussian measure
$$
\gamma(dx) = \frac{1}{\sqrt{2\pi}} e^{-x^2 / 2} \, dx,
$$
which allows us to recover a countable orthonormal basis, as we now outline.

---

# Construction of an orthonormal basis for $L^2(\mathbb{R}, \gamma)$

Due to the rapid decay of the Gaussian density $\frac{1}{\sqrt{2\pi}} e^{-x^2 / 2}$, all polynomials are square integrable with respect to $\gamma$. Moreover, the family
$$
\{1, x, x^2, \dots\}
$$
is linearly independent.

Applying the Gram–Schmidt orthonormalization process to this family yields an orthonormal system
$$
\{H_0(x), H_1(x), H_2(x), \dots\},
$$
known as the Hermite polynomials.

Since polynomials are dense in the continuous functions (by the Stone–Weierstrass theorem), it follows that this system is complete.

**Fact:** The family $\{H_0(x), H_1(x), H_2(x), \dots\}$ is an orthonormal basis of $L^2(\mathbb{R}, \gamma)$.

---

## Explicit formulas

Rodrigues' formula gives an explicit expression:
$$
H_n(x) = \frac{(-1)^n}{\sqrt{n!}} e^{x^2/2} \frac{d^n}{dx^n}\left(e^{-x^2/2}\right).
$$

For computations, the recurrence relation is often more useful:
$$
\sqrt{n+1}\,H_{n+1}(x) = xH_n(x) - \sqrt{n}\,H_{n-1}(x).
$$
The first few Hermite polynomials are given by 
$$
H_0(x) = 1, \quad
H_1(x) = x, \quad
H_2(x) = \frac{x^2 - 1}{\sqrt{2}}, \quad \dots
$$

## Algorithm: Generating Hermite polynomials on a 1D grid

**Input:** points $\{x_i\}_{i=1}^M$, truncation $N$  
**Output:** values $\{H_k(x_i)\}_{k=0,\dots,N;\, i=1,\dots,M}$

1. Initialize:
   - $H_0(x_i) = 1$ for all $i$
   - $H_1(x_i) = x_i$ for all $i$

2. For $k = 1, \dots, N-1$:
   - Use the recurrence relation
     $$
     \sqrt{k+1}\, H_{k+1}(x_i) = x_i H_k(x_i) - \sqrt{k}\, H_{k-1}(x_i)
     $$
     for all $i = 1, \dots, M$

3. Return $\{H_k(x_i)\}$