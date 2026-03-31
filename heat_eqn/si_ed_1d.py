import numpy as np

def simulate_spde(h=0.001, T=1.0, N=1024, n=1, beta=0.1, nu=0.5):
    """
    Returns:
        X_path : shape (n, num_steps+1, N)
        x      : spatial grid
    """
    L = 1.0
    dx = L / N
    num_steps = int(T / h)
    sigma = np.sqrt(2 / beta)

    x = np.linspace(0, L, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    laplacian_symbol = -(k**2)
    implicit_factor = 1.0 / (1.0 - h * nu * laplacian_symbol)

    # Initial condition
    X = np.random.normal(0, 0.2, size=(n, N))
    X_path = np.zeros((n, num_steps+1, N))
    X_path[:, 0, :] = X

    for j in range(num_steps):
        dW = np.random.normal(0, np.sqrt(h), size=(n, N))
        rhs = X + sigma * dW

        rhs_hat = np.fft.fft(rhs, axis=1)
        X_hat_new = implicit_factor * rhs_hat
        X = np.fft.ifft(X_hat_new, axis=1).real

        X_path[:, j+1, :] = X

    return X_path, x