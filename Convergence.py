import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


# STRONG AND WEAK ERROR METRICS


def l2_norm(u, v, dx):
    return np.sqrt(np.sum((u - v)**2) * dx)

def h1_seminorm(u, v, k):
    uh = fft(u); vh = fft(v)
    N =u.size
    return np.sqrt((np.sum((k**2) * np.abs(uh - vh)**2).real)/N)

def l2_space_time(U, V, dx, dt):
    # U, V shape (num_steps+1, N)
    diff = U - V
    return np.sqrt(np.sum(np.sum(diff**2, axis=1) * dx * dt))



# RUN DT-REFINEMENT EXPERIMENT


def run_dt_refinement(
    simulate_fn,
    N=1024,
    T=1.0,
    beta=0.1,
    nu=0.5,
    n=50,
    base_dt=0.01,
    levels=5
):
    L = 1.0
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    k = 2*np.pi*np.fft.fftfreq(N, d=dx)

    # ---- Fine reference ----
    dt_ref = base_dt / (2**levels)
    Xref_full, _ = simulate_fn(h=dt_ref, T=T, N=N, n=n, beta=beta, nu=nu)
    Xref_T = Xref_full[:, -1, :]

    # weak reference mean
    Phi_ref = np.mean(np.sum(Xref_T**2, axis=1) * dx)

    results = []

    for ell in range(levels+1):
        dt = base_dt / (2**ell)
        X_full, _ = simulate_fn(h=dt, T=T, N=N, n=n, beta=beta, nu=nu)
        XT = X_full[:, -1, :]

        # --- Strong L2 ---
        strong_L2 = np.sqrt(np.mean([
            l2_norm(XT[i], Xref_T[i], dx)**2
            for i in range(n)
        ]))

        # --- Strong H1 seminorm ---
        #strong_H1 = np.mean([
           #h1_seminorm(XT[i], Xref_T[i], k)
            #for i in range(n)
        #])

        # --- Space-time strong L2 ---
        #strong_space_time = np.mean([
            #l2_space_time(X_full[i], Xref_full[i], dx, dt)
            #for i in range(n)
        #])

        # --- Weak L2 functional ---
        Phi_vals = np.sum(XT**2, axis=1) * dx
        weak_err = abs(np.mean(Phi_vals) - Phi_ref)

        results.append({
            "dt": dt,
            "strong_L2": strong_L2,
            #"strong_H1": strong_H1,
            #"strong_space_time": strong_space_time,
            "weak_L2": weak_err,
        })

        C_strong = strong_L2/np.sqrt(dt)
        C_weak = weak_err/dt

        print(f"dt={dt:.5f}  strong L2={strong_L2:.4e}  weak L2={weak_err:.4e} C strong={C_strong} C weak = {C_weak}")

    return results



# PLOTTING


def plot_convergence(results):
    dts = np.array([r["dt"] for r in results])

    plt.figure()
    plt.loglog(dts, [r["strong_L2"] for r in results], "o-", label="Strong L2")
    #plt.loglog(dts, [r["strong_H1"] for r in results], "s-", label="Strong H1")
    #plt.loglog(dts, [r["strong_space_time"] for r in results], "^-", label="Strong space-time L2")
    plt.gca().invert_xaxis()
    plt.grid(True, which='both')
    plt.xlabel("dt")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Strong Convergence Rates")
    plt.show()

    plt.figure()
    plt.loglog(dts, [r["weak_L2"] for r in results], "o-", label="Weak L2 functional")
    plt.gca().invert_xaxis()
    plt.grid(True, which='both')
    plt.xlabel("dt")
    plt.ylabel("Weak error")
    plt.legend()
    plt.title("Weak Convergence Rate")
    plt.show()