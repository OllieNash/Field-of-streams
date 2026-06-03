import numpy as np
import matplotlib.pyplot as plt

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


def plot_sde_path(path): 
    for i in range(path.shape[0]):
        plt.plot(path[i, :], alpha=0.7)    
    plt.title("Trajectories of Particles over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Position")
    plt.show()   
