import numpy as np
import scipy.signal as sig
from random import gauss

# TODO : check dimensions for 2D
# TODO : optimize, a bit?


def direct(L, Sd, N):
    """
    This function generates a random gaussian perturbation to N periodic positions

    Args:
        L (float): Correlation length of the perturbation (normalized by the period)
        Sd (float): Standrd deviation of the perturbation (normalized by the period)
        N (int): Number of positions

    returns:
        z (list): list of N perturbations
    """
    nmod = 4 * N / (2 * np.pi * L)
    nmod = int(round(nmod))

    phi = np.random.rand(nmod) * 2 * np.pi
    z = np.zeros(N)
    A = np.array(
        [np.exp(-2 * (np.pi * L * i / (N)) ** 2) for i in range(1, nmod + 1)]
    )

    exp_phi = np.exp(1j * phi)

    for j in range(1, N + 1):
        exp_phase = np.exp(2 * 1j * np.pi * np.arange(1, nmod + 1) * j / (N))
        produit = exp_phi * A * exp_phase
        z[j - 1] = np.real(np.sum(produit))

    z = z * np.sqrt(N / np.sum(z**2)) * Sd

    return z


def corrective(L, Sd, N):
    """
    This function generates a random corrective gaussian perturbation to N periodic positions

    Args:
        L (float): Correlation length of the perturbation (normalized by the period)
        Sd (float): Standrd deviation of the perturbation (normalized by the period)
        N (int): Number of positions

    returns:
        z (list): list of N perturbations
    """
    z = np.zeros(N)
    x = [gauss(0, 1) for _ in range(N)]
    dx = [0] * N

    for i in range(N):
        for j in range(N):
            fact = np.exp(-((i - j) ** 2) / L**2)
            dx[i] += x[j] * fact
        z[i] = dx[i]

    z = z * np.sqrt(N / np.sum(z**2)) * Sd
    return z


def direct_2D(Lx, Ly, Sdx, Sdy, Nx, Ny=0):
    """
    This function generates a random gaussian perturbation to Nx * Ny periodic positions

    Args:
        L (float): Correlation length of the perturbation (normalized by the period)
        Sd (float): Standrd deviation of the perturbation (normalized by the period)
        N (int): Number of positions

        Each argument is given for both directions

    returns:
        z (list): list of N perturbations
    """
    if not Ny:
        Ny = Nx
    z = np.zeros((Nx * Ny, 2))

    nmodx = 4 * Nx / (2 * np.pi * Lx)
    nmody = 4 * Ny / (2 * np.pi * Ly)
    nmodx = int(round(nmodx))
    nmody = int(round(nmody))

    Ax = np.array(
        [
            [
                np.exp(-2 * (np.pi * Lx / (Nx)) ** 2 * (i**2 + j**2))
                for i in range(0, nmodx)
            ]
            for j in range(0, nmody)
        ]
    )
    Ay = np.array(
        [
            [
                np.exp(-2 * (np.pi * Ly / (Ny)) ** 2 * (i**2 + j**2))
                for i in range(0, nmodx)
            ]
            for j in range(0, nmody)
        ]
    )

    phi = np.random.rand(nmody, nmodx) * 2 * np.pi
    exp_phi = np.exp(1j * phi)

    dx = np.zeros((Nx, Ny))
    for j in range(1, Nx + 1):
        exp_phasex = np.exp(2 * 1j * np.pi * np.arange(0, nmodx) * j / (Nx))
        for k in range(1, Ny + 1):
            exp_phasey = np.exp(2 * 1j * np.pi * np.arange(0, nmody) * k / (Nx))
            exp_phase1, exp_phase2 = np.meshgrid(exp_phasex, exp_phasey)
            exp_phase = exp_phase1 * exp_phase2
            produit = exp_phi * Ax * exp_phase
            produit[0, 0] = 0  # removing n = m = 0
            dx[j - 1, k - 1] = np.real(np.sum(produit))


    phi = np.random.rand(nmody, nmodx) * 2 * np.pi
    exp_phi = np.exp(1j * phi)

    dy = np.zeros((Nx, Ny))
    for j in range(1, Nx + 1):
        exp_phasex = np.exp(2 * 1j * np.pi * np.arange(0, nmodx) * j / (Ny))
        for k in range(1, Ny + 1):
            exp_phasey = np.exp(2 * 1j * np.pi * np.arange(0, nmody) * k / (Ny))
            exp_phase1, exp_phase2 = np.meshgrid(exp_phasex, exp_phasey)
            exp_phase = exp_phase1 * exp_phase2
            produit = exp_phi * Ay * exp_phase
            produit[0, 0] = 0  # removing n = m = 0
            dy[j - 1, k - 1] = np.real(np.sum(produit))

    dx = dx * np.sqrt(Nx * Nx / np.sum(dx**2)) * Sdx
    dy = dy * np.sqrt(Ny * Ny / np.sum(dy**2)) * Sdy
    for i in range(Nx):
        for j in range(Ny):
            z[i*Ny + j] = ([dx[i, j], dy[i, j]])

    return np.array(z)


def corrective_2D(Lx, Ly, Sdx, Sdy, Nx, Ny=0):
    """
    This function generates a random corrective gaussian perturbation to Nx * Ny periodic positions

    Args:
        L (float): Correlation length of the perturbation (normalized by the period)
        Sd (float): Standrd deviation of the perturbation (normalized by the period)
        N (int): Number of positions

    returns:
        z (list): list of N perturbations
    """

    if not Ny:
        Ny = Nx
    z = np.zeros((Nx * Ny, 2))
    x = np.array([[gauss(0, 1) for _ in range(Nx)] for y in range(Ny)])
    y = np.array([[gauss(0, 1) for _ in range(Nx)] for y in range(Ny)])
    
    dx = np.zeros((Nx, Ny))
    dy = np.zeros((Nx, Ny))

    
    nb_neighbors_x = int(np.ceil(5*Lx))
    nb_neighbors_y = int(np.ceil(5*Ly))
    weightx = np.array([[np.exp(-(n_x**2+n_y**2)/(Lx)**2)
                        for n_x in range(-nb_neighbors_x, nb_neighbors_x)]
                        for n_y in range(-nb_neighbors_y, nb_neighbors_y)])
    weighty = np.array([[np.exp(-(n_x**2+n_y**2)/(Ly)**2)
                        for n_x in range(-nb_neighbors_x, nb_neighbors_x)]
                        for n_y in range(-nb_neighbors_y, nb_neighbors_y)])

    dx = sig.fftconvolve(x, weightx, mode="same")
    dy = sig.fftconvolve(y, weighty, mode="same")
    

    dx = dx * np.sqrt(Nx * Nx / np.sum(dx**2)) * Sdx
    dy = dy * np.sqrt(Ny * Ny / np.sum(dy**2)) * Sdy

    for i in range(Nx):
        for j in range(Ny):
            z[i*Ny + j] = ([dx[i, j], dy[i, j]])
        
    return z