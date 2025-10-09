import numpy as np
from scipy.special import j1
from core import ft_pdf


def S_delta_direct_2D(Lcx, Lcy, Sdx, Sdy, n_corrx, n_corry, Nx, Ny=0, mode="x"):
    """
    This function computes the standard deviation of the (n_corrx, nçcorry)-th correlation term between positions
    It depends on the correlation length

    Args:
        Lcx, Lcy (float): Correlation length of the perturbation (normalized by the period)
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        n_corrx (int): which correlation halo term's standard deviation to compute
        n_corry (int): which correlation halo term's standard deviation to compute
        Nx (int): Number of positions
        Ny (int, optional): Number of positions

    returns:
        (float) the standard deviation of the n_corr-th correlation term
        The main difference with 1D is that because we take into account the neighbours in each directions,
        we have (0, m) or (n, 0) neighbours, meaning we need to compute the values for n and m = 0
    """
    if not(Ny):
        Ny = Nx
    sigmax = 2 * Nx  / (np.pi * Lcx)
    nmodx = int(round(sigmax))
    sigmay = 2 * Ny  / (np.pi * Lcy)
    nmody = int(round(sigmay))
    if mode == "x":
        sigma = sigmax
        N = Nx
        Sd= Sdx
    elif mode == "y":
        sigma = sigmay
        N = Ny
        Sd = Sdy
    else:
        print("Invalid mode is S_delta2D:", mode, " should be x or y")
        return None

    A = np.array(
        [
            np.exp(- (4 / sigma) ** 2 * (i ** 2 + j ** 2))
            for i in range(0, nmodx + 1)
            for j in range(0, nmody + 1)
        ]
    )
    sin_phase = np.array(
        [
            np.sin(np.pi * (n_corrx * i + n_corry * j) / N) ** 2
            for i in range(0, nmodx + 1)
            for j in range(0, nmody + 1)
        ]
    )
    denom = np.sum(A) - 1 # remove the i=j=0 term
    num = np.sum(A * sin_phase) # here the i=j=0 term is cancelled by the sin = 0
    return np.sqrt(num / denom) * 2 * Sd



def diffuse_background_2D(Sdx, Sdy, list_kx, list_ky, Nx, Ny=0):
    """
    This function computes the 2D diffuse background contribution to the scattered intensity
    It does not depend on the correlation length

    Args:
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        list_kx (list): list of directions in which to compute the intensity
        N (int): Number of positions

    returns:
        N (1 - rho_eps(k)**2) (list): diffuse background
    """
    if not(Ny):
        Ny=Nx

    return Nx * Ny * (1 - ft_pdf(Sdx, list_kx) ** 2 * ft_pdf(Sdy, list_ky) ** 2)


def grating_2D(Sdx, Sdy, list_kx, list_ky, pos_ordersx, pos_ordersy, Nx, Ny=0):
    """
    This function computes the grating orders contribution to the scattered intensity
    It does not depend on the correlation length

    Args:
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        list_kx (list): list of directions in which to compute the intensity
        pos_orders (list): list of indices (in list_kx) corresponding to diffraction peak positions
        N (int): Number of positions

    returns:
        rho_eps(k)**2 * (sin(N k) / sin(k)) ** 2 (list): grating orders
    """
    if not(Ny):
        Ny=Nx

    res = (np.sin(Nx * list_kx * np.pi) / np.sin(list_kx * np.pi)) ** 2 * (
        np.sin(Ny * list_ky * np.pi) / np.sin(list_ky * np.pi)
    ) ** 2
    res[pos_ordersy, pos_ordersx] = (
        Nx ** 2 * Ny ** 2
    )  # making sure the diffraction order is computed correctly (division by zero)
    return ft_pdf(Sdx, list_kx) ** 2 * ft_pdf(Sdy, list_ky) ** 2 * res


def correlation_halo_2D(S_delta2D, Lcx, Lcy, Sdx, Sdy, n_corrx, n_corry, list_kx, list_ky, Nx, Ny=0):
    """
    This function computes the n_corr-th correlation halo contribution to the scattered intensity
    It depends on the correlation length

    Args:
        S_delta2D (function): how to compute the correlation standard deviation
        Lc (float): Correlation length of the perturbation (normalized by the period)
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        n_corr (int): which correlation halo term to compute
        list_kx (list): list of directions in which to compute the intensity
        N (int): Number of positions

    returns:
        (list): n_corr-th correlation halo contribution to the scattered intensity
    """
    if not(Ny):
        Ny = Nx
    

    std_Deltax = S_delta2D(Lcx, Lcy, Sdx, Sdy, n_corrx, n_corry, Nx, Ny, mode="x")
    std_Deltay = S_delta2D(Lcx, Lcy, Sdx, Sdy, n_corrx, n_corry, Nx, Ny, mode="y")

    res = (
        2
        * (Nx - n_corrx)
        * (Ny - n_corry)
        * np.cos(n_corrx * list_kx * 2 * np.pi)
        * np.cos(n_corry * list_ky * 2 * np.pi)
    )
    return res * (
        ft_pdf(std_Deltax, list_kx) * ft_pdf(std_Deltay, list_ky)
        - ft_pdf(Sdx, list_kx) ** 2 * ft_pdf(Sdy, list_ky) ** 2
    )



def diffraction_figure_2D(pos, list_kx, list_ky, resolutionx, resolutiony, size=0.1):
    """
    This function computes the diffraction figure along two directions of an array of points,
    by computing its Fourier Transform

    Args:
        S_delta (function): how to compute the correlation standard deviation
        Lc (float): Correlation length of the perturbation (normalized by the period)
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        n_corr (int): which correlation halo term to compute
        list_kx (list): list of directions in which to compute the intensity
        N (int): Number of positions

    returns:
        (1D list): scattered intensity along two directions
    """

    [X, Y] = np.meshgrid(list_kx, list_ky)

    B = np.zeros((resolutiony, resolutionx), dtype=complex)

    for k in range(0, len(pos)):
        x, y = pos[k]
        B = B + np.exp(2j * np.pi * x * X) * np.exp(2j * np.pi * y * Y)  # Hand computation of the FT of the points

    if size :
        # A size for the emitters was given -> use Airy formula
        r = np.sqrt(list_kx**2)
        A = 2*np.pi * size**2 / 4 * j1(np.pi * size * r)/(np.pi * size * r)
        F = ((np.abs(B*A) / len(pos))) ** 2
        return F
    
    F = ((np.abs(B) / len(pos))) ** 2

    return F


def avg_fig(S_delta, Lcx, Lcy, Sdx, Sdy, tot_n_corr, list_kx, list_ky, resolutionx, resolutiony, pos_ordersx, pos_ordersy, Nx, Ny=0):
    """
    This function computes the scattered intensity statistical average (along two directions)

    Args:
        S_delta (function): how to compute the correlation standard deviation
        Lc (float): Correlation length of the perturbation (normalized by the period)
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        tot_n_corr (int): total number of correlation halo term to compute
        list_k (list): list of directions in which to compute the intensity
        resolution (int): number of points in list_k
        pos_orders (int): index of positions within list_k of diffraction orders
        N (int): Number of positions

    returns:
        (list): analytical value fo the statistical average of the scattered intensity along two directions
    """
    if not(Ny):
        Ny = Nx

    avg_diffuse_grating = diffuse_background_2D(Sdx, Sdy, list_kx, list_ky, Nx, Ny)
    avg_grating = grating_2D(Sdx, Sdy, list_kx, list_ky, pos_ordersx, pos_ordersy, Nx, Ny)
    avg_corr = np.zeros((tot_n_corr, tot_n_corr, resolutionx, resolutiony))


    for i in range(tot_n_corr):
        for j in range(tot_n_corr):
            if i == j == 0:
                avg_corr[i, j] += 0
            else:
                avg_corr[i, j] += correlation_halo_2D(S_delta, Lcx, Lcy, Sdx, Sdy, i+1, j+1, list_kx, list_ky, Nx, Ny)


    # Normalization
    avg_diffuse_grating = avg_diffuse_grating / (Nx ** 2 * Ny ** 2)
    avg_grating = avg_grating / (Nx ** 2 * Ny ** 2)
    avg_corr = avg_corr / (Nx ** 2 * Ny ** 2)

    tot = avg_diffuse_grating + avg_grating + np.sum(avg_corr, axis=(0,1))
    return avg_diffuse_grating, avg_grating, avg_corr, tot
