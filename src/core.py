import numpy as np


def ft_pdf(Sd, list_kx):
    """
    This function computes the FT of the PDF of a gaussian variable of std Sd

    Args:
        Sd (float): Standard deviation of the gaussian variable (normalized by the period)
        list_kx (list): list of directions in which to compute the intensity

    returns:
        (float) the FT of the PDF of a gaussian variable of std Sd
    """
    return np.exp(-((list_kx * Sd * 2 * np.pi) ** 2) * (1 / 2))


def S_delta(Lc, Sd, n_corr, N):
    """
    This function computes the standard deviation of the n_corr-th correlation term between positions
    It depends on the correlation length

    Args:
        Lc (float): Correlation length of the perturbation (normalized by the period)
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        n_corr (int): which correlation halo term's standard deviation to compute
        N (int): Number of positions

    returns:
        (float) the standard deviation of the n_corr-th correlation term
    """
    sigma = 2 * N  / (np.pi * Lc)
    nmod = int(round(sigma))

    A = np.array(
        [np.exp(-4 * (2 * i / sigma) ** 2) for i in range(1, nmod + 1)]
    )
    sin_phase = np.array(
        [np.sin(np.pi * n_corr * i / N) ** 2 for i in range(1, nmod + 1)]
    )
    denom = np.sum(A)
    num = np.sum(A * sin_phase)
    return np.sqrt(num / denom) * 2 * Sd


def diffuse_background(Sd, list_kx, N):
    """
    This function computes the diffuse background contribution to the scattered intensity
    It does not depend on the correlation length

    Args:
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        list_kx (list): list of directions in which to compute the intensity
        N (int): Number of positions

    returns:
        N (1 - rho_eps(k)**2) (list): diffuse background
    """
    return N * (1 - ft_pdf_eps(Sd, list_kx) ** 2)


def grating(Sd, list_kx, pos_orders, N):
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
    res = (np.sin(N * list_kx * np.pi) / np.sin(list_kx * np.pi)) ** 2
    res[pos_orders] = (
        N ** 2
    )  # making sure the diffraction order is computed correctly (division by zero)
    return ft_pdf_eps(Sd, list_kx) ** 2 * res


def correlation_halo(Lc, Sd, n_corr, list_kx, N):
    """
    This function computes the n_corr-th correlation halo contribution to the scattered intensity
    It depends on the correlation length

    Args:
        Lc (float): Correlation length of the perturbation (normalized by the period)
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        n_corr (int): which correlation halo term to compute
        list_kx (list): list of directions in which to compute the intensity
        N (int): Number of positions

    returns:
        (list): n_corr-th correlation halo contribution to the scattered intensity
    """
    std_Delta = S_delta(Lc, Sd, n_corr, N)
    
    return (
        2
        * (N - n_corr)
        * np.cos(n_corr * list_kx * 2 * np.pi)
        * (ft_pdf(std_Delta, list_kx) - ft_pdf(Sd, list_kx) ** 2)
    )

