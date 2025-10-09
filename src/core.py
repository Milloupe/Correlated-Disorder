import numpy as np
from scipy.special import j1


def ft_pdf(Sd, list_k):
    """
    This function computes the FT of the PDF of a gaussian variable of std Sd

    Args:
        Sd (float): Standard deviation of the gaussian variable (normalized by the period)
        list_k (list): list of directions in which to compute the intensity

    returns:
        (float) the FT of the PDF of a gaussian variable of std Sd
    """
    return np.exp(-((list_k * Sd * 2 * np.pi) ** 2) * (1 / 2))


def S_delta_direct(Lc, Sd, n_corr, N):
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


def S_delta_corrective(Lc, Sd, n_corr, N):
    nmod = max(20 * Lc, 5)
    nmod = int(round(nmod))

    A1 = np.array(
        [np.exp(-2 / Lc ** 2 * k ** 2) for k in range(-nmod, nmod)]
    )
    A2 = np.array(
        [
            np.exp(-2 / Lc ** 2 * (n_corr ** 2 - 2 * n_corr * k + k ** 2))
            for k in range(-nmod, nmod)
        ]
    )
    A3 = np.array(
        [
            np.exp(-1 / Lc ** 2 * (n_corr ** 2 - 2 * n_corr * k + 2 * k ** 2))
            for k in range(-nmod, nmod)
        ]
    )
    denom = np.sum(A1)
    num = np.sum(A1) + np.sum(A2) - 2 * np.sum(A3)
    return np.sqrt(num / denom) * Sd


def diffuse_background(Sd, list_k, N):
    """
    This function computes the diffuse background contribution to the scattered intensity
    It does not depend on the correlation length

    Args:
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        list_k (list): list of directions in which to compute the intensity
        N (int): Number of positions

    returns:
        N (1 - rho_eps(k)**2) (list): diffuse background
    """
    return N * (1 - ft_pdf(Sd, list_k) ** 2)


def grating(Sd, list_k, pos_orders, N):
    """
    This function computes the grating orders contribution to the scattered intensity
    It does not depend on the correlation length

    Args:
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        list_k (list): list of directions in which to compute the intensity
        pos_orders (list): list of indices (in list_k) corresponding to diffraction peak positions
        N (int): Number of positions

    returns:
        rho_eps(k)**2 * (sin(N k) / sin(k)) ** 2 (list): grating orders
    """
    res = (np.sin(N * list_k * np.pi) / np.sin(list_k * np.pi)) ** 2
    res[pos_orders] = (
        N ** 2
    )  # making sure the diffraction order is computed correctly (division by zero)
    return ft_pdf(Sd, list_k) ** 2 * res


def correlation_halo(S_delta, Lc, Sd, n_corr, list_k, N):
    """
    This function computes the n_corr-th correlation halo contribution to the scattered intensity
    It depends on the correlation length

    Args:
        S_delta (function): how to compute the correlation standard deviation
        Lc (float): Correlation length of the perturbation (normalized by the period)
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        n_corr (int): which correlation halo term to compute
        list_k (list): list of directions in which to compute the intensity
        N (int): Number of positions

    returns:
        (list): n_corr-th correlation halo contribution to the scattered intensity
    """
    std_Delta = S_delta(Lc, Sd, n_corr, N)
    
    return (
        2
        * (N - n_corr)
        * np.cos(n_corr * list_k * 2 * np.pi)
        * (ft_pdf(std_Delta, list_k) - ft_pdf(Sd, list_k) ** 2)
    )




def diffraction_figure(pos, list_k, resolution, size=0.):
    """
    This function computes the diffraction figure along one direction of an array of points,
    by computing its Fourier Transform

    Args:
        S_delta (function): how to compute the correlation standard deviation
        Lc (float): Correlation length of the perturbation (normalized by the period)
        Sd (float): Standard deviation of the perturbation (normalized by the period)
        n_corr (int): which correlation halo term to compute
        list_k (list): list of directions in which to compute the intensity
        N (int): Number of positions

    returns:
        (list): scattered intensity along one direction
    """

    B = np.zeros((1, resolution))

    for k in range(0, len(pos)):
        B = B + np.exp(2j * np.pi * pos[k] * list_k) # Hand computation of the FT of the points

    if size :
        # A size for the emitters was given -> use Airy formula
        r = np.sqrt(list_k**2)
        A = 2*np.pi * size**2 / 4 * j1(np.pi * size * r)/(np.pi * size * r)
        F = ((np.abs(B*A) / len(pos))) ** 2
        return F
    
    F = ((np.abs(B) / len(pos))) ** 2

    return F


def analytical_average_diff_fig(S_delta, Lc, Sd, tot_n_corr, list_k, resolution, pos_orders, N):
    """
    This function computes the scattered intensity statistical average (along one direction)

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
        (list): analytical value fo the statistical average of the scattered intensity along one direction
    """


    avg_diffuse_grating = diffuse_background(Sd, list_k, N)
    avg_grating = grating(Sd, list_k, pos_orders, N)
    avg_corr = np.zeros((tot_n_corr, resolution))


    for i in range(tot_n_corr):
        avg_corr[i] += correlation_halo(S_delta, Lc, Sd, i+1, list_k, N)

        # Normalization
        avg_diffuse_grating = avg_diffuse_grating / N ** 2
        avg_grating = avg_grating / N ** 2
        avg_corr = avg_corr / N ** 2

        tot = avg_diffuse_grating + avg_grating + np.sum(avg_corr, axis=0)

    return avg_diffuse_grating, avg_grating, avg_corr, tot