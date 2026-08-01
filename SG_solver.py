"""Classical multiquadric quasi-interpolation solver for the Sine-Gordon equation.

This module implements the classical Wu-Schaback multiquadric (MQ)
quasi-interpolation operator L_W2 and its second derivative, along with the
underlying MQ basis functions and RBF interpolation matrices.
"""

import numpy as np

__all__ = [
    "paper_rms",
    "varphi",
    "phi",
    "psi",
    "LD_operator",
    "second_divided_difference",
    "rbf_matrix",
    "rbf_correction",
    "error_function",
    "L_W2",
    "d2_phi",
    "d2_psi",
    "d2_L_W2",
]

def paper_rms(exact, num):
    """Paper's RMS definition: RMS = (1 / (n + 1)) * sqrt(sum(|exact - num|^2))"""
    return (1.0 / (len(exact))) * np.sqrt(np.sum((exact - num)**2))

def varphi(r, s):
    """RBF kernel function: varphi(r) = s^2 / (s^2 + r^2)^(3/2)."""
    return s**2 / (s**2 + r**2) ** 1.5


def phi(a, b, c):
    """Multiquadric radial basis function: phi_i(a) = sqrt(c^2 + (a - b)^2)."""
    return np.sqrt(c**2 + (a - b) ** 2)


def psi(a, j, x, c):
    """Evaluates the quasi-interpolation basis function psi_j(a) on node set x.

    Parameters
    ----------
    a : float or array_like
        Evaluation point.
    j : int
        Basis index in {0, 1, ..., n}.
    x : array_like
        Spatial grid nodes.
    c : float
        MQ shape parameter.
    """
    n = len(x) - 1

    if j == 0:
        return 0.5 + (phi(a, x[1], c) - (a - x[0])) / (2 * (x[1] - x[0]))

    if j == 1:
        return (
            (phi(a, x[2], c) - phi(a, x[1], c)) / (2 * (x[2] - x[1]))
            - (phi(a, x[1], c) - (a - x[0])) / (2 * (x[1] - x[0]))
        )

    if j == n - 1:
        return (
            ((x[n] - a) - phi(a, x[n - 1], c)) / (2 * (x[n] - x[n - 1]))
            - (phi(a, x[n - 1], c) - phi(a, x[n - 2], c))
            / (2 * (x[n - 1] - x[n - 2]))
        )

    if j == n:
        return 0.5 + (phi(a, x[n - 1], c) - (x[n] - a)) / (
            2 * (x[n] - x[n - 1])
        )

    return (
        (phi(a, x[j + 1], c) - phi(a, x[j], c)) / (2 * (x[j + 1] - x[j]))
        - (phi(a, x[j], c) - phi(a, x[j - 1], c)) / (2 * (x[j] - x[j - 1]))
    )


def LD_operator(a, i, x, f, c):
    """Wu-Schaback interpolation operator: (L_D f)(a) = sum_j f_j psi_j(a).

    The `i` argument is retained for signature compatibility with notebook scripts.
    """
    d = np.array([psi(a, j, x, c) for j in range(len(x))])
    return np.asarray(f) @ d


def second_divided_difference(x, f, kj):
    """Computes the 3-point non-uniform second divided difference at interior index kj.

    Parameters
    ----------
    x : array_like
        Spatial node positions.
    f : array_like
        Function values at x.
    kj : int
        Interior node index in {1, ..., len(x)-2}.
    """
    if kj <= 0 or kj >= len(x) - 1:
        raise ValueError("kj must be a strictly interior grid index.")

    d1 = np.diff(f) / np.diff(x)
    d2 = 2.0 * np.diff(d1) / (x[2:] - x[:-2])
    return d2[kj - 1]


def rbf_matrix(xk, s):
    """Returns the RBF interpolation matrix A_ij = varphi(|x_{k_i} - x_{k_j}|)."""
    r = np.abs(xk[:, None] - xk[None, :])
    return varphi(r, s)


def rbf_correction(a, xk, alpha, s):
    """Evaluates the RBF correction R(a) = sum_j alpha_j sqrt(s^2 + (a - x_{k_j})^2)."""
    return np.sum(alpha * np.sqrt(s**2 + (a - xk) ** 2))


def error_function(i, x, f, xk, alpha, s):
    """Returns the residual error e(x_i) = f(x_i) - R(x_i)."""
    return f[i] - rbf_correction(x[i], xk, alpha, s)


def L_W2(i, x, f, xk, alpha, s, c):
    """Evaluates the Wu-Schaback quasi-interpolant (L_W2 f)(x_i).

    L_W2 f(x_i) = sum_j alpha_j sqrt(s^2 + (x_i - x_{k_j})^2) + (L_D e)(x_i).
    """
    rbf_part = rbf_correction(x[i], xk, alpha, s)
    e_vals = np.array([error_function(p, x, f, xk, alpha, s) for p in range(len(x))])
    ld_part = LD_operator(x[i], i, x, e_vals, c)
    return rbf_part + ld_part


def d2_phi(a, b, c):
    """Second derivative of phi_i(a) with center b: c^2 / (c^2 + (a - b)^2)^(3/2)."""
    return c**2 / (c**2 + (a - b) ** 2) ** 1.5


def d2_psi(a, j, x, c):
    """Second derivative of the basis function psi_j''(a)."""
    n = len(x) - 1

    if j == 0:
        return d2_phi(a, x[1], c) / (2 * (x[1] - x[0]))

    if j == 1:
        return (
            (d2_phi(a, x[2], c) - d2_phi(a, x[1], c)) / (2 * (x[2] - x[1]))
            - d2_phi(a, x[1], c) / (2 * (x[1] - x[0]))
        )

    if j == n - 1:
        return (
            -d2_phi(a, x[n - 1], c) / (2 * (x[n] - x[n - 1]))
            - (d2_phi(a, x[n - 1], c) - d2_phi(a, x[n - 2], c))
            / (2 * (x[n - 1] - x[n - 2]))
        )

    if j == n:
        return d2_phi(a, x[n - 1], c) / (2 * (x[n] - x[n - 1]))

    return (
        (d2_phi(a, x[j + 1], c) - d2_phi(a, x[j], c))
        / (2 * (x[j + 1] - x[j]))
        - (d2_phi(a, x[j], c) - d2_phi(a, x[j - 1], c))
        / (2 * (x[j] - x[j - 1]))
    )


def d2_L_W2(i, x, f, xk, alpha, s, c):
    """Evaluates the second derivative of the Wu-Schaback operator (L_W2 f)''(x_i).

    (L_W2 f)''(x_i) = sum_j alpha_j varphi(x_i - x_{k_j}) + sum_p e(x_p) psi_p''(x_i).
    """
    xx = x[i]

    # RBF correction second derivative
    rbf_part = np.sum(alpha * varphi(xx - xk, s))

    # Error function evaluations
    e_vals = np.array([error_function(p, x, f, xk, alpha, s) for p in range(len(x))])

    # Quasi-interpolation correction second derivative
    ld_part = 0.0
    for p in range(len(x)):
        ld_part += e_vals[p] * d2_psi(xx, p, x, c)

    return rbf_part + ld_part
