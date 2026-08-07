"""Fractal multiquadric quasi-interpolation solver for the Sine-Gordon equation.

This module implements the fractal multiquadric (MQ) quasi-interpolation
operator and its derivatives. It constructs an alpha-fractalized MQ basis
using a 5th-degree Hermite polynomial base function H5, matching the multiquadric
function and its first two derivatives at the boundary nodes.
"""

import numpy as np
from SG_solver import varphi, error_function

__all__ = [
    "phi",
    "dphi",
    "ddphi",
    "H5",
    "H5_dd",
    "build_fractal_second_derivative",
    "pointwise_fractal",
    "d2_fractal_phi_pointwise",
    "d2_fractal_psi",
    "d2_fractal_L_W2",
]


def phi(z, c=0.027):
    """Multiquadric (MQ) radial basis function: phi(z) = sqrt(c^2 + z^2)."""
    return np.sqrt(c**2 + z**2)


def dphi(z, c=0.027):
    """First derivative of the MQ RBF: phi'(z) = z / sqrt(c^2 + z^2)."""
    return z / np.sqrt(c**2 + z**2)


def ddphi(z, c=0.027):
    """Second derivative of the MQ RBF: phi''(z) = c^2 / (c^2 + z^2)^(3/2)."""
    return c**2 / (c**2 + z**2)**1.5


def H5(z, x1=-2, xN=2, c=0.027):
    """5th-degree Hermite polynomial base function H5(z).
    
    Interpolates phi, phi', and phi'' at the boundary points x1 and xN.
    """
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    phi1   = phi(x1, c)
    phiN   = phi(xN, c)
    phi1d  = dphi(x1, c)
    phiNd  = dphi(xN, c)
    phi1dd = ddphi(x1, c)
    phiNdd = ddphi(xN, c)

    h1 = (phiN - phi1 - phi1d * dx - 0.5 * phi1dd * dx**2) / dx**3
    h2 = (3 * (phi1 - phiN) + 2 * (phi1d + 0.5 * phiNd) * dx + 0.5 * phi1dd * dx**2) / dx**4
    h3 = (6 * (phiN - phi1) - 3 * (phi1d + phiNd) * dx + 0.5 * (phiNdd - phi1dd) * dx**2) / dx**5

    dz = z - x1
    return (
        phi1
        + phi1d * dz
        + 0.5 * phi1dd * dz**2
        + h1 * dz**3
        + h2 * dz**3 * (z - xN)
        + h3 * dz**3 * (z - xN)**2
    )


def H5_dd(z, x1=-2, xN=2, c=0.027):
    """Second derivative of the 5th-degree Hermite polynomial base function H5''(z)."""
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    phi1   = phi(x1, c)
    phiN   = phi(xN, c)
    phi1d  = dphi(x1, c)
    phiNd  = dphi(xN, c)
    phi1dd = ddphi(x1, c)
    phiNdd = ddphi(xN, c)

    h1 = (phiN - phi1 - phi1d * dx - 0.5 * phi1dd * dx**2) / dx**3
    h2 = (3 * (phi1 - phiN) + 2 * (phi1d + 0.5 * phiNd) * dx + 0.5 * phi1dd * dx**2) / dx**4
    h3 = (6 * (phiN - phi1) - 3 * (phi1d + phiNd) * dx + 0.5 * (phiNdd - phi1dd) * dx**2) / dx**5

    dz = z - x1
    w = z - xN

    return (
        phi1dd
        + 6.0 * h1 * dz
        + h2 * (6.0 * dz * w + 6.0 * dz**2)
        + h3 * (6.0 * dz * w**2 + 12.0 * dz**2 * w + 2.0 * dz**3)
    )


def build_fractal_second_derivative(x, c, f_alpha, n_iter):
    """Constructs the second derivative of the alpha-fractalized MQ basis function.

    Parameters
    ----------
    x : array_like
        Interpolation nodes partitioning [a, b].
    c : float
        MQ shape parameter.
    f_alpha : float or array_like
        Fractal scaling parameter(s) alpha_k for each subinterval.
    n_iter : int
        Number of fractal iterations.

    Returns
    -------
    dict
        Dictionary with keys 'partition' and 'values' containing the grid
        coordinates and the evaluated fractal second derivative values.
    """
    x = np.asarray(x, dtype=float)
    a = x[0]
    b = x[-1]
    N = len(x) - 1

    if np.isscalar(f_alpha):
        f_alpha = np.full(N, f_alpha, dtype=float)
    else:
        f_alpha = np.asarray(f_alpha, dtype=float)

    # Initial partition on [a, b]
    partition = np.array([a, b], dtype=float)
    ydd = ddphi(partition, c)

    for _ in range(n_iter):
        new_parts = []
        new_vals = []
        Hdd = H5_dd(partition, a, b, c)
        diff = ydd - Hdd

        for k in range(N):
            xl = x[k]
            xr = x[k + 1]
            scale = ((b - a) / (xr - xl)) ** 2
            pk = xl + (partition - a) * ((xr - xl) / (b - a))
            yk = ddphi(pk, c) + f_alpha[k] * scale * diff

            if k == 0:
                new_parts.append(pk)
                new_vals.append(yk)
            else:
                # Avoid duplicate node at adjacent subinterval boundaries
                new_parts.append(pk[1:])
                new_vals.append(yk[1:])

        partition = np.concatenate(new_parts)
        ydd = np.concatenate(new_vals)

    return {
        "partition": partition,
        "values": ydd,
    }


def pointwise_fractal(z, fractal_dd):
    """Evaluates the fractal second derivative at query points z via interpolation."""
    p = fractal_dd["partition"]
    z_arr = np.asarray(z)

    if np.any(z_arr < p[0]) or np.any(z_arr > p[-1]):
        raise ValueError(
            f"Point {z} outside interpolation domain [{p[0]}, {p[-1]}]"
        )

    return np.interp(z, p, fractal_dd["values"])


# Alias for backward compatibility
d2_fractal_phi_pointwise = pointwise_fractal


def d2_fractal_psi(a, j, x, fractal_dd):
    """Second derivative of the fractal basis function psi_j''(a).

    Parameters
    ----------
    a : float or array_like
        Evaluation coordinate.
    j : int
        Basis index in {0, 1, ..., n}.
    x : array_like
        Spatial grid nodes.
    fractal_dd : dict
        Precomputed fractal second derivative table from build_fractal_second_derivative.
    """
    n = len(x) - 1

    if j == 0:
        return (
            pointwise_fractal(a - x[1], fractal_dd)
            / (2 * (x[1] - x[0]))
        )

    if j == 1:
        return (
            (
                pointwise_fractal(a - x[2], fractal_dd)
                - pointwise_fractal(a - x[1], fractal_dd)
            )
            / (2 * (x[2] - x[1]))
            - pointwise_fractal(a - x[1], fractal_dd)
            / (2 * (x[1] - x[0]))
        )

    if j == n - 1:
        return (
            -pointwise_fractal(a - x[n - 1], fractal_dd)
            / (2 * (x[n] - x[n - 1]))
            - (
                pointwise_fractal(a - x[n - 1], fractal_dd)
                - pointwise_fractal(a - x[n - 2], fractal_dd)
            )
            / (2 * (x[n - 1] - x[n - 2]))
        )

    if j == n:
        return (
            pointwise_fractal(a - x[n - 1], fractal_dd)
            / (2 * (x[n] - x[n - 1]))
        )

    return (
        (
            pointwise_fractal(a - x[j + 1], fractal_dd)
            - pointwise_fractal(a - x[j], fractal_dd)
        )
        / (2 * (x[j + 1] - x[j]))
        - (
            pointwise_fractal(a - x[j], fractal_dd)
            - pointwise_fractal(a - x[j - 1], fractal_dd)
        )
        / (2 * (x[j] - x[j - 1]))
    )


def d2_fractal_L_W2(i, x, f, xk, alpha, s, fractal_dd):
    """Evaluates the second derivative of the fractal quasi-interpolant (L_W2 f)''(x_i).

    Parameters
    ----------
    i : int
        Grid index of the evaluation point x[i].
    x : array_like
        Spatial grid nodes.
    f : array_like
        Function values at nodes x.
    xk : array_like
        Interpolation centers.
    alpha : array_like
        RBF interpolation coefficients.
    s : float
        RBF shape parameter.
    fractal_dd : dict
        Precomputed fractal second derivative table.
    """
    xx = x[i]

    # RBF correction second derivative
    rbf_part = np.sum(alpha * varphi(xx - xk, s))

    # Error function evaluations e(x_p) = f(x_p) - R(x_p)
    e_vals = np.array([error_function(p, x, f, xk, alpha, s) for p in range(len(x))])

    # Quasi-interpolation correction second derivative
    ld_part = 0.0
    for p in range(len(x)):
        ld_part += e_vals[p] * d2_fractal_psi(xx, p, x, fractal_dd)

    return rbf_part + ld_part