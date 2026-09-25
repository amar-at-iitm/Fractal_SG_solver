import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from SG_solver import rbf_matrix, second_divided_difference

from fractal_SG_solver import d2_fractal_L_W2, ddphi, H5_dd, pointwise_fractal

from alpha_fractal_function import alpha_fractalize, alpha_fractalize_second_derivative

import os
import wandb

from fractal_sweep_config import sweep_config

# Authenticate with Weights & Biases (uses WANDB_API_KEY environment variable or cached login)
wandb_key = os.environ.get("WANDB_API_KEY")
if wandb_key:
    wandb.login(key=wandb_key)
else:
    try:
        wandb.login()
    except Exception as e:
        print(f"wandb login skipped or not configured: {e}")

# ==============================================================================
# Domain and Discretization Parameters
# ==============================================================================
a = -1.0
b = 1.0
#######################
K = 8
s = 0.8
c = 0.027 
################
h = 0.01
tau = 0.01
################
T = 1.0

n = (b - a) / h

if n != int(n):
    raise ValueError("h must divide b-a exactly.")

n = int(n)

if n % K != 0:
    raise ValueError("n must be divisible by K because the paper defines N = n / K.")

N = n // K
Nt = int(round(T / tau))

# Spatial grid: 200 intervals, 201 points from -1 to 1 with h=0.01
x = np.linspace(a, b, n + 1)

# Interpolation center indices k_j, j = 1,...,N
k_idx = np.concatenate(([1], K * np.arange(1, N - 1), [n - 1])).astype(int)

xk = x[k_idx]
xkj = []
for j in range(1, N + 1):
    if j == 1:
        value = float(x[1])
        xkj.append(value)
    elif 1 < j < N:
        value = a + ((j - 1) * (b - a)) / N
        xkj.append(value)
    elif j == N:
        value = float(x[n - 1])
        xkj.append(value)

# ==============================================================================
# Initial Displacement Function u(x, 0) = sin^{\alpha}(x)
# ==============================================================================
# Base function f(x) = sin(x) (Changed from sin(pi * x) to sin(x))
def f(x_val):
    return np.sin(x_val)

# Base function g(x) satisfying endpoint conditions g(a) == f(a) and g(b) == f(b)
# For f(x) = sin(x) on [-1, 1], the linear base function joining (-1, sin(-1)) and (1, sin(1)) is:
def g(x_val):
    return np.sin(1.0) * x_val

# Fractal parameters for generating the alpha-fractal function sin^{\alpha}(x)
f_beta = [0.005, 0.0025, 0.0025, 0.005]
subintervals = len(f_beta)
iter_count = 6
sine_fractal = alpha_fractalize(f, g, -1, 1, subintervals, f_beta, iter_count)

def sin_alpha(x_pts):
    """
    Initial displacement function u(x, 0) = sin^{\alpha}(x).
    Evaluates the alpha-fractalized sine function constructed via alpha_fractalize.
    The analytical solution depends on this choice of initial displacement.
    """
    return pointwise_fractal(x_pts, sine_fractal)

# Pre-evaluate and STORE sin^{\alpha}(x) on the spatial grid x (201 points, h=0.01)
sin_alpha_grid = np.asarray(pointwise_fractal(x, sine_fractal))

# ==============================================================================
# Analytical Exact Solution via Fourier Series & Trapezoidal Rule
# ==============================================================================
# PDE:
#   u_{tt} = u_{xx},            -1 < x < 1, t > 0
#   u(x, 0) = sin^{\alpha}(x),  -1 <= x <= 1
#   u_t(x, 0) = 0,              -1 <= x <= 1
#   u(-1, t) = u(1, t) = 0,     t >= 0
#
# Analytical Solution:
#   u(x, t) = \sum_{n=1}^{\infty} A_n \cos(n*pi*t/2) \sin(n*pi*(x+1)/2)
#
# where:
#   A_n = \int_{-1}^{1} \sin^{\alpha}(x) \sin(n*pi*(x+1)/2) dx
#
# Truncation: \sum_{n=1}^{M}
#
# Trapezoidal Formula for integration on partition a = x_0 < x_1 < ... < x_N = b:
#   \int_a^b f(x) dx \approx (h/2) * [f(x_0) + 2*{f(x_1) + ... + f(x_{N-1})} + f(x_N)]
# ==============================================================================

# User-configurable parameters (can be changed to try different numbers):
M = 2000            # Number of terms in Fourier series: \sum_{n=1}^{M}
N_partition = 2000 # Number of partitions for Trapezoidal rule: a = x_0 < ... < x_N = b

# Partition of [-1, 1]: a = x_0 < x_1 < ... < x_N = b
x_nodes = np.linspace(a, b, N_partition + 1)
h_trap = (b - a) / N_partition

# ------------------------------------------------------------------------------
# Pre-evaluate and STORE sin^{\alpha}(x) from fractalized function BEFORE computing A_n
# ------------------------------------------------------------------------------
print(f"Calling and storing sin^alpha(x) from fractalized function at {len(x_nodes)} partition nodes...")
sin_alpha_partition = np.asarray(pointwise_fractal(x_nodes, sine_fractal))

def trapezoidal_rule(f_vals, h_step):

    return (h_step / 2.0) * (f_vals[0] + 2.0 * np.sum(f_vals[1:-1]) + f_vals[-1])

def compute_fourier_coefficients(sin_alpha_vals, x_part, h_step, M_terms):
    A_coeffs = np.zeros(M_terms)
    for idx, n_mode in enumerate(range(1, M_terms + 1)):
        # Spatial basis: sin(n * pi * (x + 1) / 2)
        basis = np.sin(n_mode * np.pi * (x_part + 1.0) / 2.0)
        
        # Integrand: sin^{\alpha}(x) * sin(n * pi * (x + 1) / 2)
        # Using pre-stored values of sin^{\alpha}(x)
        integrand = sin_alpha_vals * basis
        
        # Trapezoidal rule:
        A_coeffs[idx] = trapezoidal_rule(integrand, h_step)
        
    return A_coeffs

def analytical_exact_u(x_pts, t_val, A_coeffs, M_terms):
    x_arr = np.asarray(x_pts)
    u_vals = np.zeros_like(x_arr, dtype=float)

    for idx, n_mode in enumerate(range(1, M_terms + 1)):
        A_n_val = A_coeffs[idx]
        time_factor = np.cos(n_mode * np.pi * t_val / 2.0)
        spatial_factor = np.sin(n_mode * np.pi * (x_arr + 1.0) / 2.0)
        u_vals += A_n_val * time_factor * spatial_factor

    return u_vals

# ------------------------------------------------------------------------------
# Precompute Analytical Exact Solution (Calculated ONCE outside optimization loop)
# ------------------------------------------------------------------------------
# At t = 1, u(x, t) is computed at the 200 spatial intervals (201 points) with h = 0.01
# from -1 to 1, exactly matching the numerical solution grid for direct comparison.
print("=" * 70)
print(f"Precomputing Analytical Exact Solution (CALCULATED ONCE):")
print(f"  - Initial displacement: sin^alpha(x) (from sin(x))")
print(f"  - Series truncation M = {M} terms")
print(f"  - Trapezoidal partition N = {N_partition} intervals")
print(f"  - Target time T = {T}")
print(f"  - Grid points = {len(x)} (h = {h})")
print("=" * 70)

A_n = compute_fourier_coefficients(sin_alpha_partition, x_nodes, h_trap, M)
f_u_exact = analytical_exact_u(x, T, A_n, M)

print(f"Exact solution computed successfully. ||u_exact||_max = {np.max(np.abs(f_u_exact)):.6f}")
print("=" * 70)

# ==============================================================================
# Optimization Routine
# ==============================================================================
best_Linf_error = float('inf')
best_RMS_error = float('inf')

def fractal_optimization():
    global best_Linf_error, best_RMS_error

   
    wandb.init(settings=wandb.Settings(init_timeout=3000))
    config = wandb.config
    f_alpha1 = getattr(config, "f_alpha1", 0.0005)
    f_alpha2 = getattr(config, "f_alpha2", 0.002)
    f_alpha3 = getattr(config, "f_alpha3", 0.0005)

    run_name = f"alpha1-{f_alpha1}_alpha2-{f_alpha2}_alpha3-{f_alpha3}"
    wandb.run.name = run_name
    use_wandb = True


    f_alpha = [f_alpha1, f_alpha2, f_alpha3, f_alpha3, f_alpha2, f_alpha1]

    n_subintervals = len(f_alpha)
    n_iter = 6
    fractal_dd = alpha_fractalize_second_derivative(ddphi, H5_dd, -2, 2, n_subintervals, f_alpha, n_iter)

    # Initial condition at t=0: u(x, 0) = sin^{\alpha}(x) (from pre-stored array)
    f_U = np.zeros((Nt + 2, len(x)))
    f_U[0, :] = sin_alpha_grid.copy()
    f_U[0, 0] = 0.0
    f_U[0, -1] = 0.0

    # Build approximation of u_xx at t=0
    A0 = rbf_matrix(xk, s)
    rhs0 = []
    for kj in k_idx:
        rhs0.append(second_divided_difference(x, f_U[0, :], kj))
    rhs0 = np.asarray(rhs0)

    alpha0 = np.linalg.solve(A0, rhs0)

    f_uxx0 = np.zeros(len(x))
    for i in range(len(x)):
        f_uxx0[i] = d2_fractal_L_W2(i, x, f_U[0, :], xk, alpha0, s, fractal_dd)

    # Since u_t(x, 0) = 0: u^1 = u^0 + 0.5 * tau^2 * u_xx^0
    f_U[1, :] = f_U[0, :] + 0.5 * tau**2 * f_uxx0
    f_U[1, 0] = 0.0
    f_U[1, -1] = 0.0

    # ------------------------------------
    # Time stepping:
    # ------------------------------------
    for d in tqdm(range(1, Nt + 1), desc="Time stepping", unit="step"):
        A_mat = rbf_matrix(xk, s)
        rhs_d = np.asarray([second_divided_difference(x, f_U[d], kj) for kj in k_idx])
        alpha2 = np.linalg.solve(A_mat, rhs_d)

        f_uxx = np.zeros(len(x))
        for i in range(len(x)):
            f_uxx[i] = d2_fractal_L_W2(i, x, f_U[d], xk, alpha2, s, fractal_dd)

        f_U[d + 1, :] = 2.0 * f_U[d, :] - f_U[d - 1, :] + tau**2 * f_uxx

        # Dirichlet boundary conditions: u(-1, t) = u(1, t) = 0
        f_U[d + 1, 0] = 0.0
        f_U[d + 1, -1] = 0.0
    # ------------------------------------

    f_u_num = f_U[Nt, :]

    # Comparison with precalculated exact solution
    f_err = (f_u_num - f_u_exact)
    f_abs_err = np.abs(f_err)

    Linf_error = np.max(f_abs_err)
    RMS_error = np.sqrt(np.mean(f_abs_err**2))

    if Linf_error < best_Linf_error:
        best_Linf_error = Linf_error
        best_RMS_error = RMS_error
        print(f"\nNew best Linf error: {best_Linf_error:.6e}")
        print(f"New best RMS error: {best_RMS_error:.6e}")

        file_name = f"best_results_analytical_T_{T}.txt"
        with open(file_name, "w") as f_out:
            f_out.write(f"At time T={T}:\n")
            f_out.write(f"Fourier Series M={M}, Trapezoidal Partitions N={N_partition}\n")
            f_out.write(f"Best Linf error: {best_Linf_error:.6e}\n")
            f_out.write(f"Best RMS error: {best_RMS_error:.6e}\n")
            f_out.write(f"Parameters: alpha1={f_alpha1}, alpha2={f_alpha2}, alpha3={f_alpha3}\n")

    if use_wandb:
        wandb.log({
            "alpha1": f_alpha1,
            "alpha2": f_alpha2,
            "alpha3": f_alpha3,
            "RMS_error": RMS_error,
            "Linf_error": Linf_error
        })
        wandb.finish()

    return Linf_error, RMS_error

# ==============================================================================
# Main Execution / Sweep
# ==============================================================================
if __name__ == "__main__":
    import sys

    # Allow running a single trial or full wandb sweep
    if "--sweep" in sys.argv:
        sweep_id = wandb.sweep(sweep_config, project="SG_fractal_optimization_analytical")
        wandb.agent(sweep_id, function=fractal_optimization)
        print("Sweep complete")
    else:
        print("Running single trial with precomputed analytical solution...")
        linf, rms = fractal_optimization()
        print(f"Single run completed: Linf error = {linf:.6e}, RMS error = {rms:.6e}")
        print("To run full wandb sweep, pass --sweep argument.")

