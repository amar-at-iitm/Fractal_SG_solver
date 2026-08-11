import numpy as np
from tqdm import tqdm
from SG_solver import rbf_matrix, second_divided_difference  
from fractal_SG_solver import ddphi, H5_dd, pointwise_fractal, d2_fractal_L_W2
from alpha_fractal_function import alpha_fractalize, alpha_fractalize_second_derivative

import os
import wandb

# --- ADDED FOR MULTIPROCESSING ---
from joblib import Parallel, delayed

# --- DAY/NIGHT TOGGLE ---
# True: Uses 16 cores for overnight runs (leaves 4 for OS background tasks)
# False: Uses 2 cores for daytime runs so you can use your laptop normally
NIGHT_MODE = True
CORES_TO_USE = 16 if NIGHT_MODE else 2
# ---------------------------------

from fractal_sweep_config import sweep_config

# Authenticate with Weights & Biases (uses WANDB_API_KEY environment variable or cached login)
wandb_key = os.environ.get("WANDB_API_KEY")
if wandb_key:
    wandb.login(key=wandb_key)
else:
    wandb.login()

# Paper Example 1 parameters.
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
T = 1

n = (b - a) / h

if n != int(n):
    raise ValueError("h must divide b-a exactly.")

n = int(n)

if n % K != 0:
    raise ValueError("n must be divisible by K because the paper defines N = n / K.")

N = n // K
Nt = int(round(T / tau))

x = np.linspace(a, b, n + 1)

# Interpolation center indices k_j, j = 1,...,N
k_idx = np.concatenate(([1], K * np.arange(1, N - 1), [n - 1])).astype(int)

xk = x[k_idx]
xkj = []
for j in range(1, N+1):
    if j == 1:
        value = float(x[1])
        xkj.append(value)

    elif 1 < j < N:
        value = a + ((j - 1) * (b - a)) / N
        xkj.append(value)
    elif j == N:
        value = float(x[n - 1])
        xkj.append(value)

# Define the base function f(x) = sin(pi * x) 
def f(x):
    y = np.sin(np.pi * x)
    return np.where(np.isclose(y, 0.0, atol=1e-12), 0.0, y)

# Odd base function
def g(x):
    return x * (1 - x**2)


f_beta = [ 0.005, 0.0025, 0.0025, 0.005]
subintervals = len(f_beta)
iter = 6
sine_pi_fractal = alpha_fractalize(f, g, -1, 1, subintervals, f_beta, iter)


def odd_dirichlet_extension(z):
    y = ((z + 1) % 4) - 1   # maps to [-1, 3)

    if y <= 1:
        return pointwise_fractal(y, sine_pi_fractal)

    return -pointwise_fractal(2 - y, sine_pi_fractal)

def f_exact_u(x, t):
    return 0.5 * (
        odd_dirichlet_extension(x + t)
        + odd_dirichlet_extension(x - t)
    )
# Compute the exact solution at time T
f_u_exact = np.asarray([f_exact_u(xi, T) for xi in x])


best_Linf_error = float('inf')
best_RMS_error = float('inf')

def fractal_optimization():
    global best_Linf_error, best_RMS_error

    wandb.init(settings=wandb.Settings(init_timeout=3000))
    config = wandb.config
    f_alpha1 = config.f_alpha1
    f_alpha2 = config.f_alpha2
    f_alpha3 = config.f_alpha3

    run_name = f"alpha1-{f_alpha1}_alpha2-{f_alpha2}_alpha3-{f_alpha3}"
    wandb.run.name = run_name

    # f_alpha = [0.0005, 0.002, 0.0005, 0.0005, 0.002, 0.0005]
    f_alpha = [f_alpha1, f_alpha2, f_alpha3, f_alpha3, f_alpha2, f_alpha1]

    n_subintervals =len(f_alpha)
    n_iter = 2
    fractal_dd = alpha_fractalize_second_derivative(ddphi, H5_dd, -2, 2, n_subintervals, f_alpha, n_iter)

    f_U = np.zeros((Nt + 2, len(x)))

    # Start the Parallel worker pool ONCE for the entire run
    # This prevents the massive overhead of restarting processes every time-step
    with Parallel(n_jobs=CORES_TO_USE) as parallel:

        # Initial condition at t=0
        init_vals = parallel(delayed(pointwise_fractal)(x[d], sine_pi_fractal) for d in range(len(x)))
        f_U[0, :] = np.asarray(init_vals)

        f_U[0, 0]  = 0.0
        f_U[0, -1] = 0.0

        # Build approximation of u_xx at t=0
        A0 = rbf_matrix(xk, s)
        
        rhs0_list = parallel(delayed(second_divided_difference)(x, f_U[0, :], kj) for kj in k_idx)
        rhs0 = np.asarray(rhs0_list)

        alpha0 = np.linalg.solve(A0, rhs0)

        f_uxx0_list = parallel(delayed(d2_fractal_L_W2)(i, x, f_U[0, :], xk, alpha0, s, fractal_dd) for i in range(len(x)))
        f_uxx0 = np.asarray(f_uxx0_list)

        # Since g(x)=0
        f_U[1, :] = f_U[0, :] + 0.5 * tau**2 * f_uxx0
        f_U[1, 0]  = 0.0
        f_U[1, -1] = 0.0

        #------------------------------------
        # Time stepping:
        for d in tqdm(range(1, Nt + 1), desc="Time stepping", unit="step"):
            A = rbf_matrix(xk, s)
            
            rhs_d_list = parallel(delayed(second_divided_difference)(x, f_U[d], kj) for kj in k_idx)
            rhs_d = np.asarray(rhs_d_list)
            
            alpha2 = np.linalg.solve(A, rhs_d)

            f_uxx_list = parallel(delayed(d2_fractal_L_W2)(i, x, f_U[d], xk, alpha2, s, fractal_dd) for i in range(len(x)))
            f_uxx = np.asarray(f_uxx_list)

            f_U[d + 1, :] = 2.0 * f_U[d, :] - f_U[d - 1, :] + tau**2 * f_uxx

            # Dirichlet boundary conditions
            f_U[d + 1, 0] = 0.0
            f_U[d + 1, -1] = 0.0
        #-----------------------------------


    f_u_num = f_U[Nt, :]

    f_err = (f_u_num - f_u_exact)
    f_abs_err = np.abs(f_err)

    Linf_error = np.max(f_abs_err)
    RMS_error = np.sqrt(np.mean(f_abs_err**2))

    if Linf_error < best_Linf_error:
        best_Linf_error = Linf_error
        best_RMS_error = RMS_error
        print(f"New best Linf error: {best_Linf_error}")
        print(f"New best RMS error: {best_RMS_error}")

        file_name = f"best_results_T_{T}.txt"
        with open(file_name, "w") as f:
            f.write(f"At time T={T}:\n")
            f.write(f"Best Linf error: {best_Linf_error}\n")
            f.write(f"Best RMS error: {best_RMS_error}\n")
            f.write(f"Parameters: alpha1={f_alpha1}, alpha2={f_alpha2}, alpha3={f_alpha3}\n")

    wandb.log({
        "alpha1": f_alpha1,
        "alpha2": f_alpha2,
        "alpha3": f_alpha3,
        "RMS_error": RMS_error,
        "Linf_error": Linf_error
    })

    wandb.finish()

# ---------------------------------------------------
# Run sweep
# ---------------------------------------------------
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="SG_fractal_optimization")
    wandb.agent(sweep_id, function=fractal_optimization)
    print("Sweep complete")