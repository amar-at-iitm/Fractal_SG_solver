import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

from SG_solver import rbf_matrix, second_divided_difference, paper_rms
from fractal_SG_solver import d2_fractal_L_W2, ddphi, H5_dd, pointwise_fractal
from alpha_fractal_function import alpha_fractalize, alpha_fractalize_second_derivative
from fractal_sweep_config import sweep_config

# Authenticate with Weights & Biases (uses WANDB_API_KEY environment variable or cached login)
wandb_key = os.environ.get("WANDB_API_KEY")
if wandb_key:
    wandb.login(key=wandb_key)
else:
    wandb.login()

# Paper Example 2 parameters.
a = -2.0
b = 2.0
#######################
K = 10
s = 1.0
h = 0.01
tau = 0.01
c = 0.1 * (h**(1.0 / 3.0))
################
T = 1.0
###################

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


def exact_u(x, t):
    return 4.0 * np.arctan((1.0 / np.cosh(x)) * t)


# Compute the exact solution at time T
u_exact = exact_u(x, T)


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

    f_alpha = [f_alpha1, f_alpha2, f_alpha3, f_alpha3, f_alpha2, f_alpha1]

    n_subintervals = len(f_alpha)
    n_iter = 2
    fractal_dd = alpha_fractalize_second_derivative(
        lambda z: ddphi(z, c=c),
        lambda z: H5_dd(z, x1=a - b, xN=b - a, c=c),
        a - b,
        b - a,
        n_subintervals,
        f_alpha,
        n_iter,
    )

    # Initial condition at t=0
    f_U = np.zeros((Nt + 2, len(x)))
    f_U[0, :] = exact_u(x, 0.0)  # initial condition at t=0

    # Boundary conditions
    f_U[0, 0]  = exact_u(x[0], 0.0)
    f_U[0, -1] = exact_u(x[-1], 0.0)

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

    # Since u(x, 0) = 0 and u_t(x, 0) = h(x) = 4*sech(x)
    f_U[1, :] = f_U[0, :] + tau * (4.0 / np.cosh(x)) + 0.5 * tau**2 * (f_uxx0 - np.sin(f_U[0, :]))

    #------------------------------------
    # Time stepping:
    A = rbf_matrix(xk, s)
    for d in tqdm(range(1, Nt + 1), desc="Time stepping", unit="step"):
        rhs_d = np.asarray([second_divided_difference(x, f_U[d], kj) for kj in k_idx])
        alpha2 = np.linalg.solve(A, rhs_d)

        f_uxx = np.zeros(len(x))
        for i in range(len(x)):
            f_uxx[i] = d2_fractal_L_W2(i, x, f_U[d], xk, alpha2, s, fractal_dd)

        f_U[d + 1, :] = 2.0 * f_U[d, :] - f_U[d - 1, :] + tau**2 * f_uxx - tau**2 * np.sin(f_U[d, :])

    #-----------------------------------

    f_u_num = f_U[Nt, :]

    f_err = -(u_exact - f_u_num)
    f_abs_err = np.abs(f_err)

    Linf_error = np.max(f_abs_err)
    RMS_error = np.sqrt(np.mean(f_abs_err**2))
    paper_RMS_error = paper_rms(u_exact, f_u_num)

    if Linf_error < best_Linf_error:
        best_Linf_error = Linf_error
        best_RMS_error = RMS_error
        print(f"New best Linf error: {best_Linf_error}")
        print(f"New best RMS error: {best_RMS_error}")
        print(f"New best Paper RMS error: {paper_RMS_error}")

        file_name = f"best_results_T_{T}.txt"
        with open(file_name, "w") as f:
            f.write(f"At time T={T}:\n")
            f.write(f"Best Linf error: {best_Linf_error}\n")
            f.write(f"Best RMS error: {best_RMS_error}\n")
            f.write(f"Best Paper RMS error: {paper_RMS_error}\n")
            f.write(f"Parameters: alpha1={f_alpha1}, alpha2={f_alpha2}, alpha3={f_alpha3}\n")

    wandb.log({
        "alpha1": f_alpha1,
        "alpha2": f_alpha2,
        "alpha3": f_alpha3,
        "RMS_error": RMS_error,
        "paper_RMS_error": paper_RMS_error,
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