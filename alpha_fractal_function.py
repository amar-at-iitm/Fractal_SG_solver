import numpy as np


def _alpha_fractalize_engine(func, base_func, a, b, n_sub, in_alpha, n_iter, derivative_order=0, dict=True):
    """
    Core high-performance engine for evaluating alpha-fractal functions and their derivatives.

    Mathematical Formulation:
    -------------------------
    Let I = [a, b] = [x_0, x_N] be partitioned into N subintervals I_i = [x_{i-1}, x_i] (i = 1, ..., N).
    Each subinterval is the image of I under the affine contraction:
        L_i(x) = A[i] * x + B[i],
    where
        A[i] = (x_i - x_{i-1}) / (x_N - x_0) > 0,
        B[i] = (x_N * x_{i-1} - x_0 * x_i) / (x_N - x_0).

    The Read-Bajraktarevic (RB) self-referential equation for the p-th derivative (p >= 0) is:
        (f^alpha)^{(p)}(L_i(x)) = f^{(p)}(L_i(x)) + (alpha_i / A[i]^p) * [ (f^alpha)^{(p)}(x) - g^{(p)}(x) ].

    Performance Optimizations:
    --------------------------
    1. Preallocation: The exact point count at stage k is M_k = N^k * (len(X) - 1) + 1.
       Preallocating contiguous arrays avoids dynamic resizing and list extension overhead.
    2. Zero-Duplicate Ordered Construction:
       Since each L_i is strictly increasing (A[i] > 0), the points within block i are strictly
       sorted in [x_{i-1}, x_i]. The right boundary of block i-1 equals the left boundary of block i:
           L_{i-1}(x_N) = x_i = L_i(x_0).
       By mapping all points for block 0 (size M) and slicing x_curr[1:] for blocks i >= 1 (size M-1),
       the partition is generated strictly monotonically with zero duplicate boundary points.
       This completely eliminates the need for expensive np.argsort and np.unique operations.
    3. Vectorized Evaluation:
       - (y_old - g(x_old)) is evaluated once per iteration stage rather than N times.
       - Vectorized affine scaling and function evaluation across NumPy arrays.
    """
    # 1. Discretize base domain [a, b] into N subintervals
    X = np.linspace(a, b, n_sub + 1)
    N = len(X) - 1

    # 2. Parse scaling vector alpha
    if np.isscalar(in_alpha):
        alpha = np.full(N, in_alpha, dtype=float)
    else:
        alpha = np.asarray(in_alpha, dtype=float)

    if len(alpha) != N:
        raise ValueError("alpha must have length len(X)-1.")

    x0 = X[0]
    xN = X[-1]
    inv_span = 1.0 / (xN - x0)

    # 3. Compute affine coefficients A[i] and B[i] for L_i(x) = A[i]*x + B[i]
    A = (X[1:] - X[:-1]) * inv_span
    B = (xN * X[:-1] - x0 * X[1:]) * inv_span

    # 4. Effective scale factor: alpha_i / (A_i ^ p)
    #    p = 0: alpha_i
    #    p = 1: alpha_i / A_i
    #    p = 2: alpha_i / A_i^2
    if derivative_order == 0:
        scale = alpha
    elif derivative_order == 1:
        scale = alpha / A
    elif derivative_order == 2:
        scale = alpha / (A ** 2)
    else:
        scale = alpha / (A ** derivative_order)

    # 5. Stage 0: Initial nodal coordinates and function values
    x_curr = X
    y_curr = np.asarray(func(X), dtype=float)

    # 6. Iteration loop (Picard iterations on the Read-Bajraktarevic operator)
    for _ in range(n_iter):
        # Evaluate base perturbation difference once per stage
        diff = y_curr - np.asarray(base_func(x_curr), dtype=float)

        M = len(x_curr)
        new_M = N * (M - 1) + 1  # Total unique points in refined partition

        x_next = np.empty(new_M, dtype=float)
        y_next = np.empty(new_M, dtype=float)

        # Block 0: covers [x_0, x_1] using all M points from x_curr
        x_next[:M] = A[0] * x_curr + B[0]
        y_next[:M] = func(x_next[:M]) + scale[0] * diff

        # Blocks 1 to N-1: cover (x_{i-1}, x_i] using x_curr[1:]
        # Slicing from index 1 omits x_curr[0], eliminating boundary duplicates
        idx = M
        block_len = M - 1
        for i in range(1, N):
            next_idx = idx + block_len
            xi = A[i] * x_curr[1:] + B[i]
            x_next[idx:next_idx] = xi
            y_next[idx:next_idx] = func(xi) + scale[i] * diff[1:]
            idx = next_idx

        x_curr = x_next
        y_curr = y_next

    if dict:
        return {
            "partition": x_curr,
            "values": y_curr,
        }
    else:
        return x_curr, y_curr


def alpha_fractalize_second_derivative(f2, g2, a, b, n_sub, in_alpha, n_iter, dict=True):
    """
    Computes the second derivative of the alpha-fractal function, (f^alpha)''.

    Mathematical Formula:
        (f^alpha)''(L_i(x)) = f''(L_i(x)) + (alpha_i / a_i^2) * [ (f^alpha)''(x) - g''(x) ]
    where:
        - f2: callable representing f''(x)
        - g2: callable representing g''(x)
        - a, b: interval endpoints [a, b]
        - n_sub: number of subintervals N
        - in_alpha: scaling factor (scalar or array of length N)
        - n_iter: number of recursion stages
        - dict: if True, returns a dict with keys 'partition' and 'values'
    """
    return _alpha_fractalize_engine(
        func=f2,
        base_func=g2,
        a=a,
        b=b,
        n_sub=n_sub,
        in_alpha=in_alpha,
        n_iter=n_iter,
        derivative_order=2,
        dict=dict,
    )


def alpha_fractalize_first_derivative(f1, g1, a, b, n_sub, in_alpha, n_iter, dict=True):
    """
    Computes the first derivative of the alpha-fractal function, (f^alpha)'.

    Mathematical Formula:
        (f^alpha)'(L_i(x)) = f'(L_i(x)) + (alpha_i / a_i) * [ (f^alpha)'(x) - g'(x) ]
    where:
        - f1: callable representing f'(x)
        - g1: callable representing g'(x)
        - a, b: interval endpoints [a, b]
        - n_sub: number of subintervals N
        - in_alpha: scaling factor (scalar or array of length N)
        - n_iter: number of recursion stages
        - dict: if True, returns a dict with keys 'partition' and 'values'
    """
    return _alpha_fractalize_engine(
        func=f1,
        base_func=g1,
        a=a,
        b=b,
        n_sub=n_sub,
        in_alpha=in_alpha,
        n_iter=n_iter,
        derivative_order=1,
        dict=dict,
    )


def alpha_fractalize(f, g, a, b, n_sub, in_alpha, n_iter, dict=True):
    """
    Computes the alpha-fractal function f^alpha.

    Mathematical Formula:
        f^alpha(L_i(x)) = f(L_i(x)) + alpha_i * [ f^alpha(x) - g(x) ]
    where:
        - f: original continuous function f(x)
        - g: base continuous function g(x) satisfying g(a) == f(a) and g(b) == f(b)
        - a, b: interval endpoints [a, b]
        - n_sub: number of subintervals N
        - in_alpha: scaling factor (scalar or array of length N)
        - n_iter: number of recursion stages
        - dict: if True, returns a dict with keys 'partition' and 'values'
    """
    fa, fb = f(a), f(b)
    ga, gb = g(a), g(b)

    # Print boundary evaluations matching original behavior
    print(f"f(-1) = {fa}, f(1) = {fb}")
    print(f"g(-1) = {ga}, g(1) = {gb}")

    # Boundary verification: g must match f at domain endpoints for C^0 continuity.
    # Uses numerical tolerance to avoid false positives due to machine epsilon.
    if not (np.isclose(ga, fa, atol=1e-12) and np.isclose(gb, fb, atol=1e-12)):
        raise ValueError("The boundary conditions of g must match those of f.")

    return _alpha_fractalize_engine(
        func=f,
        base_func=g,
        a=a,
        b=b,
        n_sub=n_sub,
        in_alpha=in_alpha,
        n_iter=n_iter,
        derivative_order=0,
        dict=dict,
    )
