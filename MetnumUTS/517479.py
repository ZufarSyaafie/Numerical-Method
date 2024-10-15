# --------------------------------IMPORT LIBS-----------------------------------
import math
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------IMPORT DATA---------------------------------
df = pd.read_csv(r"D:\CODING\Python\MetnumUTS\data.csv")  # Change path if needed
df.set_index("Minute", inplace=True)
x = np.array(df.index)
y = np.array(df["Output (kW)"])
collection_error = []


# Function to save figure
def save_fig(x, y, x_hat, y_hat, title):
    fig, ax = plt.subplots()
    plt.scatter(x, y, label="Data", marker="o", color="b")
    plt.plot(x_hat, y_hat, label="Regression", color="r")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    fig.savefig(f"{title}.png", dpi=300, bbox_inches="tight")


# --------------------------------LINEAR REGRESSION-----------------------------------
# Linear Regression Function
def linear_regression(x, y):
    n = np.size(x)

    # Find the value of a_0 and a_1
    a_0 = (np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / (
        n * np.sum(x**2) - np.sum(x) ** 2
    )
    a_1 = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
        n * np.sum(x**2) - np.sum(x) ** 2
    )

    y_bar = np.mean(y)
    y_hat = a_0 + a_1 * x

    Sy = np.sqrt(np.sum((y - y_bar) ** 2) / (n - 1))
    Syx = np.sqrt((np.sum((y - y_hat) ** 2)) / (n - 2))
    r2 = (Sy**2 - Syx**2) / Sy**2
    r = np.sqrt(r2)

    return [a_0, a_1], Sy, Syx, r2, r, y_hat


# Execute Linear Regression
print(f"\n{30*'='}LINEAR REGRESSION{30*'='}")
linear_const, Sy, Syx, r2, r, y_hat = linear_regression(x, y)
execution_t = timeit.timeit(lambda: linear_regression(x, y), number=100)
print(f"Intercept = {linear_const[0]}")
print(f"Slope = {linear_const[1]}")
print(f"Standard Deviation = {Sy:.4f}")
print(f"Error = {Syx:.4f}")
print(f"R^2 = {r2:.2%}")
print(f"R = {r:.2%}")
print(f"Execution time (100 iterations): {(execution_t):.4f}s")
# save_fig(x, y, x, y_hat, "Linear Regression")
collection_error.append(Syx)


# --------------------------------POLYNOMIAL REGRESSION-----------------------------------
# Matrix for Polynomial Regression
def pol_matrix(x, y, n):
    x_ = np.zeros((n, n))
    y_ = np.zeros((n, 1))

    for i in range(n):
        y_[i][0] = np.sum(y * x**i)
        for j in range(n):
            count = i + j
            if i + j == 0:
                x_[i][j] = len(x)
                continue
            x_[i][j] = np.sum(x**count)

    return x_, y_


# Function to check if matrix is diagonal dominant
def diagonal_dom(x):
    n = x.shape[0]
    for i in range(n):
        sum = 0
        for j in range(n):
            if i != j:
                sum += abs(x[i, j])
        if abs(x[i, i]) <= sum:
            return False
    return True


# Function to round to nearest
def round_nearest(value, nearest):
    return np.ceil(value / nearest) * nearest


# LU Decomposition
def lu_decomposition(X, y):
    n = X.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = X[i, j] - sum(L[i, k] * U[k, j] for k in range(i))

            L[i, i] = 1
            for j in range(i + 1, n):
                L[j, i] = (X[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    z = np.zeros((n, 1))
    for i in range(n):
        z[i] = y[i] - sum(L[i, k] * z[k] for k in range(n))

    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - sum(U[i, k] * x[k] for k in range(i + 1, n))) / U[i, i]

    return [x.reshape(-1), 0, 0, 0]


# Gauss Jordan
def gauss_jordan(x):
    n = x.shape[0]

    for i in range(n):
        x[i] = x[i] / x[i][i]
        for j in range(n):
            if i != j:
                x[j] = x[j] - x[i] * x[j][i]

    return [x[:, -1], 0, 0]


# Jacobi Iteration
def jacobi_iteration(X, y, x_init=None, tol=1e-3, max_iter=1000):
    if diagonal_dom(X) == False:
        return ["\n----Matrix is not diagonal dominant----"]

    n = len(y)

    if x_init is None:
        x_init = np.zeros(n)

    Diag = np.diag(X)
    R = X - np.diagflat(Diag)

    x_final = [x_init.tolist()]
    error = [np.inf]

    for iter in range(max_iter):
        x_new = (y - np.dot(R, x_init)) / Diag
        temp_err = abs(np.linalg.norm((x_new - x_init) / x_new, ord=np.inf))
        error.append(temp_err)
        x_final.append(x_new.tolist())
        if round_nearest(temp_err, tol) <= tol:
            return [x_final[-1][-1], error, iter + 1]

        x_init = x_new

    return [x_final[-1][-1], error, iter + 1]


# Gauss Seidel
def gauss_seidel(X, y, x_init=None, tol=1e-3, max_iter=1000):
    if diagonal_dom(X) == False:
        return ["\n----Matrix is not diagonal dominant----"]

    n = len(y)

    if x_init is None:
        x = np.zeros(n)
    else:
        x = x_init.copy()

    x_final = [x.tolist()]
    error = [np.inf]

    for k in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            sum1 = np.dot(X[i, :i], x_new[:i])
            sum2 = np.dot(X[i, i + 1 :], x[i + 1 :])

            x_new[i] = ((y[i] - sum1 - sum2) / X[i, i]).item()

        temp_err = abs(np.linalg.norm((x_new - x) / x_new, ord=np.inf))

        error.append(temp_err)
        x_final.append(x_new.tolist())
        if round_nearest(temp_err, tol) <= tol:
            return [x_final[-1], error, k + 1]

        x = x_new

    return [x, error, max_iter]


# Polynomial Regression Function
def polynomial_regression(x, y, coef, m):

    if type(coef) == str:
        print(
            coef,
            "[The program will set everything to zero.]",
            "[Try to modify the matrix for better results]",
            sep="\n",
        )
        return np.zeros(m + 1), 0, 0, 0, 0, np.zeros_like(y)

    n = len(x)

    y_bar = np.mean(y)

    y_hat = 0

    for i in range(m + 1):
        y_hat = y_hat + coef[i] * x**i

    Sy = np.sqrt(np.sum((y - y_bar) ** 2) / (n - 1))
    Syx = np.sqrt((np.sum((y - y_hat) ** 2)) / (n - m - 1))
    r2 = (Sy**2 - Syx**2) / Sy**2
    r = np.sqrt(r2)

    return coef, Sy, Syx, r2, r, y_hat


# Execute Polynomial Regression
powers = [2, 3, 4, 5]
collection_error_polynomial = []
for power in powers:
    x_, y_ = pol_matrix(x, y, power + 1)
    gj_matrix = np.column_stack((x_, y_))
    list_func = [
        lu_decomposition(x_, y_),
        gauss_jordan(gj_matrix),
        gauss_seidel(x_, y_),
        jacobi_iteration(x_, y_),
    ]
    execution_t = [
        timeit.timeit(lambda: lu_decomposition(x_, y_), number=100),
        timeit.timeit(lambda: gauss_jordan(gj_matrix), number=100),
        timeit.timeit(lambda: gauss_seidel(x_, y_), number=100),
        timeit.timeit(lambda: jacobi_iteration(x_, y_), number=100),
    ]

    tittles = ["LU Decomposition", "Gauss-Jordan", "Gauss-Seidel", "Jacobi Iteration"]

    results = []
    print(f"\n{30*'='}POLYNOMIAL REGRESSION {power}th POWER{30*'='}")
    for func in list_func:
        results.append(func[0])

    for idx, result in enumerate(results):
        tittle = tittles[idx]
        coef, Sy, Syx, r2, r, y_hat = polynomial_regression(x, y, result, power)
        print(f"\n{5*'='}Polynomial Regression {tittle}{5*'='}")
        print(f"Coef = {coef}")
        print(f"Standard Deviation = {Sy:.4f}")
        print(f"Error = {Syx:.4f}")
        print(f"R^2 = {r2:.2%}")
        print(f"R = {r:.2%}")
        print(f"Execution time(100 iterations): {(execution_t[idx]):.4f}s")
        collection_error_polynomial.append(Syx)
        # save_fig(x, y, x, y_hat, f"Polynomial Regression {power}th Power - {tittle}")

collection_error_polynomial = np.array(collection_error_polynomial)
collection_error_polynomial[collection_error_polynomial == 0] = np.nan
collection_error_polynomial = np.nanmean(collection_error_polynomial)
collection_error.append(collection_error_polynomial)


# --------------------------------EXPONENTIAL REGRESSION-----------------------------------
# Secant Method
def secant(f, a, b, e, N=100):
    step = 1
    condition = True
    while condition:
        if f(a) == f(b):
            break

        m = a - (b - a) * f(a) / (f(b) - f(a))
        error = abs(m - a) / abs(m)

        a = b
        b = m
        step = step + 1

        if step > N:
            break

        condition = error > e
    return m


# Modified Secant Method
def mod_secant(f, a, delta, e, N=1000):
    step = 1
    condition = True
    while condition:
        if f(a + delta) == f(a):
            # print("Divide by zero error!")
            break

        m = a - delta * f(a) / (f(a + delta) - f(a))
        error = abs(m - a) / abs(m)
        a = m
        step = step + 1

        if step > N:
            break

        condition = error > e

    return m


# Bisection Method
def bisection(f, a, b, tol):
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("The scalars a and b do not bound a root")

    m_old = None
    iter_count = 1

    while True:
        m = (a + b) / 2

        if m_old is None:
            error = float("inf")
        else:
            error = np.abs(m - m_old) / np.abs(m)

        if error < tol and iter_count > 1:
            return m

        m_old = m

        if np.sign(f(a)) == np.sign(f(m)):
            a = m
        else:
            b = m

        iter_count += 1


# Exponential Function for Regression
def exp_func(b):
    sum_yx_exp_bx = np.sum(y * x * np.exp(b * x))
    sum_y_exp_bx = np.sum(y * np.exp(b * x))
    sum_exp_2bx = np.sum(np.exp(2 * b * x))
    sum_x_exp_2bx = np.sum(x * np.exp(2 * b * x))

    second_term = (sum_y_exp_bx / sum_exp_2bx) * sum_x_exp_2bx

    result = sum_yx_exp_bx - second_term
    return result


# Exponential Regression Function
def exp_regression(x, y, a_1):
    n = np.size(x)
    a_0 = np.sum(y * np.exp(a_1 * x)) / np.sum(np.exp(2 * a_1 * x))

    y_bar = np.mean(y)
    y_hat = a_0 * np.exp(a_1 * x)

    Sy = np.sqrt(np.sum((y - y_bar) ** 2) / (n - 1))
    Syx = np.sqrt((np.sum((y - y_hat) ** 2)) / (n - 2))
    r2 = (Sy**2 - Syx**2) / Sy**2
    r = np.sqrt(r2)

    return [a_0, a_1], Sy, Syx, r2, r, y_hat


# Execute Exponential Regression
print(f"\n{30*'='}EXPONENTIAL REGRESSION{30*'='}")
a1 = [
    bisection(exp_func, -1, 1, 0.001),
    secant(exp_func, 0.1, 0.11, 0.001),
    mod_secant(exp_func, 1, 0.01, 0.001),
]
execution_t = [
    timeit.timeit(lambda: bisection(exp_func, -1, 1, 0.001), number=100),
    timeit.timeit(lambda: secant(exp_func, 0.1, 0.11, 0.001), number=100),
    timeit.timeit(lambda: mod_secant(exp_func, 1, 0.01, 0.001), number=100),
]
tittles = ["Bisection", "Secant", "Modified Secant"]
collection_error_exp = []
for idx, a in enumerate(a1):
    coef, Sy, Syx, r2, r, y_hat = exp_regression(x, y, a)
    print(f"\n{5*'='}Exponential Regression {tittles[idx]}{5*'='}")
    print(f"Coef = {coef}")
    print(f"Standard Deviation = {Sy:.7f}")
    print(f"Error = {Syx:.7f}")
    print(f"R^2 = {r2:.5%}")
    print(f"R = {r:.5%}")
    print(f"Execution time (100 iterations): {(execution_t[idx]):.4f}s")
    collection_error_exp.append(Syx)
    # save_fig(x, y, x, y_hat, f"Exponential Regression {tittles[idx]}")

collection_error_exp = np.array(collection_error_exp)
collection_error_exp = np.mean(collection_error_exp)
collection_error.append(collection_error_exp)

# --------------------------------FIND BEST METHOD-----------------------------------
print(f"\n{5*'='}ERROR COMPARASION{5*'='}")

tittles = ["LINEAR", "POLYNOMIAL", "EXPONENTIAL"]
for idx, error in enumerate(collection_error):
    print(f"{tittles[idx]}: {error:.7f}")
print("\nbest method is: ", tittles[np.argmin(collection_error)])


# --------------------------------INTERPOLATION--------------------------------
# Lagrange Interpolation
def lagrange_interpolating(X, Y, order, new_X_array):
    interpolated_values = []

    for new_X in new_X_array:
        for i in range(len(X)):
            if X[i] >= new_X:
                idx_new = i
                break
        else:
            idx_new = len(X)

        n = order + 1

        subs_idx_lower = max(0, idx_new - math.ceil(n / 2))
        subs_idx_upper = min(len(X), idx_new + math.floor(n / 2))

        interpolated_value = 0
        X_subset = X[subs_idx_lower:subs_idx_upper]
        Y_subset = Y[subs_idx_lower:subs_idx_upper]

        for i in range(len(X_subset)):
            L_i = 1
            for j in range(len(X_subset)):
                if i != j:
                    L_i *= (new_X - X_subset[j]) / (X_subset[i] - X_subset[j])

            interpolated_value += Y_subset[i] * L_i

        interpolated_values.append(interpolated_value)

    return np.array(interpolated_values)


# Some Interpolations Method
print(f"\n{30*'='}INTERPOLATED DATA REGRESSION{30*'='}")
print(f"{5*'='}LAGRANGE METHOD{5*'='}")
print("[Check pdf for details]\n")
print(f"{5*'='}SPLINE METHOD{5*'='}")
print("[Check pdf for details]\n")

lagrange_order = 3  # You can change this value
new_x = np.arange(0, 99.5, 0.5)  # You can change this value
new_y = lagrange_interpolating(x, y, 3, new_x)

# Execute Polynomial Regression - Interpolated
powers = [2, 3, 4, 5]
for power in powers:
    x_, y_ = pol_matrix(new_x, new_y, power + 1)
    gj_matrix = np.column_stack((x_, y_))
    list_func = [
        lu_decomposition(x_, y_),
        gauss_jordan(gj_matrix),
        gauss_seidel(x_, y_),
        jacobi_iteration(x_, y_),
    ]

    tittles = ["LU Decomposition", "Gauss-Jordan", "Gauss-Seidel", "Jacobi Iteration"]

    results = []
    print(
        f"\n{30*'='}POLYNOMIAL REGRESSION {power}th POWER - LAGRANGE {lagrange_order}th ORDER {30*'='}"
    )
    for func in list_func:
        results.append(func[0])

    for idx, result in enumerate(results):
        tittle = tittles[idx]
        coef, Sy, Syx, r2, r, y_hat = polynomial_regression(new_x, new_y, result, power)
        print(f"\n{5*'='}Polynomial Regression {tittle}{5*'='}")
        print(f"Coef = {coef}")
        print(f"Standard Deviation = {Sy:.4f}")
        print(f"Error = {Syx:.4f}")
        print(f"R^2 = {r2:.2%}")
        print(f"R = {r:.2%}")
        # save_fig(
        #     x,
        #     y,
        #     new_x,
        #     y_hat,
        #     f"Interpolated Polynomial Regression {power}th Power - {tittle}",
        # )
