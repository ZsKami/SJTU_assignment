import numpy as np
import matplotlib.pyplot as plt

def u_true(x, y):
    return x**2 * y**2 + np.sin(x * y)

def f(x, y):
    return (x**2 + y**2) * (np.sin(x * y) - 2)


def solve_poisson(N):
    h = 1 / N
    x = np.linspace(0, 1, N+1)
    y = np.linspace(0, 1, N+1)
    U = np.zeros((N+1, N+1))

    # Boundary conditions
    for i in range(N+1):
        U[i, 0] = u_true(x[i], 0)
        U[i, N] = u_true(x[i], 1)
        U[0, i] = u_true(0, y[i])
        U[N, i] = u_true(1, y[i])
    
    # Filling the grid points
    X, Y = np.meshgrid(x, y, indexing='ij')
    F = f(X, Y)
    
    # Iteration using the finite difference method
    for iteration in range(10000):
        for i in range(1, N):
            for j in range(1, N):
                U[i, j] = 0.25 * (U[i+1, j] + U[i-1, j] + U[i, j+1] + U[i, j-1] - h**2 * F[i, j])

    return X, Y, U

N = 50  # Grid size
X, Y, U_numerical = solve_poisson(N)

# Exact solution
U_exact = u_true(X, Y)

# Error calculation
error = np.sqrt(np.sum((U_numerical - U_exact)**2) / np.sum(U_exact**2))

# Plotting
plt.figure(figsize=(10, 5))
plt.contourf(X, Y, U_numerical, levels=50, cmap='viridis')
plt.colorbar()
plt.title(f'Numerical solution with relative error: {error:.2e}')
plt.show()
