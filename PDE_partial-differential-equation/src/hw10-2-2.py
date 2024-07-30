import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt

def u_true(x, y):
    return x**2 * y**2 + np.sin(x * y)

def f(x, y):
    return (x**2 + y**2) * (np.sin(x * y) - 2)

def solve_poisson_sparse(N):
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)
    y = np.linspace(0, 1, N+1)
    
    # Set up the sparse matrix
    main_diagonal = -4 * np.ones((N+1)**2)
    side_diagonal = np.ones((N+1)**2 - 1)  # Offsets of -1 and +1
    upper_lower_diagonal = np.ones((N+1)**2 - (N+1))  # Offsets of -(N+1) and (N+1)
    
    diagonals = [main_diagonal, side_diagonal, side_diagonal, upper_lower_diagonal, upper_lower_diagonal]
    offsets = [0, -1, 1, -(N+1), N+1]
    A = sp.diags(diagonals, offsets, shape=((N+1)**2, (N+1)**2), format="csr")
    
    # Boundary conditions and right-hand side
    b = np.zeros((N+1)**2)
    X, Y = np.meshgrid(x, y, indexing='ij')
    F = f(X, Y).flatten()
    
    for i in range(N+1):
        for j in range(N+1):
            index = i * (N+1) + j
            if i == 0 or i == N or j == 0 or j == N:
                A[index, :] = 0
                A[index, index] = 1
                b[index] = u_true(x[i], y[j])
            else:
                b[index] = -h**2 * F[index]
    
    # Solve the linear system
    U = sp.linalg.spsolve(A, b).reshape((N+1, N+1))
    
    return X, Y, U

N = 50  # Grid size
X, Y, U_numerical = solve_poisson_sparse(N)

# Exact solution for comparison
U_exact = u_true(X, Y)

# Error calculation
error = np.sqrt(np.sum((U_numerical - U_exact)**2) / np.sum(U_exact**2))

# Plotting
plt.figure(figsize=(10, 5))
plt.contourf(X, Y, U_numerical, levels=50, cmap='viridis')
plt.colorbar()
plt.title(f'Numerical solution with relative error: {error:.2e}')
plt.show()
