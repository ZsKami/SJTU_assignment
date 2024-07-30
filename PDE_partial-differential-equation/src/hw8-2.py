import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# 参数设置
L, T = 1.0, 1.0  # x和t的范围
Nx, Nt = 50, 100  # 空间和时间的步数

# 初始和边界条件
u0 = lambda x: x**2 * np.sin(np.pi * x) # 初始条件
g0 = lambda t: 0 * t  # u(0, t) = 0，边界条件
g1 = lambda t: np.cos(t) * np.sin(np.pi)  # u(1, t) = 0, 因为sin(pi) = 0
f = lambda t, x: np.cos(t) * (np.pi**2 * x**2 * np.sin(np.pi * x) - 2 * np.sin(np.pi * x) - 4 * np.pi * x * np.cos(np.pi * x)) \
    - np.sin(t)* x**2 * np.sin(np.pi * x)

# Crank-Nicolson方法
def crank_nicolson(L, T, Nx, Nt):
    # 参数设置
    alpha = 1
    dx, dt = L / Nx, T / Nt  # 空间和时间的步长
    x = np.linspace(0, L, Nx+1)  # 空间网格
    t = np.linspace(0, T, Nt+1)  # 时间网格    

    # 初始化u矩阵
    u = np.zeros((Nt+1, Nx+1))
    u[0, :] = u0(x)
    u[:, 0] = g0(t)
    u[:, -1] = g1(t)

    # 创建系数矩阵
    r = alpha*dt/(2*dx**2)
    A_main_diag = (1 + 2*r) * np.ones(Nx - 1)
    A_off_diag = -r * np.ones(Nx - 2)
    A = diags([A_main_diag, A_off_diag, A_off_diag], [0, -1, 1], format="csr")
    B_main_diag = (1 - 2*r) * np.ones(Nx - 1)
    B_off_diag = r * np.ones(Nx - 2)
    B = diags([B_main_diag, B_off_diag, B_off_diag], [0, -1, 1], format="csr")
    
    for n in range(0, Nt):
        b = B.dot(u[n, 1 : -1]) + dt / 2 * (f(t[n], x[1 : -1]) + f(t[n + 1], x[1 : -1]))
        b[0] += r* (u[n, 0] + u[n + 1, 0])
        b[-1] += r* (u[n, -1] + u[n + 1, -1])
        u[n + 1, 1 : -1] = spsolve(A, b)
    
    return x, t, u

x, t, u = crank_nicolson(L, T, Nx, Nt)

# 计算误差
u_exact = np.array([[(np.cos(ti) * xi**2 * np.sin(np.pi * xi)) for xi in x] for ti in t])
error = np.sqrt(np.sum((u - u_exact)**2)) / np.sqrt(np.sum(u_exact**2))
print(f'Relative error: {error}')

# 使用matplotlib绘制数值解的等高线图
plt.figure(num=1, figsize=(8, 6))
plt.contourf(x, t, u, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Numerical Crank-Nicolson Solution')
plt.xlabel('Space x')
plt.ylabel('Time t')
plt.show()


# 使用matplotlib绘制解析解的等高线图
plt.figure(num=2, figsize=(8, 6))
plt.contourf(x, t, u_exact, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Analytical Solution')
plt.xlabel('Space x')
plt.ylabel('Time t')
plt.show()

# 使用matplotlib绘制解析解的等高线图
plt.figure(num=3, figsize=(8, 6))
plt.contourf(x, t, u - u_exact, levels=50, cmap='viridis')
plt.colorbar(format='%.2e')
plt.title('Error with Crank-Nicolson format')
plt.xlabel('Space x')
plt.ylabel('Time t')
plt.show()
