import numpy as np
import matplotlib.pyplot as plt

N = 50
h = 1.0 / N
x = np.linspace(0, 1, N+1)
y = np.linspace(0, 1, N+1)
u = np.zeros((N+1, N+1))

# 系数矩阵 A 和右侧向量 b 的初始化
A = np.zeros(((N+1)**2, (N+1)**2))
b = np.zeros((N+1)**2)

def idx(i, j):
    return i * (N+1) + j

# 填充内部节点
for i in range(1, N):
    for j in range(1, N):
        index = idx(i, j)
        A[index, index] = -4
        A[index, idx(i-1, j)] = 1
        A[index, idx(i+1, j)] = 1
        A[index, idx(i, j-1)] = 1
        A[index, idx(i, j+1)] = 1
        b[index] = -16 * h**2

# 应用边界条件
# x = 1, Dirichlet, 包含了两个角点（1, 1）和（1, 0）
for j in range(N+1):
    i = N
    index = idx(i, j)
    A[index, index] = 1
    b[index] = 0
    
# x = 0, Neumann
for j in range(1, N):
    i = 0
    index = idx(i, j)
    A[index, index] = -4
    A[index, idx(i, j-1)] = 1
    A[index, idx(i, j+1)] = 1
    b[index] = -16 * h**2

# y = 0, Neumann
for i in range(1, N):
    j = 0
    index = idx(i, j)
    A[index, index] = -4
    A[index, idx(i-1, j)] = 1
    A[index, idx(i+1, j)] = 1
    b[index] = -16 * h**2

# y = 1, Neumann-like mixed boundary condition
for i in range(1, N):
    j = N
    index = idx(i, j)
    A[index, index] = -(4+2*h)
    A[index, idx(i-1, j)] = 1
    A[index, idx(i+1, j)] = 1
    A[index, idx(i, j-1)] = 2
    b[index] = -16 * h**2

# 余下的两个角点(0, 0)和(0, 1)
# (0,0)
i,j = 0,0
index = idx(i, j)
A[index, index] = -4
A[index, idx(i+1, j)] = 2
A[index, idx(i, j+1)] = 2
b[index] = -16 * h**2
# (0,1)
i,j = 0,N
index = idx(i, j)
A[index, index] = -(4+2*h)
A[index, idx(i+1, j)] = 2
A[index, idx(i, j-1)] = 2
b[index] = -16 * h**2


# 解线性系统
u_flat = np.linalg.solve(A, b)
u = u_flat.reshape((N+1, N+1))

# 可视化
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, u, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Solution of $-\Delta u = 16$')
plt.show()
