import numpy as np
import matplotlib.pyplot as plt

# 参数设置
tau = 0.02  # 时间步长
x_start, x_end = -1, 1  # 空间域范围
t_end = 1  # 时间结束点
C = 1  # Courant数，用于满足稳定性条件
# 稳定性条件： tau/h <= C = 1


# 根据CFL条件选择合适的空间步长h

h = tau / C

# 计算空间和时间的网格点数
x_points = int((x_end - x_start) / h) + 1
t_points = int(t_end / tau) + 1

# 初始化网格
x = np.linspace(x_start, x_end, x_points)
t = np.linspace(0, t_end, t_points)
u = np.zeros((t_points, x_points))
u_real = np.zeros((t_points, x_points))

# 应用初始条件
u[0, :] = (x + 1) * np.exp(-x / 2)
u_real[0, :] = u[0, :]

# 使用迎风法进行迭代求解
for n in range(0, t_points - 1):
    for j in range(1, x_points):
        u[n + 1, j] = u[n, j] - (tau / h) * (u[n, j] - u[n, j - 1])
        # 求解解析解
        if t[n + 1] <= x[j] + 1:
            u_real[n + 1, j] = (x[j] - t[n + 1] + 1) * np.exp(-(x[j] - t[n + 1]) / 2)

# 计算相对误差
relative_error = np.sqrt(np.sum((u - u_real)**2)) / np.sqrt(np.sum(u_real**2))
print("Relative error:", relative_error)

# 使用matplotlib绘制数值解的等高线图
plt.figure(num=1, figsize=(8, 6))
plt.contourf(x, t, u, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Numerical Solution with Upwind Scheme')
plt.xlabel('Space x')
plt.ylabel('Time t')
plt.show()


# 使用matplotlib绘制解析解的等高线图
plt.figure(num=2, figsize=(8, 6))
plt.contourf(x, t, u_real, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Analytical Solution with Upwind Scheme')
plt.xlabel('Space x')
plt.ylabel('Time t')
plt.show()
