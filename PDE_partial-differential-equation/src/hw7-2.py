import numpy as np
import matplotlib.pyplot as plt

# 定义初始和边界条件
err = 1e-10
def initial_condition(x):
    return np.where(np.abs(x) <= (0.5 + err), 1, 0)

def boundary_condition(t, x):
    return 0

def analytical_solution(t,x):
    return np.where(np.abs(x - t) <= (0.5 + err), 1, 0)

# Lax-Wendroff方法的实现
def lax_wendroff(u, h, tau):
    u_next = np.copy(u)  # 创建u的副本
    # 计算中间步
    u_mid = 0.5 * (u[1:] + u[:-1]) - (tau / (2*h)) * (u[1:] - u[:-1])
    # 更新u_next中间部分，忽略边界
    u_next[1:-1] = u[1:-1] - (tau / h) * (u_mid[1:] - u_mid[:-1])
    return u_next


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

# 应用初始条件
u[0, :] = initial_condition(x)
u_real = np.copy(u)

# 时间演化
for n in range(0, t_points - 1):
    u[n + 1, :] = lax_wendroff(u[n, :], h, tau)
    # 应用边界条件
    u[n + 1, 0] = boundary_condition(t[n + 1], x_start)
    u[n + 1, -1] = boundary_condition(t[n + 1], x_end)
    # 求解解析解
    u_real[n + 1,:] = analytical_solution(t[n + 1],x)

# 计算相对误差
relative_error = np.sqrt(np.sum((u - u_real)**2)) / np.sqrt(np.sum(u_real**2))
print("Relative error:", relative_error)

# 使用matplotlib绘制数值解的等高线图
plt.figure(num=1, figsize=(8, 6))
plt.contourf(x, t, u, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Numerical Lax-wendroff Solution')
plt.xlabel('Space x')
plt.ylabel('Time t')
plt.show()


# 使用matplotlib绘制解析解的等高线图
plt.figure(num=2, figsize=(8, 6))
plt.contourf(x, t, u_real, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Analytical Solution')
plt.xlabel('Space x')
plt.ylabel('Time t')
plt.show()
