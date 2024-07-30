import numpy as np
import matplotlib.pyplot as plt

# 参数设置
x_min, y_min, = 0.0, 0.0 
x_max, y_max, T = 1.0, 1.0, 1.0  # x和t的范围
Nx, Ny, Nt = 50, 50, 10000  # 空间和时间的步数

dx, dy, dt = (x_max - x_min) / Nx, (y_max - y_min) / Ny, T / Nt  # 空间和时间的步长
# print(f'dx: {dx}; dy: {dy}; dt:{dt}')
x = np.linspace(x_min, x_max, Nx+1)  # 空间网格
y = np.linspace(y_min, y_max, Ny+1)
t = np.linspace(0, T, Nt+1)  # 时间网格    

# 初始化u矩阵
u = np.zeros((Nt+1, Nx+1, Ny+1))

# 初始和边界条件
u[0, :, :] = np.exp((x[:, None] + y[None, :]) / 2)
u[:, 0, :] = np.exp(y[None, :] / 2 - t[:, None])
u[:, -1, :] = np.exp((y[None, :]+1) / 2 - t[:, None])
u[:, :, 0] = np.exp(x[None, :] / 2 - t[:, None])
u[:, :, -1] = np.exp((x[None, :] + 1) / 2 - t[:, None])
f = -3 / 2 * np.exp((x[None, :, None] + y[None, None, :]) / 2 - t[:, None, None])


r_x, r_y = dt / (dx**2), dt / (dy**2)

for n in range(0, Nt):    
    u[n + 1, 1:-1, 1:-1] = (u[n, 1:-1, 1:-1] + dt * (f[n+1, 1:-1, 1:-1]+f[n, 1:-1, 1:-1])/2 +
        r_x * (u[n,2:,1:-1] - 2 * u[n,1:-1,1:-1] + u[n,:-2,1:-1]) +
        r_y * (u[n,1:-1,2:] - 2 * u[n,1:-1,1:-1] + u[n,1:-1,:-2]) 
        )
        


# 计算误差
u_exact = np.array([[[(np.exp((xi + yi) / 2 - ti))for yi in y] for xi in x] for ti in t])
error = np.sqrt(np.sum((u - u_exact)**2)) / np.sqrt(np.sum(u_exact**2))
print(f'Relative error: {error}')

#设置图片属性
def plot_solution(ax, data, title):
    contour = ax.contourf(x, y, data, levels=50, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax)
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('Space (y)')
    ax.set_title(title)
    cbar.set_label('Solution (u)')
    
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 8})
fig, axs = plt.subplots(3, 2, figsize=(9, 10)) # 

#绘制t=0时数值解和解析解等值线图
plot_solution(axs[0, 0], u[0, :, :], 'Numerical Solution at t=0')
plot_solution(axs[0, 1], u_exact[0, :, :], 'Analytical Solution at t=0')

#绘制t=0.5时数值解和解析解等值线图
plot_solution(axs[1, 0], u[(Nt-1)//2, :, :], 'Numerical Solution at t=0.5')
plot_solution(axs[1, 1], u_exact[(Nt-1)//2, :, :], 'Analytical Solution at t=0.5')

#绘制t=1时数值解和解析解等值线图
plot_solution(axs[2, 0], u[-1, :, :], 'Numerical Solution at t=1')
plot_solution(axs[2, 1], u_exact[-1, :, :], 'Analytical Solution at t=1')

plt.tight_layout()
plt.show()
