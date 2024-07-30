import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def analytical_solution(x):
    return -x**6 / 30 + x / 30

def solve_fem(N):
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)
    K = np.zeros((N-1, N-1))
    f = np.zeros(N-1)
    
    # 构建刚度矩阵和载荷向量
    for i in range(N-1):
        K[i, i] = 2 / h
        if i > 0:
            K[i, i-1] = -1 / h
            K[i-1, i] = -1 / h
        
    # 精确计算载荷向量
    for i in range(1, N):
        f[i-1]= (quad(lambda xi: xi**4 * ((xi - x[i-1]) / h), x[i-1], x[i])[0]
                + quad(lambda xi: xi**4 * ((x[i+1] - xi) / h), x[i], x[i+1])[0])
    
    # 解线性系统
    u = np.linalg.solve(K, f)
    
    # 添加边界条件
    u = np.concatenate([[0], u, [0]])
    
    area = np.sum(h * (u[:-1] + u[1:]) / 2)  # 应用梯形法则求面积

    return x, u, area

# 绘制解和解析解
x_analytical = np.linspace(0, 1, 1000)
u_analytical = analytical_solution(x_analytical)
area_analytical = np.sum((u_analytical[:-1] + u_analytical[1:]) / 2) / 1000
plt.plot(x_analytical, u_analytical, 'k-', label='Analytical Solution')


errors = []
steps = [4, 6, 8, 10]
for N in steps:
    x, u, area = solve_fem(N)
    plt.plot(x, u, label=f'h={1/N:.2f}')
    
    # 计算相对误差
    u_exact = analytical_solution(x)
    error = np.abs(area - area_analytical) / area_analytical
    errors.append(error)

plt.title('Finite Element Method Solutions for Different h')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()

# 输出误差
for h, error in zip(steps, errors):
    print(f'步长 h={1/h:.2f} 的相对误差是 {error:.10f}')
