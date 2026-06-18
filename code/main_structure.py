"""
主程序结构 (摘自 main.py)
展示完整的仿真流程: 初始化 → 场景求解 → 可视化 → 误差分析 → 表格输出
"""

import numpy as np
from lwr_model_core import flux, shock_speed
from solvers_core import godunov_step
from scenario_ghost_jam import create_ghost_jam_ic, create_traffic_light_ic


# ============ 1. 模型参数初始化 ============
v_max = 33.33       # 自由流速度 (m/s) = 120 km/h
rho_max = 0.150     # 堵塞密度 (veh/m) = 150 veh/km
CFL = 0.8           # CFL 数
t_end = 180.0       # 仿真时长 (s)
Nx = 400            # 网格点数

# ============ 2. 构造初始条件 ============
x, rho0 = create_ghost_jam_ic(rho_max=rho_max, Nx=Nx)
dx = x[1] - x[0]

# ============ 3. 时间推进 (Godunov 格式) ============
dt = CFL * dx / v_max       # CFL 条件约束
Nt = int(np.ceil(t_end / dt))
dt = t_end / Nt             # 微调使精确覆盖 t_end

rho = rho0.copy()
t_history, rho_history = [0.0], [rho0.copy()]
store_every = max(1, Nt // 500)

for n in range(Nt):
    rho = godunov_step(rho, dx, dt, v_max, rho_max)
    if (n + 1) % store_every == 0:
        t_history.append((n + 1) * dt)
        rho_history.append(rho.copy())

t_array = np.array(t_history)
rho_array = np.array(rho_history)
print(f"Simulation done: {Nt} steps, dt={dt:.4f}s")

# ============ 4. 结果分析 ============
# 激波速度验证 (Rankine-Hugoniot 条件)
rho_L_val = np.percentile(rho0, 90)   # 扰动区密度
rho_R_val = np.percentile(rho0, 10)   # 基础密度
s_theory = shock_speed(rho_L_val, rho_R_val, v_max, rho_max)
print(f"Theoretical shock speed (R-H): {s_theory:.2f} m/s")

# 质量守恒验证
mass0 = np.sum(rho0) * dx
mass_final = np.sum(rho) * dx
drift = (mass_final - mass0) / mass0 * 100
print(f"Mass drift: {drift:.3f}%  {'PASS' if abs(drift) < 1.0 else 'WARN'}")
