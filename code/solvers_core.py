"""
数值求解器核心函数 (摘自 solvers.py)
Godunov、Lax-Friedrichs、Upwind、MacCormack 四种格式的单步推进。

所有格式统一遵循守恒型更新:
    rho_i^{n+1} = rho_i^n - (dt/dx) * (F_{i+1/2} - F_{i-1/2})
"""

import numpy as np
from lwr_model_core import godunov_flux, flux, characteristic_speed


# ---- Godunov 格式 (一阶，Riemann精确解) ----
def godunov_step(rho, dx, dt, v_max=33.33, rho_max=0.150):
    """Godunov 单步推进: 每个交界面求解 Riemann 问题"""
    rho_new = rho.copy()
    # 各交界面的 Godunov 通量 F_{i+1/2}
    flux_right = np.array([
        godunov_flux(rho[i], rho[i+1], v_max, rho_max)
        for i in range(len(rho) - 1)
    ])
    # 守恒更新
    rho_new[1:-1] = rho[1:-1] - (dt/dx) * (flux_right[1:] - flux_right[:-1])
    # 零阶外推边界
    rho_new[0] = rho_new[1]
    rho_new[-1] = rho_new[-2]
    return np.clip(rho_new, 0.0, rho_max)


# ---- Lax-Friedrichs 格式 (一阶，中心+耗散) ----
def lax_friedrichs_step(rho, dx, dt, v_max=33.33, rho_max=0.150):
    """rho_i^{n+1} = 1/2*(rho_{i+1} + rho_{i-1}) - dt/(2dx)*(q_{i+1} - q_{i-1})"""
    rho_new = rho.copy()
    q = flux(rho, v_max, rho_max)
    coeff = 0.5 * dt / dx
    rho_new[1:-1] = (0.5 * (rho[2:] + rho[:-2])
                     - coeff * (q[2:] - q[:-2]))
    rho_new[0] = rho_new[1]
    rho_new[-1] = rho_new[-2]
    return np.clip(rho_new, 0.0, rho_max)


# ---- Upwind 格式 (一阶，按特征方向) ----
def upwind_step(rho, dx, dt, v_max=33.33, rho_max=0.150):
    """根据 lambda(rho) 的正负号选择差分方向"""
    rho_new = rho.copy()
    # 交界面特征速度
    rho_interface = 0.5 * (rho[:-1] + rho[1:])
    lam = characteristic_speed(rho_interface, v_max, rho_max)
    # 迎风通量: lambda >= 0 → 左侧通量; lambda < 0 → 右侧通量
    flux_interface = np.where(
        lam >= 0,
        flux(rho[:-1], v_max, rho_max),
        flux(rho[1:], v_max, rho_max)
    )
    rho_new[1:-1] = rho[1:-1] - (dt/dx) * (flux_interface[1:] - flux_interface[:-1])
    rho_new[0] = rho_new[1]
    rho_new[-1] = rho_new[-2]
    return np.clip(rho_new, 0.0, rho_max)


# ---- MacCormack 格式 (二阶，预测-校正) ----
def maccormack_step(rho, dx, dt, v_max=33.33, rho_max=0.150):
    """预测步(前向差分) + 校正步(后向差分) → 二阶精度"""
    q = flux(rho, v_max, rho_max)

    # 预测步: 前向差分
    rho_star = rho.copy()
    rho_star[1:-1] = rho[1:-1] - (dt/dx) * (q[2:] - q[1:-1])
    rho_star = np.clip(rho_star, 0.0, rho_max)

    # 校正步: 后向差分
    q_star = flux(rho_star, v_max, rho_max)
    rho_new = rho.copy()
    rho_new[1:-1] = 0.5 * (rho[1:-1] + rho_star[1:-1]
                            - (dt/dx) * (q_star[1:-1] - q_star[:-2]))
    rho_new[0] = rho_new[1]
    rho_new[-1] = rho_new[-2]
    return np.clip(rho_new, 0.0, rho_max)
