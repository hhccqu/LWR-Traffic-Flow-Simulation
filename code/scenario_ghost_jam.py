"""
场景定义: 幽灵堵车 (摘自 scenarios.py)
初始条件: 均匀流 + 高斯扰动 → 扰动放大形成激波向后传播
"""

import numpy as np


def create_ghost_jam_ic(rho_max=0.150, L=2000.0, Nx=400):
    """构造幽灵堵车的初始条件。

    基础密度 = 0.30 * rho_max (轻度拥堵)
    扰动 = 0.05 * rho_max * exp(-(x-500)^2 / (2*30^2))

    Returns:
        x: 空间网格坐标 (m)
        rho0: 初始密度分布 (veh/m)
    """
    x = np.linspace(0, L, Nx)

    # 基础均匀流 (0.30 * rho_max = 45 veh/km，自由流状态)
    rho0 = np.full(Nx, 0.30 * rho_max)

    # 叠加高斯扰动 (幅度 5% rho_max，约 7.5 veh/km)
    x0, sigma = 500.0, 30.0
    perturbation = 0.05 * rho_max * np.exp(
        -0.5 * ((x - x0) / sigma) ** 2
    )

    # 合并并裁剪到物理范围 [0.01*rho_max, 0.99*rho_max]
    rho0 = np.clip(rho0 + perturbation, 0.01 * rho_max, 0.99 * rho_max)

    return x, rho0


def create_traffic_light_ic(rho_max=0.150, L=2000.0, Nx=400,
                            light_pos=800.0):
    """构造红绿灯启动波的初始条件 (Riemann 问题)。

    左侧 (x < light_pos): rho = 0.85 * rho_max (红灯排队)
    右侧 (x > light_pos): rho = 0.05 * rho_max (前方空旷)

    这是经典的 Riemann 问题: rho_L >> rho_R → 稀疏波解
    """
    x = np.linspace(0, L, Nx)
    rho0 = np.where(x < light_pos, 0.85 * rho_max, 0.05 * rho_max)

    # 在停车线附近平滑过渡 (避免数值振荡)
    dx = x[1] - x[0]
    smooth_width = 3 * dx
    mask = np.abs(x - light_pos) < smooth_width
    if np.any(mask):
        t_norm = (x[mask] - light_pos + smooth_width) / (2 * smooth_width)
        rho0[mask] = ((1 - t_norm) * 0.85 * rho_max
                      + t_norm * 0.05 * rho_max)

    return x, rho0
