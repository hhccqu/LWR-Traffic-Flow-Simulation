"""
LWR 模型核心函数 (摘自 lwr_model.py)
Greenshields 速度-密度关系、通量函数、特征速度、Godunov 通量。

理论背景:
  q(rho) = v_max * rho * (1 - rho/rho_max)    — 凹二次通量函数
  lambda(rho) = q'(rho) = v_max * (1 - 2*rho/rho_max)  — 特征速度
"""

import numpy as np


def velocity(rho, v_max=33.33, rho_max=0.150):
    """Greenshields 线性速度-密度关系: v(rho) = v_max * (1 - rho/rho_max)"""
    rho = np.asarray(rho, dtype=np.float64)
    return np.clip(v_max * (1.0 - rho / rho_max), 0.0, v_max)


def flux(rho, v_max=33.33, rho_max=0.150):
    """交通通量 q(rho) = rho * v(rho)，凹二次函数"""
    rho = np.asarray(rho, dtype=np.float64)
    return v_max * rho * (1.0 - rho / rho_max)


def characteristic_speed(rho, v_max=33.33, rho_max=0.150):
    """特征速度 lambda(rho) = q'(rho) = v_max * (1 - 2*rho/rho_max)
    注意: rho > rho_max/2 时 lambda < 0，波向上游传播！"""
    rho = np.asarray(rho, dtype=np.float64)
    return v_max * (1.0 - 2.0 * rho / rho_max)


def shock_speed(rho_L, rho_R, v_max=33.33, rho_max=0.150):
    """Rankine-Hugoniot 激波速度: s = [q(rho_R) - q(rho_L)] / (rho_R - rho_L)"""
    if abs(rho_R - rho_L) < 1e-14:
        return characteristic_speed(rho_L, v_max, rho_max)
    return (flux(rho_R, v_max, rho_max) - flux(rho_L, v_max, rho_max)) / (rho_R - rho_L)


def godunov_flux(rho_L, rho_R, v_max=33.33, rho_max=0.150):
    """Godunov 数值通量 —— 求解局部 Riemann 问题的精确解。

    对于 Greenshields 凹通量函数，存在三种情况:

    情况 A: rho_L <= rho_R (稀疏波)
        — 特征线发散，通量取 min(q_L, q_R)

    情况 B: rho_L > rho_R (激波)
        B1: 跨临界激波 (rho_L > rho_c > rho_R)
            — 通量 = q_max (受限于通行能力)
        B2: 非跨临界激波
            — 通量 = max(q_L, q_R)

    返回: 数值通量 F_{i+1/2}
    """
    rho_c = rho_max / 2.0
    q_max = flux(rho_c, v_max, rho_max)
    qL = flux(rho_L, v_max, rho_max)
    qR = flux(rho_R, v_max, rho_max)

    if rho_L <= rho_R:
        # 稀疏波
        return min(qL, qR)
    else:
        # 激波
        if rho_L > rho_c and rho_R < rho_c:
            return q_max  # 跨临界
        else:
            return max(qL, qR)  # 非跨临界
