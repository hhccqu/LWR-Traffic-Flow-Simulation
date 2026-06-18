"""
数值求解器模块
=============
实现 4 种有限差分格式求解 LWR 方程：

1. Godunov (1阶)     — 基于 Riemann 精确解的 Godunov 格式
2. Lax-Friedrichs (1阶) — 经典的一阶中心差分格式
3. Upwind (1阶)      — 迎风格式，按特征方向选择单侧差分
4. MacCormack (2阶)  — 预测-校正两步格式

所有求解器接口统一：
    solve(model, x, t_end, rho0, cfl, bc_type, progress)
    → (t_array, rho_history)  或  (rho_final,)
"""

import numpy as np
from numpy.typing import NDArray
from lwr_model import LWRModel


def _compute_dt(model: LWRModel, dx: float, cfl: float) -> float:
    """由 CFL 条件计算时间步长。

    Δt = CFL * Δx / max|λ(ρ)|
    对于 Greenshields: max|λ| = v_max
    """
    return cfl * dx / model.max_char_speed()


# ============================================================
# 1. Godunov 格式（一阶）
# ============================================================

def godunov_step(model: LWRModel, rho: NDArray[np.float64],
                 dx: float, dt: float) -> NDArray[np.float64]:
    """Godunov 格式单步推进。

    原理：
      1. 将每个网格交界面视为局部 Riemann 问题
      2. 用 godunov_flux 计算数值通量
      3. 守恒型更新: ρ_i^{n+1} = ρ_i^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
    """
    Nx = len(rho)
    rho_new = rho.copy()

    # 计算各交界面的 Godunov 通量 (i+1/2 for i = 0, ..., Nx-2)
    # F_{i+1/2} = godunov_flux(ρ_i, ρ_{i+1})
    flux_right = model.godunov_flux(rho[:-1], rho[1:])  # length Nx-1

    # 守恒更新（内部网格点）
    rho_new[1:-1] = rho[1:-1] - (dt / dx) * (flux_right[1:] - flux_right[:-1])

    # 边界处理：零阶外推 (∂ρ/∂x = 0)
    rho_new[0] = rho_new[1]
    rho_new[-1] = rho_new[-2]

    return np.clip(rho_new, 0.0, model.rho_max)


def solve_godunov(model: LWRModel, x: NDArray[np.float64],
                  t_end: float, rho0: NDArray[np.float64],
                  cfl: float = 0.8,
                  store_all: bool = True,
                  progress: bool = True) -> tuple:
    """用 Godunov 格式求解 LWR 方程。

    Returns
    -------
    (t_array, rho_history)  if store_all=True
    rho_final               if store_all=False
    """
    dx = x[1] - x[0]
    dt = _compute_dt(model, dx, cfl)
    Nt = int(np.ceil(t_end / dt))
    dt = t_end / Nt  # 调整 dt 精确覆盖 t_end

    if progress:
        print(f"  Godunov: dx={dx:.3f}, dt={dt:.4f}, Nt={Nt}, CFL={cfl}")

    rho = rho0.copy()

    if store_all:
        # 预估存储量
        store_every = max(1, Nt // 500)
        t_list, rho_list = [0.0], [rho0.copy()]

    for n in range(Nt):
        rho = godunov_step(model, rho, dx, dt)

        if store_all and (n + 1) % store_every == 0:
            t_list.append((n + 1) * dt)
            rho_list.append(rho.copy())

    if store_all:
        # 确保最终时刻存入
        if t_list[-1] < t_end - 1e-10:
            t_list.append(t_end)
            rho_list.append(rho.copy())
        return np.array(t_list), np.array(rho_list)
    else:
        return rho


# ============================================================
# 2. Lax-Friedrichs 格式（一阶）
# ============================================================

def lax_friedrichs_step(model: LWRModel, rho: NDArray[np.float64],
                        dx: float, dt: float) -> NDArray[np.float64]:
    """Lax-Friedrichs 格式单步推进。

    ρ_i^{n+1} = 0.5*(ρ_{i+1}^n + ρ_{i-1}^n)
                 - 0.5*(dt/dx)*(q(ρ_{i+1}^n) - q(ρ_{i-1}^n))

    特点：无条件稳定（CFL≤1），但数值耗散大。
    """
    rho_new = rho.copy()
    q = model.flux(rho)
    coeff = 0.5 * dt / dx

    rho_new[1:-1] = (
        0.5 * (rho[2:] + rho[:-2])
        - coeff * (q[2:] - q[:-2])
    )
    rho_new[0] = rho_new[1]
    rho_new[-1] = rho_new[-2]

    return np.clip(rho_new, 0.0, model.rho_max)


def solve_lax_friedrichs(model: LWRModel, x: NDArray[np.float64],
                         t_end: float, rho0: NDArray[np.float64],
                         cfl: float = 0.8,
                         store_all: bool = True,
                         progress: bool = True) -> tuple:
    """用 Lax-Friedrichs 格式求解 LWR 方程。"""
    dx = x[1] - x[0]
    dt = _compute_dt(model, dx, cfl)
    Nt = int(np.ceil(t_end / dt))
    dt = t_end / Nt

    if progress:
        print(f"  Lax-Friedrichs: dx={dx:.3f}, dt={dt:.4f}, Nt={Nt}, CFL={cfl}")

    rho = rho0.copy()

    if store_all:
        store_every = max(1, Nt // 500)
        t_list, rho_list = [0.0], [rho0.copy()]

    for n in range(Nt):
        rho = lax_friedrichs_step(model, rho, dx, dt)

        if store_all and (n + 1) % store_every == 0:
            t_list.append((n + 1) * dt)
            rho_list.append(rho.copy())

    if store_all:
        if t_list[-1] < t_end - 1e-10:
            t_list.append(t_end)
            rho_list.append(rho.copy())
        return np.array(t_list), np.array(rho_list)
    else:
        return rho


# ============================================================
# 3. Upwind 格式（一阶迎风）
# ============================================================

def upwind_step(model: LWRModel, rho: NDArray[np.float64],
                dx: float, dt: float) -> NDArray[np.float64]:
    """迎风格式单步推进。

    根据特征速度 λ(ρ) = q'(ρ) 的符号决定差分方向：
      λ > 0 → 信息从左侧来 → 用左侧通量
      λ < 0 → 信息从右侧来 → 用右侧通量

    数值通量（守恒形式）：
      如果 λ_{i+1/2} ≥ 0: F_{i+1/2} = q(ρ_i)
      如果 λ_{i+1/2} < 0: F_{i+1/2} = q(ρ_{i+1})
    """
    rho_new = rho.copy()
    Nx = len(rho)

    # 计算交界面特征速度（取平均密度）
    rho_interface = 0.5 * (rho[:-1] + rho[1:])
    lam = model.characteristic_speed(rho_interface)  # length Nx-1

    # 数值通量
    flux_interface = np.where(
        lam >= 0,
        model.flux(rho[:-1]),   # 左侧通量
        model.flux(rho[1:])     # 右侧通量
    )

    rho_new[1:-1] = rho[1:-1] - (dt / dx) * (flux_interface[1:] - flux_interface[:-1])
    rho_new[0] = rho_new[1]
    rho_new[-1] = rho_new[-2]

    return np.clip(rho_new, 0.0, model.rho_max)


def solve_upwind(model: LWRModel, x: NDArray[np.float64],
                 t_end: float, rho0: NDArray[np.float64],
                 cfl: float = 0.8,
                 store_all: bool = True,
                 progress: bool = True) -> tuple:
    """用迎风格式求解 LWR 方程。"""
    dx = x[1] - x[0]
    dt = _compute_dt(model, dx, cfl)
    Nt = int(np.ceil(t_end / dt))
    dt = t_end / Nt

    if progress:
        print(f"  Upwind: dx={dx:.3f}, dt={dt:.4f}, Nt={Nt}, CFL={cfl}")

    rho = rho0.copy()

    if store_all:
        store_every = max(1, Nt // 500)
        t_list, rho_list = [0.0], [rho0.copy()]

    for n in range(Nt):
        rho = upwind_step(model, rho, dx, dt)

        if store_all and (n + 1) % store_every == 0:
            t_list.append((n + 1) * dt)
            rho_list.append(rho.copy())

    if store_all:
        if t_list[-1] < t_end - 1e-10:
            t_list.append(t_end)
            rho_list.append(rho.copy())
        return np.array(t_list), np.array(rho_list)
    else:
        return rho


# ============================================================
# 4. MacCormack 格式（二阶预测-校正）
# ============================================================

def maccormack_step(model: LWRModel, rho: NDArray[np.float64],
                    dx: float, dt: float) -> NDArray[np.float64]:
    """MacCormack 预测-校正格式单步推进。

    预测步（向前差分）：
      ρ*_i = ρ_i^n - (dt/dx) * (q_{i+1}^n - q_i^n)

    校正步（向后差分）：
      ρ_i^{n+1} = 0.5 * [ρ_i^n + ρ*_i - (dt/dx) * (q*_i - q*_{i-1})]

    特点：二阶精度，但激波附近有 Gibbs 振荡。
    """
    Nx = len(rho)
    q = model.flux(rho)

    # ---- 预测步：前向差分 ----
    rho_star = rho.copy()
    rho_star[1:-1] = rho[1:-1] - (dt / dx) * (q[2:] - q[1:-1])
    rho_star = np.clip(rho_star, 0.0, model.rho_max)

    # 预测步的边界
    rho_star[0] = rho_star[1]
    rho_star[-1] = rho_star[-2]

    # ---- 校正步：后向差分 ----
    q_star = model.flux(rho_star)
    rho_new = rho.copy()
    rho_new[1:-1] = 0.5 * (
        rho[1:-1] + rho_star[1:-1]
        - (dt / dx) * (q_star[1:-1] - q_star[:-2])
    )
    rho_new = np.clip(rho_new, 0.0, model.rho_max)
    # 时间层对称交替的 FWD-BWD / BWD-FWD 可以抑制振荡，这里简化为 FWD-BWD

    rho_new[0] = rho_new[1]
    rho_new[-1] = rho_new[-2]

    return rho_new


def solve_maccormack(model: LWRModel, x: NDArray[np.float64],
                     t_end: float, rho0: NDArray[np.float64],
                     cfl: float = 0.8,
                     store_all: bool = True,
                     progress: bool = True) -> tuple:
    """用 MacCormack 格式求解 LWR 方程。"""
    dx = x[1] - x[0]
    dt = _compute_dt(model, dx, cfl)
    Nt = int(np.ceil(t_end / dt))
    dt = t_end / Nt

    if progress:
        print(f"  MacCormack: dx={dx:.3f}, dt={dt:.4f}, Nt={Nt}, CFL={cfl}")

    rho = rho0.copy()

    if store_all:
        store_every = max(1, Nt // 500)
        t_list, rho_list = [0.0], [rho0.copy()]

    for n in range(Nt):
        rho = maccormack_step(model, rho, dx, dt)

        if store_all and (n + 1) % store_every == 0:
            t_list.append((n + 1) * dt)
            rho_list.append(rho.copy())

    if store_all:
        if t_list[-1] < t_end - 1e-10:
            t_list.append(t_end)
            rho_list.append(rho.copy())
        return np.array(t_list), np.array(rho_list)
    else:
        return rho


# ============================================================
# 求解器注册表
# ============================================================

SOLVERS = {
    "Godunov": solve_godunov,
    "Lax-Friedrichs": solve_lax_friedrichs,
    "Upwind": solve_upwind,
    "MacCormack": solve_maccormack,
}

SOLVER_INFO = {
    "Godunov": {"order": 1, "desc": "基于Riemann精确解，物理意义强，激波捕捉准确"},
    "Lax-Friedrichs": {"order": 1, "desc": "经典中心差分，无条件稳定，数值耗散大"},
    "Upwind": {"order": 1, "desc": "迎风格式，按特征方向差分，稳定且耗散适中"},
    "MacCormack": {"order": 2, "desc": "预测-校正两步格式，二阶精度，激波附近有振荡"},
}


# ============================================================
# 5. 周期红绿灯专用求解器
# ============================================================

def solve_periodic_light(model: LWRModel, x: NDArray[np.float64],
                         t_end: float, rho0: NDArray[np.float64],
                         light_position: float = 800.0,
                         period: float = 120.0,
                         green_fraction: float = 0.5,
                         cfl: float = 0.8,
                         progress: bool = True) -> tuple:
    """带周期红绿灯的 Godunov 求解器。

    在红绿灯位置处，红灯时通量为 0（车辆不能通过），绿灯时正常通行。
    实现方式：在红绿灯所在的网格交界面，根据当前时间决定通量。
    """
    dx = x[1] - x[0]
    dt = _compute_dt(model, dx, cfl)
    Nt = int(np.ceil(t_end / dt))
    dt = t_end / Nt

    # 找到红绿灯所在的网格交界面索引
    light_interface = np.argmin(np.abs(x - light_position))
    # 确保 light_interface 在有效范围
    light_interface = max(1, min(len(x) - 2, light_interface))

    if progress:
        print(f"  Periodic Light Godunov: dx={dx:.3f}, dt={dt:.4f}, Nt={Nt}")
        print(f"    Light at x≈{x[light_interface]:.1f}m (interface {light_interface})")
        print(f"    Period={period}s, Green={green_fraction*100:.0f}%")

    rho = rho0.copy()

    store_every = max(1, Nt // 500)
    t_list, rho_list = [0.0], [rho0.copy()]

    for n in range(Nt):
        t_now = (n + 1) * dt

        # 判断红绿灯状态
        phase = (t_now % period) / period
        is_green = phase < green_fraction

        rho_new = rho.copy()

        # 正常 Godunov 通量（所有交界面）
        flux_right = model.godunov_flux(rho[:-1], rho[1:])

        # 红灯时在红绿灯交界处阻塞通量
        if not is_green:
            # 红绿灯处交界面通量为 0
            if 0 <= light_interface < len(flux_right):
                flux_right[light_interface] = 0.0
            # 同时也阻塞相邻交界面（确保停车线处完全阻塞）
            if light_interface + 1 < len(flux_right):
                flux_right[light_interface + 1] = 0.0

        # 守恒更新
        rho_new[1:-1] = rho[1:-1] - (dt / dx) * (flux_right[1:] - flux_right[:-1])
        rho_new[0] = rho_new[1]
        rho_new[-1] = rho_new[-2]
        rho = np.clip(rho_new, 0.0, model.rho_max)

        if (n + 1) % store_every == 0:
            t_list.append(t_now)
            rho_list.append(rho.copy())

    if t_list[-1] < t_end - 1e-10:
        t_list.append(t_end)
        rho_list.append(rho.copy())

    return np.array(t_list), np.array(rho_list)


# ============================================================
# 6. 匝道汇入专用求解器
# ============================================================

def solve_onramp(model: LWRModel, x: NDArray[np.float64],
                 t_end: float, rho0: NDArray[np.float64],
                 ramp_position: float = 1200.0,
                 ramp_flow_rate: float = 0.04,
                 ramp_width: float = 60.0,
                 cfl: float = 0.8,
                 progress: bool = True) -> tuple:
    """带匝道源项的 Godunov 求解器。

    守恒方程：∂ρ/∂t + ∂q/∂x = S(x, t)

    其中 S(x, t) 是匝道汇入的源项。离散化：
      ρ_i^{n+1} = ρ_i^n - (dt/dx)(F_{i+1/2} - F_{i-1/2}) + dt * S_i

    源项 S_i 在匝道位置附近为非零，形状为高斯函数。
    """
    dx = x[1] - x[0]
    dt = _compute_dt(model, dx, cfl)
    Nt = int(np.ceil(t_end / dt))
    dt = t_end / Nt

    # 构建空间源项分布（高斯形）
    source_spatial = np.exp(-0.5 * ((x - ramp_position) / (ramp_width / 3)) ** 2)
    # 归一化使积分 = ramp_flow_rate
    source_spatial = source_spatial / (np.sum(source_spatial) * dx) * ramp_flow_rate

    if progress:
        print(f"  On-ramp Godunov: dx={dx:.3f}, dt={dt:.4f}, Nt={Nt}")
        print(f"    Ramp at x≈{ramp_position:.0f}m, width≈{ramp_width:.0f}m")
        print(f"    Flow rate={ramp_flow_rate:.3f} veh/s")

    rho = rho0.copy()

    store_every = max(1, Nt // 500)
    t_list, rho_list = [0.0], [rho0.copy()]

    for n in range(Nt):
        rho_new = rho.copy()

        # Godunov 通量
        flux_right = model.godunov_flux(rho[:-1], rho[1:])
        rho_new[1:-1] = rho[1:-1] - (dt / dx) * (flux_right[1:] - flux_right[:-1])

        # 源项贡献
        rho_new += dt * source_spatial

        rho_new[0] = rho_new[1]
        rho_new[-1] = rho_new[-2]
        rho = np.clip(rho_new, 0.0, model.rho_max)

        if (n + 1) % store_every == 0:
            t_list.append((n + 1) * dt)
            rho_list.append(rho.copy())

    if t_list[-1] < t_end - 1e-10:
        t_list.append(t_end)
        rho_list.append(rho.copy())

    return np.array(t_list), np.array(rho_list)
