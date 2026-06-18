"""
仿真场景定义模块
===============
定义 5 个交通流场景的初始条件和边界条件。

场景1: 幽灵堵车 — 均匀流+高斯扰动 → 激波形成
场景2: 红绿灯   — 红灯堆积+绿灯启动 → 稀疏波
场景3: 慢车效应 — 慢车导致后方拥挤 → 运动激波
场景4: 周期红绿灯 — 周期性激波+稀疏波交替
场景5: 匝道汇入 — 匝道车流汇入主线 → 拥堵传播
"""

import numpy as np
from numpy.typing import NDArray
from lwr_model import LWRModel


# ============================================================
# 通用工具
# ============================================================

def make_grid(L: float = 2000.0, Nx: int = 400) -> NDArray[np.float64]:
    """生成均匀空间网格。

    Parameters
    ----------
    L : float, 道路长度 (m)
    Nx : int, 网格点数

    Returns
    -------
    x : array, 空间坐标
    dx : float, 网格间距
    """
    x = np.linspace(0, L, Nx)
    return x


def uniform_density(Nx: int, model: LWRModel,
                    fraction: float = 0.3) -> NDArray[np.float64]:
    """均匀密度分布。"""
    return np.full(Nx, fraction * model.rho_max)


def gaussian_bump(x: NDArray[np.float64], x0: float, sigma: float,
                  amplitude: float) -> NDArray[np.float64]:
    """高斯扰动。"""
    return amplitude * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


# ============================================================
# 场景 1：幽灵堵车 (Ghost Jam)
# ============================================================

def scenario_ghost_jam(model: LWRModel, L: float = 2000.0,
                       Nx: int = 400, t_end: float = 180.0) -> dict:
    """幽灵堵车：均匀流上叠加小扰动，观察激波形成。

    初始条件：
      ρ(x,0) = 0.3·ρ_max + δρ(x)
      δρ(x) = 0.05·ρ_max · exp(-(x-500)²/(2·30²))

    边界条件：零阶外推（开放边界）
    """
    x = make_grid(L, Nx)
    dx = x[1] - x[0]

    rho_base = uniform_density(Nx, model, fraction=0.30)
    perturbation = gaussian_bump(x, x0=500, sigma=30,
                                 amplitude=0.05 * model.rho_max)
    rho0 = np.clip(rho_base + perturbation, 0.01 * model.rho_max, 0.99 * model.rho_max)

    return {
        "name": "幽灵堵车 (Ghost Jam)",
        "tag": "ghost_jam",
        "x": x, "dx": dx, "Nx": Nx, "L": L,
        "t_end": t_end,
        "rho0": rho0,
        "description": "均匀流(ρ=0.3ρmax)上叠加高斯扰动，扰动放大形成向后传播的激波",
        "key_phenomena": ["激波形成", "特征线汇聚", "扰动增长"],
    }


# ============================================================
# 场景 2：红绿灯 (Traffic Light)
# ============================================================

def scenario_traffic_light(model: LWRModel, L: float = 2000.0,
                           Nx: int = 400, t_end: float = 120.0,
                           light_position: float = 800.0) -> dict:
    """红绿灯变绿：红灯时车辆在停车线后方堆积，绿灯亮起后稀疏波传播。

    初始条件 (t=0 红灯刚变绿)：
      x < light_position (上游): ρ = 0.85·ρ_max（红灯排队）
      x > light_position (下游): ρ = 0.05·ρ_max（空旷）

    这是一个经典的 Riemann 问题：ρ_L >> ρ_R。
    解：稀疏波向后（上游）传播。
    """
    x = make_grid(L, Nx)
    dx = x[1] - x[0]

    rho0 = np.where(
        x < light_position,
        0.85 * model.rho_max,  # 排队的密集车流
        0.05 * model.rho_max   # 前方空旷
    )
    # 在停车线附近加一点平滑过渡
    transition_width = 3 * dx
    smooth_mask = np.abs(x - light_position) < transition_width
    if np.any(smooth_mask):
        t_norm = (x[smooth_mask] - light_position + transition_width) / (2 * transition_width)
        rho0[smooth_mask] = (
            (1 - t_norm) * 0.85 * model.rho_max + t_norm * 0.05 * model.rho_max
        )

    return {
        "name": "红绿灯启动波 (Traffic Light)",
        "tag": "traffic_light",
        "x": x, "dx": dx, "Nx": Nx, "L": L,
        "t_end": t_end,
        "rho0": rho0,
        "light_position": light_position,
        "description": "红灯时车辆在停车线后堆积(ρ=0.85ρmax)，绿灯后稀疏波向上游传播",
        "key_phenomena": ["稀疏波", "Riemann问题", "启动波波速"],
    }


# ============================================================
# 场景 3：慢车效应 (Slow Vehicle)
# ============================================================

def scenario_slow_vehicle(model: LWRModel, L: float = 2000.0,
                          Nx: int = 400, t_end: float = 200.0,
                          slow_start: float = 1500.0,
                          slow_speed_fraction: float = 0.25) -> dict:
    """慢车效应：模拟一辆慢速车辆在道路上行驶，后方车辆堆积。

    初始条件：
      全路段均匀密度 ρ = 0.25·ρ_max，车辆速度 v = 0.75·v_max
      在 x = slow_start 处有一辆"慢车"——通过局部高密度模拟

    用移动的"高密度区"模拟慢车，高密度区以慢车速度移动。
    由于模型是欧拉描述，这里通过初始条件中的局部高密度来表示慢车的阻碍效应。
    """
    x = make_grid(L, Nx)
    dx = x[1] - x[0]

    # 基础均匀流
    rho0 = np.full(Nx, 0.25 * model.rho_max)

    # 慢车位置用一个小的高密度脉冲表示（模拟慢车的瓶颈效应）
    slow_width = 30.0  # 慢车影响区域宽度 (m)
    slow_amplitude = 0.35 * model.rho_max
    rho0 += gaussian_bump(x, slow_start, slow_width / 3, slow_amplitude)

    # 在慢车之前降低密度（慢车前方车少）
    ahead_mask = x > slow_start + slow_width
    rho0[ahead_mask] *= 0.5

    # 在慢车后方密度稍高
    behind_mask = (x < slow_start) & (x > slow_start - 200)
    rho0[behind_mask] += 0.08 * model.rho_max

    rho0 = np.clip(rho0, 0.01 * model.rho_max, 0.98 * model.rho_max)

    return {
        "name": "慢车效应 (Slow Vehicle)",
        "tag": "slow_vehicle",
        "x": x, "dx": dx, "Nx": Nx, "L": L,
        "t_end": t_end,
        "rho0": rho0,
        "slow_start": slow_start,
        "description": "一辆慢速车阻挡后方交通，后方形成激波向后传播",
        "key_phenomena": ["运动激波", "瓶颈效应", "单车轨迹"],
    }


# ============================================================
# 场景 4：周期红绿灯 (Periodic Traffic Light)
# ============================================================

def scenario_periodic_light(model: LWRModel, L: float = 2000.0,
                            Nx: int = 400, t_end: float = 300.0,
                            light_position: float = 800.0,
                            period: float = 120.0,
                            green_fraction: float = 0.5) -> dict:
    """周期红绿灯：红灯周期性地在停车线处阻断交通。

    这个场景需要时变边界条件——在停车线处周期性地修改通量。
    实现方式：在 solver 中检测到停车线位置时，根据当前时间修改局部通量。

    初始条件：均匀流
    """
    x = make_grid(L, Nx)
    dx = x[1] - x[0]

    rho0 = np.full(Nx, 0.25 * model.rho_max)

    return {
        "name": "周期红绿灯 (Periodic Light)",
        "tag": "periodic_light",
        "x": x, "dx": dx, "Nx": Nx, "L": L,
        "t_end": t_end,
        "rho0": rho0,
        "light_position": light_position,
        "period": period,
        "green_fraction": green_fraction,
        "description": "周期红绿灯产生周期性的激波-稀疏波交替结构",
        "key_phenomena": ["周期定常结构", "激波-稀疏波交替", "排队长度振荡"],
    }


# ============================================================
# 场景 5：匝道汇入 (On-ramp Merge)
# ============================================================

def scenario_onramp(model: LWRModel, L: float = 2000.0,
                    Nx: int = 400, t_end: float = 250.0,
                    ramp_position: float = 1200.0,
                    ramp_flow_rate: float = 0.04) -> dict:
    """匝道汇入：在 ramp_position 处有车辆持续汇入主线。

    这需要在守恒方程中加入源项：
      ∂ρ/∂t + ∂q/∂x = S(x, t)

    其中 S(x, t) 是匝道汇入的源项，在汇入点附近有值。

    初始条件：主线均匀流
    """
    x = make_grid(L, Nx)
    dx = x[1] - x[0]

    rho0 = np.full(Nx, 0.45 * model.rho_max)  # 主线较拥挤

    return {
        "name": "匝道汇入 (On-ramp Merge)",
        "tag": "onramp",
        "x": x, "dx": dx, "Nx": Nx, "L": L,
        "t_end": t_end,
        "rho0": rho0,
        "ramp_position": ramp_position,
        "ramp_flow_rate": ramp_flow_rate,
        "description": "匝道车流汇入主线，汇入处形成瓶颈→拥堵向上游传播",
        "key_phenomena": ["带源项守恒律", "瓶颈拥堵", "汇入率-拥堵长度关系"],
    }


# ============================================================
# 场景注册表
# ============================================================

SCENARIOS = {
    "ghost_jam": scenario_ghost_jam,
    "traffic_light": scenario_traffic_light,
    "slow_vehicle": scenario_slow_vehicle,
    "periodic_light": scenario_periodic_light,
    "onramp": scenario_onramp,
}
