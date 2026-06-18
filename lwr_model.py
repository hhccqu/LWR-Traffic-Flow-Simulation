"""
LWR 交通流模型核心模块
======================
实现 Greenshields 速度-密度关系、通量函数、特征速度，
以及 Rankine-Hugoniot 条件和 Godunov 通量的 Riemann 求解器。

理论：
  - 守恒形式: ∂ρ/∂t + ∂q(ρ)/∂x = 0
  - q(ρ) = ρ * v(ρ)
  - Greenshields: v(ρ) = v_max * (1 - ρ/ρ_max)
  - 特征速度: λ(ρ) = q'(ρ) = v_max * (1 - 2ρ/ρ_max)
"""

import numpy as np
from numpy.typing import NDArray


class LWRModel:
    """Lighthill-Whitham-Richards 宏观交通流模型。

    Parameters
    ----------
    v_max : float
        自由流速度 (m/s)
    rho_max : float
        堵塞密度 (veh/m)，转换为 veh/km 需 ×1000
    """

    def __init__(self, v_max: float = 33.33, rho_max: float = 0.150):
        # rho_max 默认 0.150 veh/m = 150 veh/km
        # v_max 默认 33.33 m/s = 120 km/h
        self.v_max = v_max
        self.rho_max = rho_max
        # 临界密度（通量最大处）：对于 Greenshields，rho_c = rho_max / 2
        self.rho_c = rho_max / 2.0
        # 最大通量（道路通行能力）
        self.q_max = self.flux(self.rho_c)

    # ---- 核心物理函数 ----

    def velocity(self, rho: NDArray[np.float64]) -> NDArray[np.float64]:
        """Greenshields 线性速度-密度关系。

        v(ρ) = v_max * (1 - ρ/ρ_max),  clipp至 [0, v_max]
        """
        rho = np.asarray(rho, dtype=np.float64)
        v = self.v_max * (1.0 - rho / self.rho_max)
        return np.clip(v, 0.0, self.v_max)

    def flux(self, rho: NDArray[np.float64]) -> NDArray[np.float64]:
        """交通通量 q(ρ) = ρ * v(ρ)。

        对于 Greenshields: q(ρ) = v_max * ρ * (1 - ρ/ρ_max)
        这是关于 ρ_c = ρ_max/2 对称的二次函数。
        """
        rho = np.asarray(rho, dtype=np.float64)
        return self.v_max * rho * (1.0 - rho / self.rho_max)

    def characteristic_speed(self, rho: NDArray[np.float64]) -> NDArray[np.float64]:
        """特征速度 λ(ρ) = q'(ρ) = v_max * (1 - 2ρ/ρ_max)。

        注意：λ(ρ) 可以是负值！当 ρ > ρ_max/2 时，波向上游传播。
        """
        rho = np.asarray(rho, dtype=np.float64)
        return self.v_max * (1.0 - 2.0 * rho / self.rho_max)

    def max_char_speed(self) -> float:
        """最大特征速度的绝对值，用于计算 CFL 条件的 dt。"""
        return self.v_max

    # ---- Riemann 问题与激波/稀疏波 ----

    def shock_speed(self, rho_L: float, rho_R: float) -> float:
        """Rankine-Hugoniot 激波速度。

        s = (q(ρ_R) - q(ρ_L)) / (ρ_R - ρ_L)
        """
        if abs(rho_R - rho_L) < 1e-14:
            return self.characteristic_speed(rho_L)
        return (self.flux(rho_R) - self.flux(rho_L)) / (rho_R - rho_L)

    def rarefaction_head_tail_speed(self, rho_L: float, rho_R: float) -> tuple:
        """计算稀疏波的头部速度和尾部速度。

        对于稀疏波（ρ_left > ρ_right），稀疏波扇连接两个常状态。
        - 头部（Head）：进入低密度区域的波前，速度 = λ(ρ_lower)
        - 尾部（Tail）：从高密度区域开始扩展，速度 = λ(ρ_higher)

        注意：头部速度 > 尾部速度（扇向外张开）。

        Returns
        -------
        (head_speed, tail_speed) : 头部向前传播速度，尾部向后传播速度
        """
        rho_lower = min(rho_L, rho_R)
        rho_higher = max(rho_L, rho_R)
        head_speed = self.characteristic_speed(rho_lower)  # 进入低密度区，向前快
        tail_speed = self.characteristic_speed(rho_higher)  # 从高密度区，向后（可能为负）
        return head_speed, tail_speed

    # ---- Godunov 通量（Riemann 问题的精确解）----

    def godunov_flux(self, rho_L: NDArray[np.float64],
                     rho_R: NDArray[np.float64]) -> NDArray[np.float64]:
        """Godunov 数值通量：求解局部 Riemann 问题。

        对于 Greenshields 凹通量函数：

        情况 A: ρ_L ≤ ρ_R（稀疏波）
            ─ 两个密度都在通量曲线的同一侧或跨临界但左侧密度低
            ─ Godunov 通量 = min(q(ρ_L), q(ρ_R))

        情况 B: ρ_L > ρ_R（激波）
            ─ 需要检查是否跨越临界密度 ρ_c
            ─ 如果 ρ_c 在 [ρ_R, ρ_L] 之间：Godunov 通量 = q(ρ_c) = q_max
            ─ 否则：Godunov 通量 = max(q(ρ_L), q(ρ_R))

        Parameters
        ----------
        rho_L : array, 左侧密度
        rho_R : array, 右侧密度

        Returns
        -------
        array, Godunov 数值通量
        """
        rho_L = np.asarray(rho_L, dtype=np.float64)
        rho_R = np.asarray(rho_R, dtype=np.float64)
        scalar_input = rho_L.ndim == 0

        rho_L = np.atleast_1d(rho_L)
        rho_R = np.atleast_1d(rho_R)

        qL = self.flux(rho_L)
        qR = self.flux(rho_R)

        result = np.zeros_like(rho_L)

        # 情况 A: ρ_L <= ρ_R — 稀疏波
        mask_rarefaction = rho_L <= rho_R
        result[mask_rarefaction] = np.minimum(
            qL[mask_rarefaction], qR[mask_rarefaction]
        )

        # 情况 B: ρ_L > ρ_R — 激波
        mask_shock = ~mask_rarefaction
        # 检查临界密度是否在区间内
        rho_c = self.rho_c
        mask_cross_critical = mask_shock & (
            (rho_L > rho_c) & (rho_R < rho_c)
        )
        result[mask_cross_critical] = self.q_max
        # 不跨临界的激波
        mask_non_cross = mask_shock & ~mask_cross_critical
        result[mask_non_cross] = np.maximum(
            qL[mask_non_cross], qR[mask_non_cross]
        )

        return result[0] if scalar_input else result

    # ---- 参数摘要 ----

    def summary(self) -> dict:
        """返回模型参数摘要字典。"""
        return {
            "v_max (m/s)": self.v_max,
            "v_max (km/h)": self.v_max * 3.6,
            "rho_max (veh/m)": self.rho_max,
            "rho_max (veh/km)": self.rho_max * 1000,
            "rho_c (veh/m)": self.rho_c,
            "rho_c (veh/km)": self.rho_c * 1000,
            "q_max (veh/s)": self.q_max,
            "q_max (veh/h)": self.q_max * 3600,
        }

    def print_summary(self):
        """打印模型参数摘要。"""
        print("=" * 55)
        print("  LWR 交通流模型参数 (Greenshields)")
        print("=" * 55)
        for k, v in self.summary().items():
            print(f"  {k:<25s}: {v:.4g}")
        print("=" * 55)
