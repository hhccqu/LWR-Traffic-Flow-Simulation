"""
误差分析模块
===========
收敛性分析、波速提取、质量守恒验证、CFL稳定性验证、数据表格生成。
"""

import os
import time
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from lwr_model import LWRModel
from solvers import SOLVERS, SOLVER_INFO

TABLE_DIR = os.path.join(os.path.dirname(__file__), "output", "tables")


def _ensure_table_dir():
    os.makedirs(TABLE_DIR, exist_ok=True)


def save_table(df: pd.DataFrame, filename: str, caption: str = "",
               label: str = ""):
    """保存 DataFrame 为 CSV 和 LaTeX table 格式。"""
    _ensure_table_dir()
    # CSV
    df.to_csv(os.path.join(TABLE_DIR, f"{filename}.csv"), index=False,
              encoding="utf-8-sig")
    # LaTeX
    with open(os.path.join(TABLE_DIR, f"{filename}.tex"), "w",
              encoding="utf-8") as f:
        f.write(df.to_latex(index=False, caption=caption, label=label,
                            escape=False, float_format="%.4g"))
    print(f"  Table saved: {filename}.csv / .tex")


# ============================================================
# 1. L1 误差计算
# ============================================================

def compute_l1_error(rho_num: NDArray[np.float64],
                     rho_ref: NDArray[np.float64]) -> float:
    """计算 L1 范数误差。

    L1_error = (1/N) * Σ|ρ_num - ρ_ref|
    """
    return np.mean(np.abs(rho_num - rho_ref))


def compute_l1_error_total(rho_num: NDArray[np.float64],
                           rho_ref: NDArray[np.float64],
                           dx: float) -> float:
    """计算 L1 误差（连续范数近似）。

    L1_error ≈ Σ_i |ρ_i - ρ_ref_i| * Δx / L
    """
    return np.sum(np.abs(rho_num - rho_ref)) * dx / (len(rho_num) * dx)


# ============================================================
# 2. 收敛性分析
# ============================================================

def convergence_analysis(model: LWRModel, x_coarse: NDArray[np.float64],
                         rho0_func, t_end: float = 120.0,
                         dx_values: list = None,
                         scheme_names: list = None,
                         cfl: float = 0.8) -> dict:
    """自收敛分析：对每种格式，用 2 倍分辨率作为自身参考解。

    对每种格式，在 dx 网格上求解，与该格式在 dx/2 网格上的解比较。
    这样可以独立评估每种格式的收敛阶，不受跨格式差异影响。

    Parameters
    ----------
    model : LWRModel
    x_coarse : 粗网格坐标（用于获取域长 L）
    rho0_func : callable(Nx) -> rho0 array
    t_end : 终止时间
    dx_values : 要测试的 Δx 列表（不含最密参考 dx/2）
    scheme_names : 要测试的格式名列表
    cfl : CFL 数

    Returns
    -------
    results : dict {scheme_name: [err_dx1, err_dx2, ...]}
    """
    if dx_values is None:
        dx_values = [20.0, 10.0, 5.0, 2.5]
    if scheme_names is None:
        scheme_names = ["Godunov", "Lax-Friedrichs", "Upwind", "MacCormack"]

    L = x_coarse[-1]

    print("\n" + "=" * 60)
    print("  Convergence Analysis (Self-Convergence)")
    print("=" * 60)
    print(f"  Each scheme compared against itself at dx/2")
    print(f"  Test dx values: {dx_values}")
    print(f"  Schemes: {scheme_names}")
    print(f"  t_end = {t_end}s, CFL = {cfl}")

    results = {}

    for scheme_name in scheme_names:
        solver = SOLVERS[scheme_name]
        errors = []
        print(f"\n  {scheme_name}:")
        for dx in dx_values:
            # 测试网格
            Nx_test = int(L / dx) + 1
            x_test = np.linspace(0, L, Nx_test)
            rho0_test = rho0_func(Nx_test)
            rho_test = solver(model, x_test, t_end, rho0_test,
                              cfl=cfl, store_all=False, progress=False)

            # 参考网格：dx/2
            dx_ref = dx / 2.0
            Nx_ref = int(L / dx_ref) + 1
            x_ref = np.linspace(0, L, Nx_ref)
            rho0_ref = rho0_func(Nx_ref)
            rho_ref = solver(model, x_ref, t_end, rho0_ref,
                             cfl=cfl, store_all=False, progress=False)

            # 每隔一个点采样参考解（对应 dx 网格点位置）
            # 由于 x_ref[::2] 不一定精确等于 x_test，使用插值
            from scipy.interpolate import interp1d
            interp = interp1d(x_ref, rho_ref, kind="linear",
                              bounds_error=False, fill_value="extrapolate")
            rho_ref_on_test = interp(x_test)

            err = compute_l1_error(rho_test, rho_ref_on_test)
            # 避免零误差（当测试与参考一致时）
            err = max(err, 1e-16)
            errors.append(err)
            print(f"    dx={dx:5.1f}m (ref dx/2={dx_ref:4.1f}m) → L1_error={err:.6e}")

        results[scheme_name] = errors

    return {"dx_values": dx_values, "errors": results}


# ============================================================
# 3. 波速提取
# ============================================================

def extract_shock_speed(t_array: NDArray[np.float64],
                        x: NDArray[np.float64],
                        rho_history: NDArray[np.float64],
                        model: LWRModel) -> tuple:
    """从数值解中提取激波速度。

    方法：追踪 ρ = (ρ_max + ρ_min)/2 的等值线位置随时间变化。
    用线性回归拟合位置-时间关系得到波速。

    Returns
    -------
    numerical_speed : float, 数值波速 (m/s)
    theoretical_speed : float, 理论波速 (m/s)
    r_squared : float, 线性拟合的 R²
    positions : array, 各时刻的激波位置
    times : array, 对应的时间点
    """
    # 找出初始剖面中的两个主要密度水平
    rho_init = rho_history[0]
    unique_levels = np.sort(np.unique(np.round(rho_init * 1000)))
    if len(unique_levels) >= 2:
        rho_L = unique_levels[-2] / 1000  # 较高密度
        rho_R = unique_levels[-1] / 1000  # 较低密度
        # 更稳健的方式
        rho_L_val = np.percentile(rho_init, 90)
        rho_R_val = np.percentile(rho_init, 10)
    else:
        rho_L_val = np.max(rho_init)
        rho_R_val = np.min(rho_init)

    threshold = (rho_L_val + rho_R_val) / 2

    positions = []
    times_list = []

    for i, t_val in enumerate(t_array):
        rho = rho_history[i]
        # 找到跨越阈值的最大梯度位置
        grad = np.abs(np.diff(rho))
        if np.max(grad) > 1e-8:
            idx_max = np.argmax(grad)
            # 子像素精度：在最大梯度附近用加权平均
            if 0 < idx_max < len(rho) - 1:
                pos = x[idx_max]
                positions.append(pos)
                times_list.append(t_val)

    positions = np.array(positions)
    times_arr = np.array(times_list)

    if len(times_arr) < 3:
        return 0.0, 0.0, 0.0, positions, times_arr

    # 线性回归
    coeffs = np.polyfit(times_arr, positions, 1)
    numerical_speed = coeffs[0]

    # 理论激波速度 (Rankine-Hugoniot)
    theoretical_speed = model.shock_speed(rho_L_val, rho_R_val)

    # R²
    pos_pred = np.polyval(coeffs, times_arr)
    ss_res = np.sum((positions - pos_pred) ** 2)
    ss_tot = np.sum((positions - np.mean(positions)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return numerical_speed, theoretical_speed, r_squared, positions, times_arr


def extract_rarefaction_speeds(t_array: NDArray[np.float64],
                               x: NDArray[np.float64],
                               rho_history: NDArray[np.float64],
                               model: LWRModel,
                               rho_L: float, rho_R: float,
                               discontinuity_x: float = 800.0) -> dict:
    """从数值解中提取稀疏波头部和尾部速度。

    对于初始间断（ρ_L 在左，ρ_R 在右，ρ_L > ρ_R → 稀疏波）：
    - 头部向前传播（速度 = λ(ρ_R)），追踪 ρ ≈ 1.1*ρ_R 的等值线
    - 尾部向后传播（速度 = λ(ρ_L)），追踪 ρ ≈ 0.9*ρ_L 的等值线

    追踪策略：从间断位置出发，在每个时刻找到最靠近间断位置的密度等值线。
    这避免了被域内其他区域的相同密度值误导。
    """
    head_speed_theory, tail_speed_theory = \
        model.rarefaction_head_tail_speed(rho_L, rho_R)

    # 追踪阈值：头部追踪略高于 ρ_R 的密度，尾部追踪略低于 ρ_L 的密度
    head_rho_target = rho_R * 1.5 if rho_R < 0.1 * model.rho_max else rho_R * 1.1
    tail_rho_target = rho_L * 0.9

    head_positions, head_times = [], []
    tail_positions, tail_times = [], []

    for i, t_val in enumerate(t_array):
        rho = rho_history[i]

        # 头部追踪：从间断位置向右找，密度下降到 head_rho_target 的位置
        light_idx = np.argmin(np.abs(x - discontinuity_x))
        # 向右搜索
        for j in range(light_idx, len(x)):
            if rho[j] <= head_rho_target:
                head_positions.append(x[j])
                head_times.append(t_val)
                break

        # 尾部追踪：从间断位置向左找，密度下降到 tail_rho_target 的位置
        for j in range(light_idx, -1, -1):
            if rho[j] >= tail_rho_target:
                tail_positions.append(x[j])
                tail_times.append(t_val)
                break

    # 线性拟合
    head_speed_num, tail_speed_num = 0.0, 0.0

    if len(head_times) >= 3:
        head_coeffs = np.polyfit(head_times, head_positions, 1)
        head_speed_num = head_coeffs[0]

    if len(tail_times) >= 3:
        tail_coeffs = np.polyfit(tail_times, tail_positions, 1)
        tail_speed_num = tail_coeffs[0]

    return {
        "head_speed_num": head_speed_num,
        "head_speed_theory": head_speed_theory,
        "tail_speed_num": tail_speed_num,
        "tail_speed_theory": tail_speed_theory,
    }


# ============================================================
# 4. 质量守恒验证
# ============================================================

def check_mass_conservation(t_array: NDArray[np.float64],
                            rho_history: NDArray[np.float64],
                            dx: float) -> dict:
    """验证质量守恒：总质量随时间的变化。

    Mass(t) = Σ ρ_i(t) * Δx
    """
    mass = np.sum(rho_history, axis=1) * dx
    mass0 = mass[0]
    drift = mass - mass0
    drift_percent = drift / mass0 * 100

    return {
        "initial_mass": mass0,
        "final_mass": mass[-1],
        "max_drift_percent": np.max(np.abs(drift_percent)),
        "final_drift_percent": drift_percent[-1],
        "passed": np.max(np.abs(drift_percent)) < 1.0,
    }


# ============================================================
# 5. 表格生成
# ============================================================

def table_model_parameters(model: LWRModel):
    """表1: 模型参数汇总。"""
    data = [
        ["自由流速度 v_max", f"{model.v_max:.2f} m/s", f"{model.v_max*3.6:.1f} km/h"],
        ["堵塞密度 ρ_max", f"{model.rho_max:.4f} veh/m", f"{model.rho_max*1000:.0f} veh/km"],
        ["临界密度 ρ_c", f"{model.rho_c:.4f} veh/m", f"{model.rho_c*1000:.0f} veh/km"],
        ["最大通量 q_max", f"{model.q_max:.4f} veh/s", f"{model.q_max*3600:.0f} veh/h"],
        ["道路长度 L", "2000 m", "—"],
        ["默认网格数 Nx", "400", "—"],
        ["默认网格间距 Δx", "5 m", "—"],
        ["默认 CFL 数", "0.8", "—"],
        ["密度-速度模型", "Greenshields 线性模型", "v = v_max(1-ρ/ρ_max)"],
    ]
    df = pd.DataFrame(data, columns=["Parameter", "Value (SI)", "Value (practical)"])
    save_table(df, "table_model_params",
               caption="LWR模型参数汇总 (Greenshields)",
               label="tab:params")


def table_scheme_info():
    """表2: 数值格式对比。"""
    data = []
    for name, info in SOLVER_INFO.items():
        data.append([
            name,
            f"{info['order']}阶",
            info["desc"],
        ])
    df = pd.DataFrame(data,
                      columns=["Scheme", "Order", "Description"])
    save_table(df, "table_scheme_info",
               caption="数值格式对比",
               label="tab:schemes")


def table_wave_speed(shock_results: dict, rarefaction_results: dict = None):
    """表3/4: 波速对比。"""
    rows = []

    if shock_results:
        rows.append([
            "激波速度",
            f"{shock_results['num_speed']:.3f}",
            f"{shock_results['theory_speed']:.3f}",
            f"{abs(shock_results['num_speed'] - shock_results['theory_speed']):.4f}",
            f"{shock_results['r_squared']:.4f}",
        ])

    if rarefaction_results:
        r = rarefaction_results
        rows.append([
            "稀疏波头部速度",
            f"{r['head_speed_num']:.3f}",
            f"{r['head_speed_theory']:.3f}",
            f"{abs(r['head_speed_num'] - r['head_speed_theory']):.4f}",
            "—",
        ])
        rows.append([
            "稀疏波尾部速度",
            f"{r['tail_speed_num']:.3f}",
            f"{r['tail_speed_theory']:.3f}",
            f"{abs(r['tail_speed_num'] - r['tail_speed_theory']):.4f}",
            "—",
        ])

    df = pd.DataFrame(rows,
                      columns=["Wave Type", "Numerical (m/s)", "Theoretical (m/s)",
                               "Abs Error (m/s)", "R²"])
    save_table(df, "table_wave_speed",
               caption="波速：数值解 vs 理论值对比",
               label="tab:waves")


def table_convergence(dx_values: list, errors: dict):
    """表5: 收敛性数据。"""
    data = {"Δx (m)": dx_values}
    for name, err_list in errors.items():
        data[name] = [f"{e:.6f}" for e in err_list]

    df = pd.DataFrame(data)
    save_table(df, "table_convergence",
               caption="不同网格精度下的L1误差",
               label="tab:conv")


def table_cfl_results(cfl_results: dict):
    """表6: CFL 条件验证。"""
    data = []
    for cfl, info in cfl_results.items():
        data.append([
            cfl,
            info.get("stable", "Unknown"),
            f"{info.get('max_val', 0):.4f}",
            f"{info.get('min_val', 0):.4f}",
        ])
    df = pd.DataFrame(data, columns=["CFL", "Stable", "Max ρ", "Min ρ"])
    save_table(df, "table_cfl",
               caption="不同CFL数下的数值稳定性",
               label="tab:cfl")


def table_timing(timing_results: dict):
    """表7: 计算时间对比。"""
    data = []
    for name, t in timing_results.items():
        data.append([name, f"{t:.4f}"])
    df = pd.DataFrame(data, columns=["Scheme", "Wall Time (s)"])
    save_table(df, "table_timing",
               caption="各数值格式计算时间对比",
               label="tab:timing")


def table_congestion(congestion_data: dict):
    """表8: 拥堵长度 vs 汇入率。"""
    data = []
    for rate, length in congestion_data.items():
        data.append([f"{rate:.3f} veh/s", f"{length:.1f} m"])
    df = pd.DataFrame(data, columns=["Ramp Flow Rate", "Max Congestion Length"])
    save_table(df, "table_congestion",
               caption="不同匝道汇入率下的最大拥堵长度",
               label="tab:congestion")
