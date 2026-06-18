"""
可视化模块
=========
所有仿真图像输出函数。
统一使用 Matplotlib，保存为 PDF（矢量）+ PNG（预览）。
"""

import os
import warnings
import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from lwr_model import LWRModel

# ============================================================
# 全局绘图设置
# ============================================================

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "figures")

# 字体设置：优先使用有良好 Unicode 覆盖的字体
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
_font_candidates = ["DejaVu Sans", "Arial", "SimHei", "Microsoft YaHei"]
for font in _font_candidates:
    try:
        plt.rcParams["font.sans-serif"] = [font] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig, name: str):
    """保存为 PDF 和 PNG。"""
    _ensure_dir()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
    plt.close(fig)


# ============================================================
# 1. 通量函数与特征速度
# ============================================================

def plot_flux_function(model: LWRModel):
    """绘制 q(ρ) 和 λ(ρ) 曲线，标注临界密度。"""
    rho = np.linspace(0, model.rho_max, 300)
    q = model.flux(rho)
    lam = model.characteristic_speed(rho)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # 通量函数
    ax1.plot(rho * 1000, q * 3600, "b-", linewidth=2)
    ax1.axvline(x=model.rho_c * 1000, color="red", linestyle="--",
                alpha=0.7, label=f"ρc={model.rho_c*1000:.0f} veh/km")
    ax1.scatter([model.rho_c * 1000], [model.q_max * 3600],
                color="red", s=80, zorder=5)
    ax1.set_xlabel("Density ρ (veh/km)")
    ax1.set_ylabel("Flux q(ρ) (veh/h)")
    ax1.set_title("Fundamental Diagram: q(ρ) (Greenshields)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, model.rho_max * 1000)
    ax1.set_ylim(0, model.q_max * 3600 * 1.15)

    # 特征速度
    ax2.plot(rho * 1000, lam * 3.6, "b-", linewidth=2)
    ax2.axvline(x=model.rho_c * 1000, color="red", linestyle="--",
                alpha=0.7, label=f"ρc={model.rho_c*1000:.0f} veh/km")
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.fill_between([0, model.rho_c * 1000], [0, 0], [model.v_max * 3.6, 0],
                     alpha=0.1, color="green", label="λ>0 (forward)")
    ax2.fill_between([model.rho_c * 1000, model.rho_max * 1000],
                     [0, 0], [0, -model.v_max * 3.6],
                     alpha=0.1, color="orange", label="λ<0 (backward)")
    ax2.set_xlabel("Density ρ (veh/km)")
    ax2.set_ylabel("Char. Speed λ(ρ) (km/h)")
    ax2.set_title("Characteristic Speed λ(ρ)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("LWR Model: Flux Function & Characteristic Speed",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "flux_function")


# ============================================================
# 2. x-t 密度热力图（通用）
# ============================================================

def plot_xt_heatmap(t_array: NDArray[np.float64],
                    x: NDArray[np.float64],
                    rho_history: NDArray[np.float64],
                    model: LWRModel,
                    title: str,
                    filename: str,
                    show_char_directions: bool = True,
                    vmin: float = None, vmax: float = None):
    """绘制 x-t 密度热力图。

    Parameters
    ----------
    t_array : (Nt_out,) 输出时间点
    x : (Nx,) 空间坐标
    rho_history : (Nt_out, Nx) 密度历史
    model : LWRModel
    title : 图标题
    filename : 保存文件名（不含扩展名）
    show_char_directions : 是否标注特征方向
    """
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = model.rho_max * 1000

    X, T_grid = np.meshgrid(x, t_array)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.jet.copy()
    im = ax.pcolormesh(X, T_grid, rho_history * 1000,
                        cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, label="Density ρ (veh/km)")
    cbar.ax.yaxis.label.set_size(11)

    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Time t (s)")
    ax.set_title(title, fontweight="bold")

    # 标注临界密度等高线
    cs = ax.contour(X, T_grid, rho_history * 1000,
                    levels=[model.rho_c * 1000],
                    colors="white", linewidths=1.5, linestyles="--", alpha=0.7)
    ax.clabel(cs, fmt="ρc=%.0f", colors="white", fontsize=8)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(t_array[0], t_array[-1])

    plt.tight_layout()
    _save(fig, filename)


# ============================================================
# 3. 特征线图
# ============================================================

def plot_characteristics(model: LWRModel,
                         x: NDArray[np.float64],
                         t_end: float,
                         rho0: NDArray[np.float64],
                         filename: str,
                         n_lines: int = 60):
    """绘制特征线图。

    特征线方程: dx/dt = λ(ρ(x,t))
    此处用初始条件近似计算特征线方向（仅可视化用途）。
    真正特征线需要追踪整条线，这里展示初始方向。
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 选择出发位置（均匀分布）
    x0_array = np.linspace(x[0], x[-1], n_lines)

    for x0 in x0_array:
        idx = np.argmin(np.abs(x - x0))
        rho_val = rho0[idx]
        lam = model.characteristic_speed(rho_val)
        # 绘制特征线 x = x0 + λ*t
        t_line = np.array([0, t_end])
        x_line = x0 + lam * t_line
        # 裁剪到域内
        mask = (x_line >= x[0]) & (x_line <= x[-1]) & (t_line <= t_end)

        if lam > 0:
            color = "green"
            alpha = 0.3
        else:
            color = "orange"
            alpha = 0.3

        if np.sum(mask) >= 2:
            ax.plot(x_line[mask], t_line[mask], color=color, alpha=alpha,
                    linewidth=0.6)

    # 激波形成区域（特征线汇聚处）
    # 找到密度梯度最大的时刻对应的位置
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Time t (s)")
    ax.set_title("Characteristics dx/dt = λ(ρ)  (Green: forward, Orange: backward)",
                 fontweight="bold")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, t_end)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    _save(fig, filename)


# ============================================================
# 4. 密度剖面演化
# ============================================================

def plot_density_profiles(t_array: NDArray[np.float64],
                          x: NDArray[np.float64],
                          rho_history: NDArray[np.float64],
                          model: LWRModel,
                          title: str,
                          filename: str,
                          n_profiles: int = 6):
    """绘制不同时刻的密度剖面曲线。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    Nt = len(t_array)
    indices = np.linspace(0, Nt - 1, n_profiles, dtype=int)

    colors = plt.cm.viridis(np.linspace(0, 0.9, n_profiles))

    for i, (idx, c) in enumerate(zip(indices, colors)):
        t_val = t_array[idx]
        ax.plot(x, rho_history[idx] * 1000, color=c,
                linewidth=1.5, label=f"t = {t_val:.0f} s")

    # 临界密度线
    ax.axhline(y=model.rho_c * 1000, color="red", linestyle="--",
               alpha=0.5, label=f"ρc = {model.rho_c*1000:.0f} veh/km")

    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Density ρ (veh/km)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, filename)


# ============================================================
# 5. 波速对比图
# ============================================================

def plot_wave_speed_comparison(model: LWRModel,
                               t_array: NDArray[np.float64],
                               x: NDArray[np.float64],
                               rho_history: NDArray[np.float64],
                               theoretical_shock_speed: float,
                               theoretical_head_speed: float = None,
                               theoretical_tail_speed: float = None,
                               title: str = "Wave Speed Comparison",
                               filename: str = "wave_speed"):
    """绘制数值波速与理论波速的对比。

    通过追踪密度跳变位置来提取数值波速。
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # 对每个时刻追踪密度跳跃的"中心"位置
    # 中心定义为密度 = (ρ_max + ρ_min)/2 的位置
    threshold = (np.max(rho_history[0]) + np.min(rho_history[0])) / 2
    shock_positions = []
    shock_times = []

    for i, t_val in enumerate(t_array):
        rho = rho_history[i]
        # 找到密度剖面中跨越阈值的最大梯度位置
        grad = np.abs(np.diff(rho))
        idx_max_grad = np.argmax(grad)
        if grad[idx_max_grad] > 1e-6:
            shock_positions.append(x[idx_max_grad])
            shock_times.append(t_val)

    shock_positions = np.array(shock_positions)
    shock_times = np.array(shock_times)

    if len(shock_times) > 2:
        # 数值波速（线性拟合）
        coeffs_num = np.polyfit(shock_times, shock_positions, 1)
        num_speed = coeffs_num[0]

        # 绘制追踪到的激波位置
        ax.scatter(shock_times, shock_positions, s=15, alpha=0.5,
                   label="Detected shock position")
        ax.plot(shock_times, np.polyval(coeffs_num, shock_times),
                "r-", linewidth=2, label=f"Numerical speed = {num_speed:.2f} m/s")

        # 理论激波速度线
        t_fit = np.array([shock_times[0], shock_times[-1]])
        x_ref = shock_positions[0] + theoretical_shock_speed * (t_fit - t_fit[0])
        ax.plot(t_fit, x_ref, "b--", linewidth=2,
                label=f"Theoretical (R-H) = {theoretical_shock_speed:.2f} m/s")

        # 计算均方根误差
        pos_pred_theory = shock_positions[0] + theoretical_shock_speed * (shock_times - shock_times[0])
        rmse = np.sqrt(np.mean((shock_positions - pos_pred_theory) ** 2))
        ax.text(0.02, 0.98,
                f"RMSE = {rmse:.2f} m\nSpeed error = {abs(num_speed - theoretical_shock_speed):.3f} m/s",
                transform=ax.transAxes, verticalalignment="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Shock Position x (m)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, filename)


# ============================================================
# 6. 格式对比图
# ============================================================

def plot_scheme_comparison(results: dict, x: NDArray[np.float64],
                           model: LWRModel, t_snapshot: float = 120.0,
                           filename: str = "scheme_comparison"):
    """对比不同数值格式在同一时刻的密度剖面。

    Parameters
    ----------
    results : dict, {scheme_name: (t_array, rho_history)}
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for ax, (name, (t_arr, rho_hist)) in zip(axes_flat, results.items()):
        # 找到最接近 t_snapshot 的时刻
        idx = np.argmin(np.abs(t_arr - t_snapshot))
        t_actual = t_arr[idx]

        ax.plot(x, rho_hist[idx] * 1000, "b-", linewidth=1.5)
        ax.axhline(y=model.rho_c * 1000, color="red", linestyle="--",
                   alpha=0.5)
        ax.fill_between(x, 0, rho_hist[idx] * 1000, alpha=0.15, color="blue")
        ax.set_title(f"{name}  (t = {t_actual:.0f} s)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("ρ (veh/km)")
        ax.set_ylim(0, model.rho_max * 1000 * 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Scheme Comparison at t ≈ {t_snapshot:.0f}s",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, filename)


# ============================================================
# 7. 收敛性分析图
# ============================================================

def plot_convergence(dx_values: list, errors: dict, filename: str = "convergence"):
    """绘制 L1 误差 vs dx 的对数图，验证收敛阶。

    Parameters
    ----------
    dx_values : list of float
    errors : dict, {scheme_name: [err_N1, err_N2, ...]}
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    markers = {"Godunov": "o", "Lax-Friedrichs": "s", "Upwind": "^",
               "MacCormack": "D"}
    colors = {"Godunov": "C0", "Lax-Friedrichs": "C1", "Upwind": "C2",
              "MacCormack": "C3"}

    for name, err_list in errors.items():
        marker = markers.get(name, "x")
        color = colors.get(name, "gray")
        ax.loglog(dx_values, err_list, marker=marker, color=color,
                  linewidth=1.5, markersize=7, label=name)

        # 线性拟合求收敛阶
        if len(err_list) >= 2:
            coeffs = np.polyfit(np.log(dx_values), np.log(err_list), 1)
            order = coeffs[0]
            ax.loglog(dx_values, np.exp(coeffs[1]) * np.array(dx_values) ** order,
                      "--", color=color, alpha=0.4, linewidth=1)

            # 在线上标注
            mid_idx = len(dx_values) // 2
            ax.annotate(f"O(h^{order:.2f})",
                        (dx_values[mid_idx], err_list[mid_idx]),
                        textcoords="offset points", xytext=(10, -10),
                        fontsize=8, color=color)

    # 参考线
    ax.plot(dx_values, [dx_values[0] * (dx / dx_values[0]) ** 1 for dx in dx_values],
            "k:", alpha=0.3, label="O(h)")
    ax.plot(dx_values, [dx_values[0] * (dx / dx_values[0]) ** 2 for dx in dx_values],
            "k-.", alpha=0.3, label="O(h²)")

    ax.set_xlabel("Grid spacing Δx (m)")
    ax.set_ylabel("L₁ Error")
    ax.set_title("Convergence Analysis: L₁ Error vs Grid Size", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, filename)


# ============================================================
# 8. CFL 条件验证图
# ============================================================

def plot_cfl_verification(results: dict, x: NDArray[np.float64],
                          model: LWRModel,
                          filename: str = "cfl_verification"):
    """绘制不同 CFL 数下的结果对比（含 CFL>1 发散）。

    Parameters
    ----------
    results : dict, {cfl_label: rho_final_array}
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, (label, rho) in zip(axes, results.items()):
        ax.plot(x, rho * 1000, "b-", linewidth=1.5)
        ax.fill_between(x, 0, rho * 1000, alpha=0.15, color="blue")
        ax.axhline(y=model.rho_c * 1000, color="red", linestyle="--", alpha=0.5)

        # 检查是否数值不稳定（超出范围）
        if np.any(rho < 0) or np.any(rho > model.rho_max):
            ax.set_facecolor("#fff5f5")
            ax.text(0.5, 0.5, "UNSTABLE!", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, color="red",
                    fontweight="bold", alpha=0.3)

        ax.set_title(f"CFL = {label}", fontweight="bold")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("ρ (veh/km)")
        ax.set_ylim(0, model.rho_max * 1000 * 1.2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("CFL Condition Verification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, filename)


# ============================================================
# 9. 单车轨迹图
# ============================================================

def plot_vehicle_trajectories(t_array: NDArray[np.float64],
                              x: NDArray[np.float64],
                              rho_history: NDArray[np.float64],
                              model: LWRModel,
                              filename: str = "trajectories",
                              n_vehicles: int = 15):
    """绘制单车轨迹线（时空图中跟踪车的路径）。

    车辆速度: dx/dt = v(ρ(x,t))
    从不同初始位置出发，用数值积分追踪每辆车的轨迹。
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 选择初始位置
    x0_array = np.linspace(x[0] + 50, x[-1] - 50, n_vehicles)

    # 时间步长（子采样以提高效率）
    dt_sample = t_array[1] - t_array[0]

    colors = plt.cm.plasma(np.linspace(0, 0.9, n_vehicles))

    for x0, c in zip(x0_array, colors):
        traj_t = [0.0]
        traj_x = [x0]
        x_current = x0

        for t_idx in range(1, len(t_array)):
            t_val = t_array[t_idx]
            # 在当前密度场中插值获取速度
            rho_field = rho_history[t_idx]
            idx_interp = np.searchsorted(x, x_current)
            idx_interp = np.clip(idx_interp, 1, len(x) - 1)
            # 线性插值
            if idx_interp < len(x) - 1:
                frac = (x_current - x[idx_interp - 1]) / (x[idx_interp] - x[idx_interp - 1])
                rho_local = (1 - frac) * rho_field[idx_interp - 1] + frac * rho_field[idx_interp]
            else:
                rho_local = rho_field[-1]
            v_local = model.velocity(rho_local)

            # 更新位置
            x_current += v_local * dt_sample
            x_current = np.clip(x_current, x[0], x[-1])

            traj_t.append(t_val)
            traj_x.append(x_current)

        ax.plot(traj_x, traj_t, color=c, linewidth=0.8, alpha=0.8)

    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Time t (s)")
    ax.set_title("Vehicle Trajectories in the (x, t) Plane", fontweight="bold")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(t_array[0], t_array[-1])
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    _save(fig, filename)


# ============================================================
# 10. 扰动增长曲线
# ============================================================

def plot_perturbation_growth(t_array: NDArray[np.float64],
                              rho_history: NDArray[np.float64],
                              model: LWRModel,
                              filename: str = "perturbation_growth"):
    """绘制扰动幅度随时间增长曲线。

    扰动幅度 = max(ρ) - min(ρ) 或标准差。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    rho_min = np.min(rho_history, axis=1) * 1000
    rho_max = np.max(rho_history, axis=1) * 1000
    rho_std = np.std(rho_history, axis=1) * 1000
    amplitude = rho_max - rho_min

    ax1.plot(t_array, rho_max, "r-", linewidth=1.5, label="max ρ")
    ax1.plot(t_array, rho_min, "b-", linewidth=1.5, label="min ρ")
    ax1.fill_between(t_array, rho_min, rho_max, alpha=0.2)
    ax1.set_xlabel("Time t (s)")
    ax1.set_ylabel("Density ρ (veh/km)")
    ax1.set_title("Density Range Evolution", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_array, amplitude, "r-", linewidth=2, label="Amplitude (max-min)")
    ax2.plot(t_array, rho_std, "b--", linewidth=1.5, label="Std deviation")
    ax2.set_xlabel("Time t (s)")
    ax2.set_ylabel("Amplitude / Std (veh/km)")
    ax2.set_title("Perturbation Growth", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Ghost Jam: Perturbation Amplification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, filename)


# ============================================================
# 11. 拥堵长度分析
# ============================================================

def plot_congestion_length(t_array: NDArray[np.float64],
                           rho_history: NDArray[np.float64],
                           x: NDArray[np.float64],
                           model: LWRModel,
                           filename: str = "congestion_length"):
    """绘制拥堵长度（ρ>ρ_c 的路段长度）随时间变化。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    congested_length = np.array([
        np.sum(rho_hist > model.rho_c) * (x[1] - x[0])
        for rho_hist in rho_history
    ])

    ax.plot(t_array, congested_length, "r-", linewidth=2)
    ax.fill_between(t_array, 0, congested_length, alpha=0.2, color="red")
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Congested Length (m)")
    ax.set_title("Congested Road Length (ρ > ρc) vs Time", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 标注最大拥堵长度
    max_idx = np.argmax(congested_length)
    ax.annotate(f"Max: {congested_length[max_idx]:.0f}m at t={t_array[max_idx]:.0f}s",
                (t_array[max_idx], congested_length[max_idx]),
                textcoords="offset points", xytext=(0, 15),
                ha="center", fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.5))

    plt.tight_layout()
    _save(fig, filename)


# ============================================================
# 12. 总质量守恒检查
# ============================================================

def plot_mass_conservation(t_array: NDArray[np.float64],
                           rho_history: NDArray[np.float64],
                           dx: float,
                           filename: str = "mass_conservation"):
    """绘制总质量（总车辆数）随时间的漂移，验证守恒性。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    total_mass = np.sum(rho_history, axis=1) * dx
    initial_mass = total_mass[0]
    drift_percent = (total_mass - initial_mass) / initial_mass * 100

    ax.plot(t_array, drift_percent, "b-", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Mass Drift (%)")
    ax.set_title(f"Mass Conservation Check\nInitial={initial_mass:.1f} veh, "
                 f"Final drift={drift_percent[-1]:.4f}%",
                 fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, filename)


# ============================================================
# 13. 综合场景报告图
# ============================================================

def plot_all_scenarios_summary(results_dict: dict, model: LWRModel,
                               filename: str = "scenarios_summary"):
    """5个场景的x-t热力图汇总在一张大图上。"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes_flat = axes.flatten()

    scenario_names = [
        "ghost_jam", "traffic_light", "slow_vehicle",
        "periodic_light", "onramp"
    ]

    for i, (tag, name) in enumerate(zip(scenario_names,
                                         ["Ghost Jam", "Traffic Light",
                                          "Slow Vehicle", "Periodic Light",
                                          "On-ramp"])):
        if tag in results_dict:
            t_arr, x_arr, rho_hist = results_dict[tag]
            X, T_grid = np.meshgrid(x_arr, t_arr)
            ax = axes_flat[i]
            im = ax.pcolormesh(X, T_grid, rho_hist * 1000,
                               cmap="jet", shading="auto",
                               vmin=0, vmax=model.rho_max * 1000)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("t (s)")
            ax.set_title(name, fontweight="bold")
            plt.colorbar(im, ax=ax, label="ρ (veh/km)")

    # 最后一个子图放参数总结
    ax = axes_flat[-1]
    ax.axis("off")
    summary_text = (
        f"Simulation Parameters\n"
        f"{'─' * 30}\n"
        f"v_max = {model.v_max:.1f} m/s ({model.v_max*3.6:.0f} km/h)\n"
        f"ρ_max = {model.rho_max*1000:.0f} veh/km\n"
        f"ρ_c = {model.rho_c*1000:.0f} veh/km\n"
        f"q_max = {model.q_max*3600:.0f} veh/h\n"
        f"{'─' * 30}\n"
        f"L = 2000 m\n"
        f"T = 120-300 s\n"
        f"Nx = 400, Δx = 5 m\n"
        f"CFL = 0.8\n"
        f"Scheme: Godunov"
    )
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=12, ha="center", va="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))

    fig.suptitle("LWR Traffic Flow Model — All Scenarios Summary",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    _save(fig, filename)
