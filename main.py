#!/usr/bin/env python3
"""
LWR 交通流激波模型 — 主程序
============================
运行所有仿真场景，生成可视化图像和数据分析表格。
"""

import os
import sys
import time
import io
import numpy as np

# 修复 Windows GBK 编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(__file__))

from lwr_model import LWRModel
from solvers import (
    SOLVERS, SOLVER_INFO,
    solve_periodic_light, solve_onramp,
)
from scenarios import SCENARIOS
from visualization import (
    plot_flux_function, plot_xt_heatmap, plot_characteristics,
    plot_density_profiles, plot_wave_speed_comparison,
    plot_scheme_comparison, plot_convergence, plot_cfl_verification,
    plot_vehicle_trajectories, plot_perturbation_growth,
    plot_congestion_length, plot_mass_conservation,
    plot_all_scenarios_summary,
)
from analysis import (
    convergence_analysis, extract_shock_speed, extract_rarefaction_speeds,
    check_mass_conservation,
    table_model_parameters, table_scheme_info, table_wave_speed,
    table_convergence, table_cfl_results, table_timing, table_congestion,
)


def print_banner():
    print("\n" + "█" * 70)
    print("█  LWR Traffic Flow Model — Numerical Simulation Suite")
    print("█  Lighthill-Whitham-Richards (LWR) 交通流激波模型")
    print("█  工程数值分析课程个人作业")
    print("█" + "─" * 70)


# ============================================================
# 主程序
# ============================================================

def main():
    print_banner()

    # ---- 初始化模型 ----
    model = LWRModel(v_max=33.33, rho_max=0.150)
    model.print_summary()

    # ---- 生成表格 ----
    print("\n" + "=" * 60)
    print("  Generating Tables")
    print("=" * 60)
    table_model_parameters(model)
    table_scheme_info()

    # ---- 图 1: 通量函数 ----
    print("\n" + "=" * 60)
    print("  Plot: Flux Function & Characteristic Speed")
    print("=" * 60)
    plot_flux_function(model)

    # ---- 运行所有场景 ----
    all_results = {}

    # ========== 场景 1: 幽灵堵车 ==========
    print("\n" + "=" * 60)
    print("  Scenario 1: Ghost Jam")
    print("=" * 60)
    s1 = SCENARIOS["ghost_jam"](model)
    t1, rho1 = SOLVERS["Godunov"](model, s1["x"], s1["t_end"], s1["rho0"])
    all_results["ghost_jam"] = (t1, s1["x"], rho1)

    # 可视化
    plot_xt_heatmap(t1, s1["x"], rho1, model,
                    "Scenario 1: Ghost Jam — Formation of Shock Wave\n"
                    "ρ₀=0.30ρmax + Gaussian perturbation",
                    "scenario1_ghost_jam_xt")
    plot_characteristics(model, s1["x"], s1["t_end"], s1["rho0"],
                         "scenario1_characteristics")
    plot_density_profiles(t1, s1["x"], rho1, model,
                          "Scenario 1: Density Profiles at Different Times",
                          "scenario1_density_profiles")
    plot_perturbation_growth(t1, rho1, model,
                             "scenario1_perturbation_growth")

    # 波速分析
    s1_shock = extract_shock_speed(t1, s1["x"], rho1, model)
    print(f"  Shock speed: num={s1_shock[0]:.3f}, theory={s1_shock[1]:.3f} m/s, "
          f"R²={s1_shock[2]:.4f}")
    plot_wave_speed_comparison(model, t1, s1["x"], rho1,
                               theoretical_shock_speed=s1_shock[1],
                               title="Scenario 1: Numerical vs Theoretical Shock Speed",
                               filename="scenario1_wave_speed")

    # 质量守恒
    s1_mass = check_mass_conservation(t1, rho1, s1["dx"])
    print(f"  Mass conservation: drift={s1_mass['final_drift_percent']:.4f}%, "
          f"{'PASS' if s1_mass['passed'] else 'FAIL'}")

    # 波速表
    table_wave_speed({
        "num_speed": s1_shock[0], "theory_speed": s1_shock[1],
        "r_squared": s1_shock[2],
    })

    # ========== 场景 2: 红绿灯 ==========
    print("\n" + "=" * 60)
    print("  Scenario 2: Traffic Light")
    print("=" * 60)
    s2 = SCENARIOS["traffic_light"](model)
    t2, rho2 = SOLVERS["Godunov"](model, s2["x"], s2["t_end"], s2["rho0"])
    all_results["traffic_light"] = (t2, s2["x"], rho2)

    plot_xt_heatmap(t2, s2["x"], rho2, model,
                    "Scenario 2: Traffic Light — Rarefaction Wave (Green Light)\n"
                    "Initial: ρ_left=0.85ρmax (queue), ρ_right=0.05ρmax (empty)",
                    "scenario2_traffic_light_xt")
    plot_density_profiles(t2, s2["x"], rho2, model,
                          "Scenario 2: Density Profiles (Rarefaction Wave)",
                          "scenario2_density_profiles")

    # 稀疏波波速分析
    # 初始条件: x<light → ρ=0.85ρmax (高密度), x>light → ρ=0.05ρmax (低密度)
    # 稀疏波从高密度向低密度扩展
    rho_high = 0.85 * model.rho_max  # 左侧（红灯排队）
    rho_low = 0.05 * model.rho_max   # 右侧（前方空旷）
    s2_rarefaction = extract_rarefaction_speeds(
        t2, s2["x"], rho2, model, rho_high, rho_low,
        discontinuity_x=s2["light_position"]
    )
    print(f"  Rarefaction head: num={s2_rarefaction['head_speed_num']:.3f}, "
          f"theory={s2_rarefaction['head_speed_theory']:.3f} m/s")
    print(f"  Rarefaction tail: num={s2_rarefaction['tail_speed_num']:.3f}, "
          f"theory={s2_rarefaction['tail_speed_theory']:.3f} m/s")
    table_wave_speed({}, s2_rarefaction)

    plot_mass_conservation(t2, rho2, s2["dx"],
                           "scenario2_mass_conservation")

    # ========== 场景 3: 慢车效应 ==========
    print("\n" + "=" * 60)
    print("  Scenario 3: Slow Vehicle")
    print("=" * 60)
    s3 = SCENARIOS["slow_vehicle"](model)
    t3, rho3 = SOLVERS["Godunov"](model, s3["x"], s3["t_end"], s3["rho0"])
    all_results["slow_vehicle"] = (t3, s3["x"], rho3)

    plot_xt_heatmap(t3, s3["x"], rho3, model,
                    "Scenario 3: Slow Vehicle — Moving Bottleneck\n"
                    "Slow vehicle creates backward-propagating shock wave",
                    "scenario3_slow_vehicle")
    plot_vehicle_trajectories(t3, s3["x"], rho3, model,
                              "scenario3_trajectories")

    # ========== 场景 4: 周期红绿灯 ==========
    print("\n" + "=" * 60)
    print("  Scenario 4: Periodic Traffic Light")
    print("=" * 60)
    s4 = SCENARIOS["periodic_light"](model)
    t4, rho4 = solve_periodic_light(
        model, s4["x"], s4["t_end"], s4["rho0"],
        light_position=s4["light_position"],
        period=s4["period"],
        green_fraction=s4["green_fraction"],
    )
    all_results["periodic_light"] = (t4, s4["x"], rho4)

    plot_xt_heatmap(t4, s4["x"], rho4, model,
                    "Scenario 4: Periodic Traffic Light\n"
                    f"Period={s4['period']}s, Green ratio={s4['green_fraction']}",
                    "scenario4_periodic")
    plot_congestion_length(t4, rho4, s4["x"], model,
                           "scenario4_congestion_length")

    # ========== 场景 5: 匝道汇入 ==========
    print("\n" + "=" * 60)
    print("  Scenario 5: On-ramp Merge")
    print("=" * 60)
    s5 = SCENARIOS["onramp"](model)
    t5, rho5 = solve_onramp(
        model, s5["x"], s5["t_end"], s5["rho0"],
        ramp_position=s5["ramp_position"],
        ramp_flow_rate=s5["ramp_flow_rate"],
    )
    all_results["onramp"] = (t5, s5["x"], rho5)

    plot_xt_heatmap(t5, s5["x"], rho5, model,
                    "Scenario 5: On-ramp Merge — Bottleneck Congestion\n"
                    f"Ramp at x={s5['ramp_position']:.0f}m, "
                    f"flow rate={s5['ramp_flow_rate']:.3f} veh/s",
                    "scenario5_onramp")
    plot_congestion_length(t5, rho5, s5["x"], model,
                           "scenario5_congestion_length")

    # ---- 匝道汇入率 vs 拥堵长度 ----
    print("\n  On-ramp flow rate vs congestion analysis...")
    congestion_data = {}
    for ramp_rate in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        s5_test = SCENARIOS["onramp"](model, ramp_flow_rate=ramp_rate)
        _, rho_test = solve_onramp(
            model, s5_test["x"], s5_test["t_end"], s5_test["rho0"],
            ramp_position=s5_test["ramp_position"],
            ramp_flow_rate=ramp_rate,
            progress=False,
        )
        cong_len = np.sum(rho_test[-1] > model.rho_c) * s5_test["dx"]
        congestion_data[ramp_rate] = cong_len
        print(f"    rate={ramp_rate:.3f} → max congestion length={cong_len:.0f}m")
    table_congestion(congestion_data)

    # ========== 格式对比（场景1） ==========
    print("\n" + "=" * 60)
    print("  Scheme Comparison (Scenario 1, all schemes)")
    print("=" * 60)
    scheme_results = {}
    timing_results = {}
    for name in ["Godunov", "Lax-Friedrichs", "Upwind", "MacCormack"]:
        solver = SOLVERS[name]
        t_start = time.perf_counter()
        t_arr, rho_arr = solver(model, s1["x"], s1["t_end"], s1["rho0"])
        t_elapsed = time.perf_counter() - t_start
        scheme_results[name] = (t_arr, rho_arr)
        timing_results[name] = t_elapsed
        print(f"  {name}: {t_elapsed:.4f}s")

    plot_scheme_comparison(scheme_results, s1["x"], model,
                           t_snapshot=s1["t_end"] * 0.5,
                           filename="scheme_comparison")
    table_timing(timing_results)

    # ========== 收敛性分析 ==========
    print("\n" + "=" * 60)
    print("  Convergence Analysis")
    print("=" * 60)

    def rho0_factory_scenario1(Nx: int):
        """场景1初始条件的工厂函数（用于不同Nx）。"""
        L_conv = 2000.0
        x_conv = np.linspace(0, L_conv, Nx)
        rho_base = np.full(Nx, 0.30 * model.rho_max)
        pert = 0.05 * model.rho_max * np.exp(
            -0.5 * ((x_conv - 500) / 30) ** 2
        )
        return np.clip(rho_base + pert, 0.01 * model.rho_max, 0.99 * model.rho_max)

    dx_test = [20.0, 10.0, 6.67, 5.0]
    conv_results = convergence_analysis(
        model, s1["x"], rho0_factory_scenario1,
        t_end=80.0, dx_values=dx_test,
        scheme_names=["Godunov", "Lax-Friedrichs", "Upwind", "MacCormack"],
    )
    plot_convergence(conv_results["dx_values"],
                     conv_results["errors"],
                     filename="convergence")
    table_convergence(conv_results["dx_values"], conv_results["errors"])

    # ========== CFL 验证 ==========
    print("\n" + "=" * 60)
    print("  CFL Verification")
    print("=" * 60)
    cfl_results = {}
    # 使用场景2（红绿灯）做CFL验证
    # 用不同的Δt/Δx比值展示稳定性变化
    for cfl_val in [0.5, 0.9, 2.5, 5.0]:
        dx = s2["dx"]
        dt_forced = cfl_val * dx / model.max_char_speed()
        Nt = int(np.ceil(60.0 / dt_forced))

        rho = s2["rho0"].copy()
        stable = True
        from solvers import godunov_step
        try:
            for _ in range(Nt):
                rho = godunov_step(model, rho, dx, dt_forced)
                if np.any(np.isnan(rho)) or np.any(np.isinf(rho)) or np.any(rho < -1e-6) or np.any(rho > model.rho_max * 1.1):
                    stable = False
                    break
        except Exception as e:
            stable = False

        cfl_results[str(cfl_val)] = {
            "stable": "Yes" if stable else "No (Blow-up!)",
            "max_val": float(np.max(rho)),
            "min_val": float(np.min(rho)),
        }
        print(f"  CFL={cfl_val}: {'STABLE' if stable else 'UNSTABLE!'} "
              f"ρ∈[{np.min(rho)*1000:.0f}, {np.max(rho)*1000:.0f}]")

    # 为可视化准备CFL数据（用于plot）
    cfl_plot_data = {}
    for cfl_val in [0.5, 0.9, 2.5, 5.0]:
        dx = s2["dx"]
        dt_forced = cfl_val * dx / model.max_char_speed()
        Nt = int(np.ceil(60.0 / dt_forced))
        rho = s2["rho0"].copy()
        from solvers import godunov_step
        for _ in range(Nt):
            rho = godunov_step(model, rho, dx, dt_forced)
        cfl_plot_data[f"{cfl_val}"] = rho

    plot_cfl_verification(cfl_plot_data, s2["x"], model,
                          filename="cfl_verification")
    table_cfl_results(cfl_results)

    # ========== 综合汇总图 ==========
    print("\n" + "=" * 60)
    print("  Summary Plot")
    print("=" * 60)
    plot_all_scenarios_summary(all_results, model, filename="scenarios_summary")

    # ========== 完成 ==========
    print("\n" + "█" * 70)
    print("█  All simulations complete!")
    print("█")
    print("█  Output figures:  output/figures/")
    print("█  Output tables:   output/tables/")
    print("█" * 70)
    print()

    # 列出所有输出文件
    fig_dir = os.path.join(os.path.dirname(__file__), "output", "figures")
    tab_dir = os.path.join(os.path.dirname(__file__), "output", "tables")

    print("Figures generated:")
    for f in sorted(os.listdir(fig_dir)):
        print(f"  - {f}")

    print("\nTables generated:")
    for f in sorted(os.listdir(tab_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
