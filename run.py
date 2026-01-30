# -*- coding: utf-8 -*-
"""
run.py
------
统一入口（直接在文件里改参数，然后点运行）：
- 支持 CCG (Column and Constraint Generation) 方法
- 支持 Benders (Benders Decomposition) 方法

使用方法:
    1. 在下方 CONFIG 中修改 mm_path 和 method
    2. 直接运行本文件
"""

import time
import json
import sys
import csv
from pathlib import Path

# 导入项目模块
import ccg
import benders
import mrcpsp
import translate
from generate_mode_meta import generate_mode_meta_from_mm


# =========================================================
# ✅ 在这里直接改参数（然后点运行）
# =========================================================

# 选择求解方法: "ccg" 或 "benders"
METHOD = "ccg"

# PSPLIB .mm/.bas 文件路径（相对/绝对均可）
mm_path = r'E:\github\mrcpsp-ddu\instances\j10.mm\j102_2.mm'

# 生成模式元数据（自动根据 mm_path 生成）
mode_meta_csv, deviations, cost = generate_mode_meta_from_mm(mm_path, seed=42)

CONFIG = {
    # PSPLIB .mm/.bas 文件路径
    "mm_path": mm_path,

    # DDU 预算不确定集参数
    "Gamma":7,

    # 工期货币化成本 e
    "e_overhead": 100,

    # （可选）toy接口：若提供该CSV，则直接读取指定的 u_abs(偏离上界) 与 cost(模式成本)
    # 若为 None，则保持原逻辑：u_min/u_max 随机生成偏离，cost 按资源工时价计算
    "mode_meta_csv": mode_meta_csv,

    # 用于生成 u 的相对范围（内部会转为绝对偏差：u_abs = bar_d * u_rel）
    "u_max": 0.5,
    "u_min": 0.2,

    # 时间约束的 Big-M（None 表示自动按论文公式生成）
    "M_big": None,

    # CCG 求解参数
    "tol": 0.01,
    "max_iter": 50,
    "alpha_max": 1e4,

    # Benders 求解参数
    "time_limit": 6000,  # Benders 的求解时间限制（秒）

    # 是否启用资源流 f（建议 True，仅 CCG 使用）
    "use_flow": True,

    # 是否打印 Gurobi 日志
    "verbose": True,

    # 可选：把 translate 的解析结果保存为 json（None 表示不保存）
    # 也支持传目录：r"out_json/"，会自动用 mm 同名 json
    "json_out": None,
}


# =========================================================
# CCG 求解方法
# =========================================================
def run_ccg():
    """执行 CCG 方法"""
    print("\n" + "=" * 50)
    print("Running CCG Method")
    print("=" * 50)
    t0_total = time.perf_counter()

    # 0) 读取配置
    mm_path = Path(CONFIG["mm_path"])
    Gamma = float(CONFIG["Gamma"])
    e_overhead = float(CONFIG["e_overhead"])
    u_max = float(CONFIG["u_max"])
    u_min = float(CONFIG["u_min"])
    M_big = CONFIG["M_big"]
    tol = float(CONFIG["tol"])
    max_iter = int(CONFIG["max_iter"])
    alpha_max = float(CONFIG["alpha_max"])
    use_flow = bool(CONFIG["use_flow"])
    verbose = bool(CONFIG["verbose"])
    json_out = CONFIG["json_out"]
    mode_meta_csv = CONFIG.get("mode_meta_csv", None)

    if not mm_path.exists():
        raise FileNotFoundError(f"mm_path 不存在：{mm_path.resolve()}")

    # toy接口：CSV 存在则启用覆盖（u_abs 与 cost）
    if mode_meta_csv is not None:
        mode_meta_csv = str(Path(mode_meta_csv))
        if not Path(mode_meta_csv).exists():
            raise FileNotFoundError(f"mode_meta_csv 不存在：{Path(mode_meta_csv).resolve()}")

    # 1) 解析 mm
    t0 = time.perf_counter()
    data = translate.parse_psplib_mm(str(mm_path))
    t_parse = time.perf_counter() - t0

    # 2) 可选：保存 json
    t0 = time.perf_counter()
    if json_out is not None:
        out = Path(json_out)
        if out.is_dir():
            out = out / (mm_path.stem + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] saved json -> {out.resolve()}")
    t_save = time.perf_counter() - t0

    # 3) 构造实例 + 直接导入 ccg
    t0 = time.perf_counter()
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    inst = ccg.build_instance_from_psplib_json(
        data=data,
        Gamma=Gamma,
        e_overhead=e_overhead,
        b_price=None,
        u_max=u_max,
        u_min=u_min,
        M_big=M_big,
        use_flow=use_flow,
        mode_meta_csv=mode_meta_csv,
    )
    solver = ccg.Variant2Solver(
        inst=inst,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
        alpha_max=alpha_max,
    )
    t_build = time.perf_counter() - t0

    # 4) 求解
    t0 = time.perf_counter()
    sol = solver.run()
    t_solve = time.perf_counter() - t0

    x_sol, y_sol, eta_sol, LB, UB = sol

    # 4.1) 输出最差情形不确定参数 d*（来自最后一次 SP2）
    wc = getattr(solver, "last_worst_case", None)
    if wc is not None:
        print("\n[Worst-case uncertainty (d*) under the final solution]")
        print(f"budget_used = {wc['budget_used']:.6f} (Gamma={inst.Gamma})")
        print(f"path_nodes  = {wc['path_nodes']}")
        print("i\tmode\tbar\tu_abs\td*\txi")
        rows = []
        for i in solver.real:
            mm = None
            for m_ in inst.modes[i].keys():
                if x_sol.get((i, m_), 0) == 1:
                    mm = m_
                    break
            if mm is None:
                mm = next(iter(inst.modes[i].keys()))
            bar_i = inst.modes[i][mm].bar_d
            u_i = inst.modes[i][mm].u
            d_i = wc["d"].get(i, bar_i)
            xi_i = wc["xi"].get(i, 0.0)
            if xi_i > 1e-9:
                rows.append((i, mm, bar_i, u_i, d_i, xi_i))
        rows.sort(key=lambda t: -t[5])
        if len(rows) == 0:
            print("(no deviations: all d_i = bar_i)")
        else:
            for (i, mm, bar_i, u_i, d_i, xi_i) in rows:
                print(f"{i}\t{mm}\t{bar_i:.6f}\t{u_i:.6f}\t{d_i:.6f}\t{xi_i:.6f}")

    # 4.2) 保存最差情形甘特图
    try:
        script_dir = Path(__file__).resolve().parent
        out_jpg_path = script_dir / f"worstcase_gantt_{mm_path.stem}.jpg"
        out_jpg = solver.save_worstcase_gantt(str(out_jpg_path))
        print(f"[OK] saved worst-case gantt -> {Path(out_jpg).resolve()}")
    except Exception as _e:
        print(f"[ERROR] gantt not saved: {_e}")
        print("常见原因：①未安装 matplotlib；②当前目录无写权限；③last_worst_case/best_solution 为空。")

    t_total = time.perf_counter() - t0_total

    # 5) 输出
    print("\n=== CCG Done ===")
    print(f"mm        = {mm_path.name}")
    print(f"Gamma     = {Gamma}")
    print(f"mode_meta = {mode_meta_csv}")
    print(f"eta       = {eta_sol}")
    print(f"LB        = {LB}")
    print(f"UB        = {UB}")

    print("\n[Time]")
    print(f"parse mm  = {t_parse:.6f} s")
    print(f"save json = {t_save:.6f} s")
    print(f"build     = {t_build:.6f} s")
    print(f"solve     = {t_solve:.6f} s")
    print(f"total     = {t_total:.6f} s")

    print("\n[Summary] selected modes:")
    for (i, m), v in sorted(x_sol.items()):
        if v == 1:
            print(f"  activity {i}: mode {m}")

    print("\n[Summary] activated arcs (sample):")
    cnt = 0
    for (i, j), v in sorted(y_sol.items()):
        if v == 1:
            print(f"  y[{i},{j}] = 1")
            cnt += 1
            if cnt >= 30:
                print("  ... (truncated)")
                break


# =========================================================
# Benders 求解方法
# =========================================================
def run_benders():
    """执行 Benders 方法"""
    print("\n" + "=" * 50)
    print("Running Benders Method")
    print("=" * 50)

    # 0) 读取配置
    mm_path_str = CONFIG["mm_path"]
    mm_path = Path(mm_path_str)
    Gamma = int(CONFIG["Gamma"])
    time_limit = int(CONFIG["time_limit"])
    e_over = float(CONFIG["e_overhead"])
    verbose = bool(CONFIG["verbose"])
    csv_path = CONFIG.get("mode_meta_csv", None)

    if not mm_path.exists():
        raise FileNotFoundError(f"mm_path 不存在：{mm_path.resolve()}")

    # 1) 从 CSV 读取 deviations 和 cost，jobnr 减 1
    deviations = {}
    cost = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        current_job = None
        job_costs = []
        for row in reader:
            jobnr = int(row['jobnr']) - 1  # jobnr 减 1
            if jobnr not in deviations:
                deviations[jobnr] = []
            deviations[jobnr].append(int(row['u_abs']))

            if current_job != jobnr:
                if job_costs:
                    cost.append(job_costs)
                job_costs = []
                current_job = jobnr
            job_costs.append(int(row['cost']))
        if job_costs:
            cost.append(job_costs)

    # 2) 加载实例
    instance = mrcpsp.load_nominal_mrcpsp(mm_path_str)
    instance.set_dbar_explicitly(deviations)

    # 3) 求解
    print(f"Solving {instance.name} using Benders' decomposition:")
    print('"""""""""""""""""""""""""""""""""""""""""""""\n')

    benders_sol = benders.Benders(instance, Gamma, time_limit, cost=cost, e_over=e_over).solve(print_log=verbose)

    # 4) 输出结果
    print("\n=== Benders Done ===")
    print(f"mm        = {mm_path.name}")
    print(f"Gamma     = {Gamma}")
    print("objval:", benders_sol['objval'])
    print("runtime:", benders_sol['runtime'])
    print("n_iterations:", benders_sol['n_iterations'])
    print("modes:", benders_sol['modes'])
    print("network:", benders_sol['network'])
    print("resource flows:", benders_sol['flows'])


# =========================================================
# 主函数
# =========================================================
def main():
    """根据 METHOD 选择执行 CCG 或 Benders"""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    if METHOD.lower() == "ccg":
        run_ccg()
    elif METHOD.lower() == "benders":
        run_benders()
    else:
        print(f"Error: Unknown method '{METHOD}'")
        print("Please set METHOD to 'ccg' or 'benders'")
        sys.exit(1)


if __name__ == "__main__":
    main()