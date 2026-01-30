# -*- coding: utf-8 -*-
"""
ccgrun.py
---------
统一入口（直接在文件里改参数，然后点运行）：
- 解析 PSPLIB .mm/.bas -> dict (translate.parse_psplib_mm)
- （可选）保存 json
- 直接导入 ccg.py（Variant2Solver）求解
- 记录运行时间（解析/建模/求解/总耗时）
"""

from __future__ import annotations

from pathlib import Path
import json
import time


# =========================================================
# ✅ 在这里直接改参数（然后点运行）
# =========================================================
CONFIG = {
    # PSPLIB .mm/.bas 文件路径（相对/绝对均可）
    "mm_path": r"instances/j30.mm/j301_1.mm",

    # DDU 预算不确定集参数
    "Gamma": 2,

    # 工期货币化成本 e
    "e_overhead": 2,

    # （可选）toy接口：若提供该CSV，则直接读取指定的 u_abs(偏离上界) 与 cost(模式成本)
    # 若为 None，则保持原逻辑：u_min/u_max 随机生成偏离，cost 按资源工时价计算
    "mode_meta_csv": None,
    # 用于生成 u 的相对范围（内部会转为绝对偏差：u_abs = bar_d * u_rel）
    "u_max": 0.5,
    "u_min": 0.2,

    # 时间约束的 Big-M（None 表示自动按论文公式生成）
    "M_big": None,

    # CCG 求解参数
    "tol": 1,
    "max_iter": 50,
    "alpha_max": 1e4,

    # 是否启用资源流 f（建议 True）
    "use_flow": True,

    # 是否打印 Gurobi 日志
    "verbose": True,

    # 可选：把 translate 的解析结果保存为 json（None 表示不保存）
    # 也支持传目录：r"out_json/"，会自动用 mm 同名 json
    "json_out": None,
}


def main():
    t0_total = time.perf_counter()

    # 0) 读取配置
    mm_path = Path(CONFIG["mm_path"])
    Gamma = float(CONFIG["Gamma"])
    e_overhead = float(CONFIG["e_overhead"])
    u_max = float(CONFIG["u_max"])
    u_min = float(CONFIG["u_min"])
    M_big = CONFIG["M_big"]  # None or float
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
    import translate  # 你的 translate.py
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
    #    为避免“工作目录不在脚本目录”导致导入失败，这里把脚本目录插到 sys.path 最前面
    t0 = time.perf_counter()
    import sys
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    import ccg as ccg_mod  # ✅ 直接导入 ccg

    inst = ccg_mod.build_instance_from_psplib_json(
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
    solver = ccg_mod.Variant2Solver(
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
            # 找到该活动的选定模式
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

    # 4.2) 保存最差情形甘特图（jpg）,命名为worstcase_gantt_{input_mm}.jpg
    try:
        out_jpg_path = script_dir / f"worstcase_gantt_{mm_path.stem}.jpg"
        out_jpg = solver.save_worstcase_gantt(str(out_jpg_path))
        print(f"[OK] saved worst-case gantt -> {Path(out_jpg).resolve()}")
    except Exception as _e:
        print(f"[ERROR] gantt not saved: {_e}")
        print("常见原因：①未安装 matplotlib；②当前目录无写权限；③last_worst_case/best_solution 为空。")

    t_total = time.perf_counter() - t0_total

    # 5) 输出
    print("\n=== Done ===")
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

    # y 太多，默认只打印前 30 条
    print("\n[Summary] activated arcs (sample):")
    cnt = 0
    for (i, j), v in sorted(y_sol.items()):
        if v == 1:
            print(f"  y[{i},{j}] = 1")
            cnt += 1
            if cnt >= 30:
                print("  ... (truncated)")
                break


if __name__ == "__main__":
    main()