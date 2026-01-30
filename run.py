#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py - 统一入口脚本
=====================

用于求解带决策依赖不确定性的多模式资源受限项目调度问题 (MRCPSP-DDU)。

支持两种求解算法：
1. CCG (Column-and-Constraint Generation) 算法
2. Benders 分解算法

用法示例：
    # 使用 CCG 算法求解
    python run.py --input mrcpsp_toy_example.mm --algorithm ccg --gamma 2

    # 使用 Benders 算法求解
    python run.py --input mrcpsp_toy_example.mm --algorithm benders --gamma 2

    # 使用 CSV 文件指定 cost 和 deviations
    python run.py --input mrcpsp_toy_example.mm --algorithm benders --gamma 2 --params-csv params.csv

CSV 文件格式（params.csv）：
    job,mode,cost,deviation
    0,0,0,0
    1,0,10,1
    1,1,20,2
    ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# 导入项目模块
from translate import parse_psplib_mm, save_json
import mrcpsp
from benders import Benders
from ccg import build_instance_from_psplib_json, Variant2Solver


def load_params_csv(csv_path: str) -> Tuple[Dict[int, List[float]], Dict[int, List[int]]]:
    """
    从 CSV 文件加载 cost 和 deviation (u_abs) 参数。
    
    CSV 格式要求（与 mrcpsp_toy_mode_meta.csv 保持一致）：
    - 必须包含列: jobnr, mode, cost, u_abs (或 maxT-barT 作为 deviation)
    - jobnr: 任务编号（从 1 开始，1 是 dummy source，n+2 是 dummy sink）
    - mode: 模式编号（从 1 开始）
    - cost: 该模式的成本
    - u_abs: 该模式的最大工期偏差（绝对值）
    
    :param csv_path: CSV 文件路径
    :return: (cost_dict, deviation_dict) 两个字典，键为 0-based job 索引
    """
    cost_dict: Dict[int, List[float]] = {}
    deviation_dict: Dict[int, List[int]] = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        
        # 检查必需的列（支持两种格式）
        # 格式 1: jobnr, mode, cost, u_abs (用户原有格式)
        # 格式 2: job, mode, cost, deviation (新格式)
        use_original_format = 'jobnr' in fieldnames
        
        if use_original_format:
            required_cols = {'jobnr', 'mode', 'cost', 'u_abs'}
            if not required_cols.issubset(fieldnames):
                raise ValueError(f"CSV 文件必须包含以下列: {required_cols}")
        else:
            required_cols = {'job', 'mode', 'cost', 'deviation'}
            if not required_cols.issubset(fieldnames):
                raise ValueError(f"CSV 文件必须包含以下列: {required_cols} 或 {{'jobnr', 'mode', 'cost', 'u_abs'}}")
        
        for row in reader:
            if use_original_format:
                # 原有格式: jobnr 从 1 开始, mode 从 1 开始
                jobnr = int(row['jobnr'])
                mode = int(row['mode'])
                cost = float(row['cost'])
                deviation = int(row['u_abs'])
                # 转换为 0-based 索引
                job = jobnr - 1
                mode_idx = mode - 1
            else:
                # 新格式: job 从 0 开始, mode 从 0 开始
                job = int(row['job'])
                mode_idx = int(row['mode'])
                cost = float(row['cost'])
                deviation = int(row['deviation'])
            
            # 初始化列表
            if job not in cost_dict:
                cost_dict[job] = []
                deviation_dict[job] = []
            
            # 确保模式按顺序添加
            while len(cost_dict[job]) <= mode_idx:
                cost_dict[job].append(0.0)
                deviation_dict[job].append(0)
            
            cost_dict[job][mode_idx] = cost
            deviation_dict[job][mode_idx] = deviation
    
    return cost_dict, deviation_dict


def dict_to_list(d: Dict[int, List], n_jobs: int) -> List[List]:
    """
    将字典格式转换为列表格式。
    
    :param d: 字典 {job_id: [values]}
    :param n_jobs: 任务总数
    :return: 列表 [[values], ...]
    """
    result = []
    for j in range(n_jobs):
        if j in d:
            result.append(d[j])
        else:
            result.append([0])  # 默认值
    return result


def solve_with_benders(
    mm_file: str,
    gamma: int,
    time_limit: int,
    uncertainty_level: float = 0.7,
    e_over: float = 1.0,
    cost: Optional[list] = None,
    deviations: Optional[dict] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    使用 Benders 分解算法求解 MRCPSP-DDU 问题。

    :param mm_file: PSPLIB .mm 文件路径
    :param gamma: 鲁棒性参数 Gamma
    :param time_limit: 时间限制（秒）
    :param uncertainty_level: 不确定性水平（默认 0.7，仅在未提供 deviations 时使用）
    :param e_over: 工期惩罚系数
    :param cost: 模式成本（列表格式）
    :param deviations: 工期偏差（字典格式）
    :param verbose: 是否打印详细日志
    :return: 求解结果字典
    """
    # 加载实例
    instance = mrcpsp.load_nominal_mrcpsp(mm_file)
    
    # 设置工期偏差
    if deviations:
        instance.set_dbar_explicitly(deviations)
    else:
        instance.set_dbar_uncertainty_level(uncertainty_level)
    
    # 如果未提供 cost，则自动生成（全零成本）
    if cost is None:
        cost = [[0 for _ in instance.jobs[j].M] for j in instance.V]
    
    # 创建并运行 Benders 求解器
    solver = Benders(instance, gamma, time_limit, cost=cost, e_over=e_over)
    solution = solver.solve(print_log=verbose)
    
    return {
        "algorithm": "benders",
        "instance": instance.name,
        "gamma": gamma,
        "objective_value": solution['objval'],
        "objective_bound": solution['objbound'],
        "runtime": solution['runtime'],
        "iterations": solution['n_iterations'],
        "avg_iteration_time": solution['avg_iteration'],
        "modes": solution['modes'],
        "network": solution['network'],
        "flows": solution['flows']
    }


def solve_with_ccg(
    mm_file: str,
    gamma: float,
    e_overhead: float = 1.0,
    b_price: Optional[Dict[int, float]] = None,
    mode_meta_csv: Optional[str] = None,
    max_iter: int = 50,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    使用 CCG (Column-and-Constraint Generation) 算法求解 MRCPSP-DDU 问题。

    :param mm_file: PSPLIB .mm 文件路径
    :param gamma: 鲁棒性参数 Gamma
    :param e_overhead: 工期惩罚系数
    :param b_price: 资源单价（可选）
    :param mode_meta_csv: 模式元数据 CSV 文件（可选）
    :param max_iter: 最大迭代次数
    :param verbose: 是否打印详细日志
    :return: 求解结果字典
    """
    # 解析 .mm 文件为 JSON 格式
    data = parse_psplib_mm(mm_file)
    
    # 设置默认资源单价
    if b_price is None:
        n_renew = data.get("n_renew", 1)
        b_price = {k: 1.0 for k in range(1, n_renew + 1)}
    
    # 构建 DDU 实例
    inst = build_instance_from_psplib_json(
        data=data,
        Gamma=gamma,
        e_overhead=e_overhead,
        b_price=b_price,
        mode_meta_csv=mode_meta_csv,
        use_flow=True
    )
    
    # 创建并运行 CCG 求解器
    solver = Variant2Solver(inst, max_iter=max_iter, verbose=verbose)
    solver.run()
    
    # 提取结果
    best = solver.best_solution
    
    # 转换模式选择格式
    modes = {}
    for (i, m), val in best['x'].items():
        if val == 1:
            modes[i] = m
    
    # 转换网络格式
    network = [(i, j) for (i, j), val in best['y'].items() if val == 1]
    
    return {
        "algorithm": "ccg",
        "instance": os.path.basename(mm_file),
        "gamma": gamma,
        "lower_bound": best['LB'],
        "upper_bound": best['UB'],
        "gap": best['UB'] - best['LB'],
        "eta": best['eta'],
        "modes": modes,
        "network": network,
        "iterations": len(solver.pis) + 1
    }


def print_result(result: Dict[str, Any]) -> None:
    """
    格式化打印求解结果。
    """
    print("\n" + "=" * 60)
    print("求解结果")
    print("=" * 60)
    print(f"算法:           {result['algorithm'].upper()}")
    print(f"实例:           {result['instance']}")
    print(f"Gamma:          {result['gamma']}")
    
    if result['algorithm'] == 'benders':
        print(f"目标值:         {result['objective_value']:.4f}")
        print(f"目标下界:       {result['objective_bound']:.4f}")
        print(f"运行时间:       {result['runtime']:.2f} 秒")
        print(f"迭代次数:       {result['iterations']}")
        print(f"平均迭代时间:   {result['avg_iteration_time']:.4f} 秒")
    else:  # ccg
        print(f"下界 (LB):      {result['lower_bound']:.4f}")
        print(f"上界 (UB):      {result['upper_bound']:.4f}")
        print(f"Gap:            {result['gap']:.6f}")
        print(f"迭代次数:       {result['iterations']}")
    
    print(f"\n模式选择:       {result['modes']}")
    print(f"网络结构:       {result['network'][:10]}{'...' if len(result['network']) > 10 else ''}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="MRCPSP-DDU 统一求解器 - 支持 CCG 和 Benders 算法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py --input mrcpsp_toy_example.mm --algorithm ccg --gamma 2
  python run.py --input mrcpsp_toy_example.mm --algorithm benders --gamma 2 --time-limit 60
  python run.py --input mrcpsp_toy_example.mm --algorithm benders --gamma 2 --params-csv params.csv
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="PSPLIB .mm 文件路径"
    )
    parser.add_argument(
        "--algorithm", "-a",
        required=True,
        choices=["ccg", "benders"],
        help="求解算法: 'ccg' 或 'benders'"
    )
    
    # 通用参数
    parser.add_argument(
        "--gamma", "-g",
        type=float,
        default=2,
        help="鲁棒性参数 Gamma（默认: 2）"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印详细求解日志"
    )
    parser.add_argument(
        "--output", "-o",
        help="输出结果到 JSON 文件（可选）"
    )
    
    # 参数 CSV 文件（用于传入 cost 和 deviation）
    parser.add_argument(
        "--params-csv", "-p",
        help="参数 CSV 文件路径，包含 cost 和 deviation（格式: job,mode,cost,deviation）"
    )
    
    # Benders 特定参数
    parser.add_argument(
        "--time-limit", "-t",
        type=int,
        default=60,
        help="Benders 算法时间限制，单位秒（默认: 60）"
    )
    parser.add_argument(
        "--uncertainty-level", "-u",
        type=float,
        default=0.7,
        help="Benders 算法不确定性水平（默认: 0.7，仅在未提供 params-csv 时使用）"
    )
    
    # CCG 特定参数
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="CCG 算法最大迭代次数（默认: 50）"
    )
    parser.add_argument(
        "--mode-meta",
        help="CCG 算法模式元数据 CSV 文件路径（可选）"
    )
    
    # 目标函数参数
    parser.add_argument(
        "--e-over",
        type=float,
        default=1.0,
        help="工期惩罚系数 e_overhead（默认: 1.0）"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在！", file=sys.stderr)
        sys.exit(1)
    
    # 加载参数 CSV（如果提供）
    cost = None
    deviations = None
    if args.params_csv:
        if not os.path.exists(args.params_csv):
            print(f"错误: 参数 CSV 文件 '{args.params_csv}' 不存在！", file=sys.stderr)
            sys.exit(1)
        
        print(f"正在从 {args.params_csv} 加载 cost 和 deviation 参数...")
        cost_dict, deviation_dict = load_params_csv(args.params_csv)
        
        # 获取任务数量
        instance = mrcpsp.load_nominal_mrcpsp(args.input)
        n_jobs = len(instance.V)
        
        # 转换格式
        cost = dict_to_list(cost_dict, n_jobs)
        deviations = deviation_dict
        
        print(f"  已加载 {len(cost_dict)} 个任务的参数")
    
    print(f"正在使用 {args.algorithm.upper()} 算法求解 {args.input}...")
    print(f"参数: Gamma={args.gamma}, e_over={args.e_over}")
    
    try:
        if args.algorithm == "benders":
            result = solve_with_benders(
                mm_file=args.input,
                gamma=int(args.gamma),
                time_limit=args.time_limit,
                uncertainty_level=args.uncertainty_level,
                e_over=args.e_over,
                cost=cost,
                deviations=deviations,
                verbose=args.verbose
            )
        else:  # ccg
            result = solve_with_ccg(
                mm_file=args.input,
                gamma=args.gamma,
                e_overhead=args.e_over,
                mode_meta_csv=args.mode_meta,
                max_iter=args.max_iter,
                verbose=args.verbose
            )
        
        # 打印结果
        print_result(result)
        
        # 如果指定了输出文件，保存结果
        if args.output:
            # 转换不可序列化的类型
            output_result = result.copy()
            if 'modes' in output_result:
                output_result['modes'] = {str(k): v for k, v in output_result['modes'].items()}
            if 'network' in output_result:
                output_result['network'] = [[i, j] for i, j in output_result['network']]
            if 'flows' in output_result:
                output_result['flows'] = {f"{k[0]},{k[1]}": v for k, v in output_result['flows'].items()}
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_result, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"求解过程中发生错误: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
