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

    # 指定时间限制和详细输出
    python run.py --input mrcpsp_toy_example.mm --algorithm benders --gamma 2 --time-limit 120 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# 导入项目模块
from translate import parse_psplib_mm, save_json
import mrcpsp
from benders import Benders
from ccg import build_instance_from_psplib_json, Variant2Solver


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
    :param uncertainty_level: 不确定性水平（默认 0.7）
    :param e_over: 工期惩罚系数
    :param cost: 模式成本（可选）
    :param deviations: 工期偏差（可选）
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
  python run.py --input mrcpsp_toy_example.mm --algorithm benders --gamma 2 --verbose
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
        help="Benders 算法不确定性水平（默认: 0.7）"
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
